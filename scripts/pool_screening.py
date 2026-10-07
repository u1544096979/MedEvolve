"""患者池筛查：枚举患者、测算生成输入规模、按上下文预算划分运行名单。

用法（环境实例）：
    python scripts/pool_screening.py --config configs/main-pool-physicianbench-template.json \
        --output outputs/pool-screening/physicianbench

用法（源数据离线估算，用于 FHIR-AgentBench）：
    python scripts/pool_screening.py --offline-source /path/to/mimic-iv-fhir-demo/2.1.0/fhir \
        --name FHIR-AgentBench --output outputs/pool-screening/fhir-agentbench

筛查只读患者材料，不写环境数据。名单划分使用固定种子：先按上下文预算筛出可用患者，
再对患者 ID 排序后打乱，前 N 名进入进化组，其余按批大小切分。
"""

import argparse
import asyncio
import collections
import gzip
import json
import random
from pathlib import Path

from contracts import EnvironmentConfig

CHARS_PER_TOKEN = 2.611
CONTEXT_TOKENS = 128000
OUTPUT_TOKENS = 16384
INSTRUCTION_TOKENS = 4096
GUIDANCE_TOKENS = 8192
DEFAULT_BUDGET = CONTEXT_TOKENS - OUTPUT_TOKENS - INSTRUCTION_TOKENS - GUIDANCE_TOKENS


def _save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _environment(config_path: Path) -> EnvironmentConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return EnvironmentConfig.model_validate(payload.get("environment", payload))


async def _list_patients(instance, client_class) -> list[str]:
    client = client_class(instance.endpoint, ["placeholder"])
    identifiers, url = [], "Patient?_elements=id&_count=1000"
    while url:
        bundle = await client._request("GET", url)
        identifiers.extend(
            entry["resource"]["id"] for entry in bundle.get("entry", []) if entry.get("resource")
        )
        next_link = next(
            (link["url"] for link in bundle.get("link", []) if link.get("relation") == "next"), None
        )
        url = next_link.split("/fhir/", 1)[-1] if next_link else None
    return sorted(set(identifiers))


def _payload_chars(trajectory) -> int:
    return len(json.dumps(trajectory.model_dump(mode="json"), ensure_ascii=False))


async def _read_environment(config: EnvironmentConfig, output: Path) -> list[dict]:
    from environments import create_environment, prepare_trajectory, release_environment
    from environments.fhir import FHIRClient

    instance = await create_environment(config)
    try:
        patients = await _list_patients(instance, FHIRClient)
        print(f"{config.name}: {len(patients)} patients enumerated", flush=True)
        rows = []
        for index, patient in enumerate(patients, start=1):
            trajectory = await prepare_trajectory(instance, patient)
            _save(output / "trajectories" / f"{patient}.json", trajectory.model_dump(mode="json"))
            rows.append({
                "patient_id": patient,
                "sources": len(trajectory.sources),
                "payload_chars": _payload_chars(trajectory),
            })
            if index % 20 == 0 or index == len(patients):
                print(f"  read {index}/{len(patients)}", flush=True)
        return rows
    finally:
        await release_environment(config, instance)


def _offline_rows(source: Path) -> list[dict]:
    owned = collections.defaultdict(lambda: [0, 0, collections.Counter()])
    shared, referenced = {}, collections.defaultdict(set)

    def patient_of(resource: dict):
        for key in ("subject", "patient"):
            reference = (resource.get(key) or {}).get("reference")
            if reference and reference.startswith("Patient/"):
                return reference.split("/", 1)[1]
        return None

    for path in sorted(source.glob("Mimic*.ndjson.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                resource = json.loads(line)
                size = len(line.rstrip("\n"))
                resource_type, identifier = resource.get("resourceType", ""), resource.get("id")
                patient = patient_of(resource)
                if patient is None:
                    shared[f"{resource_type}/{identifier}"] = size
                    continue
                record = owned[patient]
                record[0] += 1
                record[1] += size
                record[2][resource_type] += 1
                for key in ("medication", "location", "specimen", "serviceProvider"):
                    value = resource.get(key)
                    for item in value if isinstance(value, list) else [value]:
                        reference = item.get("reference") if isinstance(item, dict) else None
                        if reference:
                            referenced[patient].add(reference)

    rows = []
    for patient, (count, total, _) in owned.items():
        extra = sum(shared.get(reference, 0) for reference in referenced[patient])
        rows.append({
            "patient_id": patient,
            "sources": count,
            "payload_chars": int((total + extra) * 1.08),
        })
    return rows


def _write_report(output: Path, name: str, rows: list[dict], cohort: dict, budget: int) -> None:
    skipped = cohort["skipped"]
    lines = [
        f"# {name} 患者池筛查",
        "",
        f"- 患者总数：{len(rows)}",
        f"- 可用（生成输入 ≤ {budget} token）：{len(rows) - len(skipped)}",
        f"- 跳过（超上下文）：{len(skipped)}",
        f"- 进化患者：{len(cohort['evolution_patients'])}",
        f"- 评测批：{len(cohort['batches'])} 批，共 {sum(len(b) for b in cohort['batches'])} 名患者",
        "",
        "| 患者 | 记录数 | 估算 token | 判定 |",
        "| --- | ---: | ---: | --- |",
    ]
    skipped_ids = {item["patient_id"] for item in skipped}
    for row in sorted(rows, key=lambda item: item["payload_chars"]):
        verdict = "跳过" if row["patient_id"] in skipped_ids else "可用"
        lines.append(
            f"| {row['patient_id']} | {row['sources']} | "
            f"{round(row['payload_chars'] / CHARS_PER_TOKEN)} | {verdict} |"
        )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--offline-source", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--evolution", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--budget-tokens", type=int, default=DEFAULT_BUDGET)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.offline_source:
        name = args.name or args.offline_source.parent.name
        rows = _offline_rows(args.offline_source)
    else:
        config = _environment(args.config)
        name = args.name or config.name
        rows = asyncio.run(_read_environment(config, args.output))

    rows.sort(key=lambda row: row["patient_id"])
    for row in rows:
        row["tokens"] = round(row["payload_chars"] / CHARS_PER_TOKEN)
        row["eligible"] = row["tokens"] <= args.budget_tokens
    _save(args.output / "pool_sizes.json", rows)

    eligible = [row["patient_id"] for row in rows if row["eligible"]]
    random.Random(args.seed).shuffle(eligible)
    evolution = eligible[: args.evolution]
    remainder = eligible[args.evolution:]
    batches = [
        remainder[index:index + args.batch_size]
        for index in range(0, len(remainder), args.batch_size)
    ]
    cohort = {
        "environment": name,
        "seed": args.seed,
        "budget_tokens": args.budget_tokens,
        "patient_pool": len(rows),
        "evolution_patients": evolution,
        "batches": batches,
        "skipped": [
            {"patient_id": row["patient_id"], "reason": "context_exceeded", "tokens": row["tokens"]}
            for row in rows if not row["eligible"]
        ],
    }
    _save(args.output / "cohort.json", cohort)
    _write_report(args.output, name, rows, cohort, args.budget_tokens)
    print(
        f"{name}: pool={len(rows)} eligible={len(eligible)} "
        f"evolution={len(evolution)} batches={len(batches)} "
        f"skipped={len(cohort['skipped'])}"
    )


if __name__ == "__main__":
    main()
