"""为全量对比实验生成运行配置：三环境 × 六方法 × 全部患者。

    # 基础方法（含 medevolve 的进化运行）
    python scripts/emit_comparison_configs.py --slug physicianbench --method selfgen \
        --cohort outputs/pool-screening/physicianbench/cohort.json

    # medevolve 批量运行（进化完成后，用冻结策略与缓存指导）
    python scripts/emit_comparison_configs.py --slug physicianbench --method medevolve \
        --cohort outputs/pool-screening/physicianbench/cohort.json \
        --strategy outputs/full-comparison/physicianbench/medevolve/evolution/final_strategy.json

配置只改方法、患者名单、运行身份与预算；环境、题型分配与评测口径沿用公共配置。
"""

import argparse
import json
from pathlib import Path

from contracts import ExperimentConfig

# 生成阶段角色与预算：单一模型、零工具调用的方法保持 1/0；多阶段方法按其阶段数配置。
METHOD_SPEC = {
    "medevolve": {
        "roles": ["data_explorer", "environment_explorer", "generator", "feedback", "updater"],
        "budgets": {"data_explorer": (30, 200), "environment_explorer": (30, 200),
                    "generator": (1, 0)},
        "exploration": True,
    },
    "envexplore": {
        "roles": ["environment_explorer", "generator"],
        "budgets": {"environment_explorer": (30, 200), "generator": (1, 0)},
        "exploration": True,
        "exploration_in_batches": True,
    },
    "selfgen": {"roles": ["generator"], "budgets": {"generator": (6, 0)}},
    "singleshot": {"roles": ["generator"], "budgets": {"generator": (1, 0)}},
    "workflow": {"roles": ["generator"], "budgets": {"generator": (8, 40)}},
    "taskcraft": {"roles": ["generator"], "budgets": {"generator": (8, 40)}},
}
ENVIRONMENT_BASE = {
    "physicianbench": "physicianbench-comparison",
    "medagentbench": "medagentbench-flash",
    "fhir-agentbench": "fhir-agentbench-flash",
}
OPENLUX_MODEL = {
    "model": "gpt-4o-mini",
    "base_url": "https://api.openlux.ai/v1",
    "api_key_env": "OPENLUX_API_KEY",
    "instructions_role": "system",
    "parallel_tool_calls": True,
}
EVALUATION_BUDGETS = {
    role: (30, 200) for role in ("verifier", "quality", "difficulty", "solver", "completion")
}
BATCH_SIZE = 16
EXPLORATION_BATCH_SIZE = 24


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _runtime(spec: dict, template: dict) -> dict:
    models = {role: dict(OPENLUX_MODEL) for role in spec["roles"]}
    budgets = {role: {"max_model_turns": turns, "max_tool_calls": calls}
               for role, (turns, calls) in spec["budgets"].items()}
    runtime = {**template["runtime"], "models": models, "budgets": budgets}
    return runtime


def _evaluation(template: dict) -> dict:
    models = {role: dict(OPENLUX_MODEL) for role in EVALUATION_BUDGETS}
    budgets = {role: {"max_model_turns": turns, "max_tool_calls": calls}
               for role, (turns, calls) in EVALUATION_BUDGETS.items()}
    evaluation = {**template["evaluation"], "runtime": {"models": models, "budgets": budgets}}
    return evaluation


def _payload(template: dict, spec: dict, *, run_id: str, output_dir: Path, slug: str,
             method: str, evolution: list[str], evaluation: list[str],
             rounds: list[list[str]], manager_url: str, strategy: dict | None = None,
             guidance: str | None = None) -> dict:
    payload = {
        **template,
        "run_id": run_id,
        "method": method,
        "environment": {**template["environment"], "manager_url": manager_url},
        "runtime": _runtime(spec, template),
        "evaluation": _evaluation(template),
        "evolution_patients": evolution,
        "evaluation_patients": evaluation,
        "round_patients": rounds,
        "initial_strategy": strategy or template["initial_strategy"],
        "output_dir": str(output_dir),
        "compare_initial_strategy": False,
    }
    if guidance:
        payload["guidance_path"] = guidance
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slug", required=True, choices=sorted(ENVIRONMENT_BASE))
    parser.add_argument("--method", required=True, choices=sorted(METHOD_SPEC))
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--strategy", type=Path, help="medevolve 批量运行使用的冻结策略")
    parser.add_argument("--configs-dir", type=Path, default=Path("configs"))
    parser.add_argument("--template-dir", type=Path, default=Path("configs"))
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--manager-url", default="http://127.0.0.1:18084")
    args = parser.parse_args()

    spec = METHOD_SPEC[args.method]
    template = json.loads(
        (args.template_dir / f"{ENVIRONMENT_BASE[args.slug]}.json").read_text(encoding="utf-8")
    )
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    root = args.output_root or Path("outputs/full-comparison") / args.slug / args.method
    batch_size = args.batch_size or (EXPLORATION_BATCH_SIZE if args.method == "envexplore"
                                     else BATCH_SIZE)
    evolution_patients = list(cohort["evolution_patients"])
    remainder = [patient for batch in cohort["batches"] for patient in batch]
    batches = [remainder[index:index + batch_size]
               for index in range(0, len(remainder), batch_size)]
    written = []

    if args.method == "medevolve" and args.strategy is None:
        payload = _payload(template, spec, run_id=f"full-{args.slug}-medevolve-evolution",
                           output_dir=root / "evolution", slug=args.slug, method=args.method,
                           evolution=evolution_patients, evaluation=[],
                           rounds=[list(evolution_patients), list(evolution_patients)],
                           manager_url=args.manager_url)
        ExperimentConfig.model_validate(payload)
        path = args.configs_dir / f"full-{args.slug}-medevolve-evolution.json"
        _save(path, payload)
        written.append(path)
        print(f"wrote {path} (evolution patients {len(evolution_patients)})")
        print("批量运行配置在进化完成后使用 --strategy 生成")
        return

    strategy = None
    guidance = None
    if args.strategy is not None:
        strategy = json.loads(args.strategy.read_text(encoding="utf-8"))
        guidance = str(args.strategy.parent / "guidance.json")
        if not Path(guidance).exists():
            raise ValueError(f"guidance is missing next to the frozen strategy: {guidance}")

    for index, batch in enumerate(batches, start=1):
        payload = _payload(
            template, spec, run_id=f"full-{args.slug}-{args.method}-batch-{index:02d}",
            output_dir=root / f"batch-{index:02d}", slug=args.slug, method=args.method,
            evolution=(evolution_patients if spec.get("exploration_in_batches") else []),
            evaluation=batch, rounds=[], manager_url=args.manager_url,
            strategy=strategy, guidance=guidance,
        )
        ExperimentConfig.model_validate(payload)
        path = args.configs_dir / f"full-{args.slug}-{args.method}-batch-{index:02d}.json"
        _save(path, payload)
        written.append(path)
    print(f"wrote {len(written)} configs, {sum(len(b) for b in batches)} evaluation patients")


if __name__ == "__main__":
    main()
