"""读取官方任务并校验仓库版本与评分资产。"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from harnesses.contracts import BenchmarkCase

REFSOL_URL = "https://stanfordmedicine.box.com/s/fizv0unyjgkb1r3a83rfn5p3dc673uho"


class BenchmarkUnavailable(ValueError):
    """官方资产缺失或版本不一致。"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cases(environment, settings):
    root = Path(settings["source_dir"]).expanduser().resolve()
    try:
        actual = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        if actual != settings["commit"]:
            raise BenchmarkUnavailable("benchmark_unavailable: 官方仓库提交不匹配")
        cases, files = [], []
        if environment == "PhysicianBench":
            files.extend([root / "utils/eval_helpers.py", root / "tools/fhir_api_functions.py"])
            for instruction in sorted((root / settings.get("task_path", "tasks/v1")).glob("*/instruction.md")):
                text = instruction.read_text(encoding="utf-8")
                patients = set(re.findall(r"\bMRN\d+\b", text))
                if len(patients) != 1:
                    raise BenchmarkUnavailable(f"benchmark_unavailable: 题目患者范围不明确：{instruction}")
                grader = instruction.parent / "tests/test_outputs.py"
                files.extend([instruction, grader])
                cases.append(BenchmarkCase(id=instruction.parent.name, environment=environment,
                    instruction=text, patient_id=next(iter(patients)), asset_path=grader))
        elif environment == "MedAgentBench":
            data = root / settings["dataset_path"]
            directory = root / "src/server/tasks/medagentbench"
            refsol = directory / "refsol.py"
            if settings.get("refsol_source") != REFSOL_URL or not settings.get("refsol_sha256"):
                raise BenchmarkUnavailable("benchmark_unavailable: 参考评分器必须记录官方来源与 SHA-256")
            if digest(refsol) != settings["refsol_sha256"]:
                raise BenchmarkUnavailable("benchmark_unavailable: refsol.py 摘要不匹配")
            files.extend([data, directory / "eval.py", directory / "utils.py", refsol,
                          root / "data/medagentbench/funcs_v1.json"])
            for row in json.loads(data.read_text(encoding="utf-8")):
                # 不把 eval_MRN 或 sol 当作患者提示传给解题器。
                text = row["instruction"] + ("\n\n" + row["context"] if row.get("context") else "")
                patients = set(re.findall(r"\bS\d{7}\b", text))
                cases.append(BenchmarkCase(id=row["id"], environment=environment,
                    instruction=text, patient_id=next(iter(patients)) if len(patients) == 1 else "",
                    asset_path=directory / "eval.py", reference=row))
        else:
            raise BenchmarkUnavailable(f"benchmark_unavailable: 未支持环境 {environment}")
        if not cases or len({case.id for case in cases}) != len(cases):
            raise BenchmarkUnavailable("benchmark_unavailable: 官方任务为空或 ID 重复")
        tracked = [str(path.relative_to(root)) for path in files if path.name != "refsol.py"]
        changes = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--", *tracked],
            capture_output=True, text=True, check=True, timeout=10).stdout
        if changes:
            raise BenchmarkUnavailable("benchmark_unavailable: 官方任务或评分源码存在修改")
        manifest = {"repository":settings["repository"], "commit":actual,
            "dataset_path":settings.get("dataset_path", settings.get("task_path", "tasks/v1")),
            "files":{str(path.relative_to(root)):digest(path) for path in files},
            "task_ids":[case.id for case in cases], "refsol_source":settings.get("refsol_source")}
        return cases, manifest
    except (OSError, subprocess.SubprocessError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise BenchmarkUnavailable(f"benchmark_unavailable: {type(exc).__name__}: {exc}") from exc
