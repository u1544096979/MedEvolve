"""TaskCraft 真实验收：环境、模型、预算、种子与患者全部取自同一份显式配置。

    MEDEVOLVE_TASKCRAFT_LIVE_CONFIG=configs/taskcraft-mechanism-physicianbench.json \
    /Users/guagua/code/MedEvolve/.venv/bin/python -m pytest tests/method/test_taskcraft_live.py -q -s

配置路径指向一份公共 `ExperimentConfig`。未给出路径时本测试跳过，不发起任何真实调用；
给出路径后按该配置的模型与预算发起一次真实的生成调用，并占用一个独立容器实例。机制
验收配置里的 generator 预算是机制预算，不是公共比较预算，成本字段按真实用量打印。

一次真实运行的产物写入配置的 `output_dir`：`report.json`（验收报告）、`pipeline.json`
（完整候选池、阶段记录、派生种子、分支顺序与选题记录）、`result.json`（运行状态、用量
与完整 trace）、`config.json`（本次使用的配置）。已有产物的目录不再写入。
"""

import json
import os
import random
from pathlib import Path

import pytest

from contracts import ExperimentConfig, TaskBatch
from methods.taskcraft import bind_generator

CONFIG_ENV = "MEDEVOLVE_TASKCRAFT_LIVE_CONFIG"
# 作者源码基线：断言 trace 里的 author_commit 必须等于该提交，与实现侧的常量互相独立。
AUTHOR_BASELINE = "1e405e0b02c4738654e2d885b16677b46c9fe969"
ORIGINS = ("atomic", "depth", "width")

pytestmark = pytest.mark.skipif(
    not os.environ.get(CONFIG_ENV),
    reason=f"set {CONFIG_ENV} to an explicit experiment configuration to run live",
)


def load_config() -> ExperimentConfig:
    path = os.environ[CONFIG_ENV]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{CONFIG_ENV} 指向的文件不存在：{path}")
    with open(path, encoding="utf-8") as handle:
        return ExperimentConfig.model_validate_json(handle.read())


def directions_for(config: ExperimentConfig):
    """与公共 runner 的评测组同一取样方式：按方向权重取两个生成位置。"""
    allocation = config.initial_strategy.allocation
    sampling = random.Random(config.random_seed)
    return sampling.choices(allocation, weights=[direction.weight for direction in allocation], k=2)


def pipeline_entry(result):
    return next(item for item in result.trace if item.get("stage") == "taskcraft/pipeline")


def stage_counts(pipeline) -> dict:
    return {origin: sum(1 for item in pipeline["pool"] if item["origin"] == origin)
            for origin in ORIGINS}


def check_contract(pipeline, result, known_sources) -> None:
    """真实阶段成功判据：每阶段至少一个带答案与可回查证据的候选，且发生真实调用。"""
    assert result.status != "failed", f"生成整体失败：{result.error}"
    assert pipeline["author_commit"] == AUTHOR_BASELINE
    for origin in ORIGINS:
        candidates = [item for item in pipeline["pool"] if item["origin"] == origin]
        assert candidates, (f"{origin} 阶段没有真实成功候选；阶段记录："
                            f"{json.dumps(pipeline['records'], ensure_ascii=False)}")
        for item in candidates:
            question = item["question"]
            evidence = question["evidence"]
            assert question["reference_answer"].strip(), f"{origin} 候选缺少参考答案"
            assert evidence, f"{origin} 候选缺少证据"
            assert {entry["source_id"] for entry in evidence} <= known_sources, \
                f"{origin} 候选证据不在当前患者轨迹内"
            if origin != "atomic":
                assert item["parents"], f"{origin} 候选缺少父题"
    assert result.usage["model_turns"] > 0
    assert result.usage["tool_calls"] > 0
    if result.status == "completed":
        assert result.output is not None, "completed 状态没有输出，不算阶段成功"
        assert TaskBatch.model_validate(result.output).questions, \
            "completed 状态输出为空批，不算阶段成功"


def save(path: Path, value) -> None:
    """与公共 runner 同一写法：先写临时文件再替换，避免留下半份产物。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def persist(config: ExperimentConfig, report: dict, pipeline: dict, result) -> Path:
    """把一次真实验收的全部产物写入配置的 output_dir；已有产物的目录不覆盖。"""
    root = Path(config.output_dir)
    if (root / "report.json").exists():
        raise FileExistsError(f"验收目录已有产物，请把配置指向新的 output_dir：{root}")
    save(root / "config.json", config.model_dump(mode="json"))
    save(root / "report.json", report)
    save(root / "pipeline.json", pipeline)
    save(root / "result.json", {"status": result.status, "error": result.error,
                                 "usage": result.usage, "output": result.output,
                                 "trace": result.trace})
    return root


@pytest.mark.asyncio
async def test_live_batch_produces_three_real_stages_with_evidence_and_cost():
    from environments import create_environment, get_tools, prepare_trajectory, release_environment

    config = load_config()
    assert config.evaluation_patients, "配置没有评测患者，无法选定真实患者"
    patient = config.evaluation_patients[0]
    directions = directions_for(config)
    assert len(directions) == 2

    instance = await create_environment(config.environment)
    try:
        trajectory = await prepare_trajectory(instance, patient)
        tools = await get_tools(instance, [patient])
    finally:
        await release_environment(config.environment, instance)
    assert trajectory.patient_id == patient
    assert trajectory.environment == config.environment.name
    assert tools is not None, "环境没有提供工具"
    known_sources = {source.source_id for source in trajectory.sources}
    assert known_sources, "患者轨迹没有可回查来源"

    generate = bind_generator(config.random_seed)
    result = await generate(trajectory, config.environment, directions, config.runtime)
    pipeline = pipeline_entry(result)
    counts = stage_counts(pipeline)

    report = {
        "run_id": config.run_id,
        "environment": config.environment.name,
        "seed_data": config.environment.seed_data,
        "patient": patient,
        "generator_model": config.runtime.models["generator"].model_dump(mode="json"),
        "generator_budget": config.runtime.budgets["generator"].model_dump(mode="json"),
        "random_seed": config.random_seed,
        "derived_seeds": pipeline["derived_seeds"],
        "directions": [direction.model_dump(mode="json") for direction in directions],
        "status": result.status,
        "error": result.error,
        "usage": result.usage,
        "candidate_counts": counts,
        "branch_order": pipeline["branch_order"],
        "stage_records": pipeline["records"],
        "selection": pipeline["selection"],
        "selected_questions": [candidate["instruction"]
                               for candidate in pipeline["selection"]["selected"]],
    }
    report["run_dir"] = str(Path(config.output_dir))
    persist(config, report, pipeline, result)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    check_contract(pipeline, result, known_sources)
