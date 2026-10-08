"""Opt-in acceptance against configured local upstreams and authorized services."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from experiments.harness_train import (ROOT, _require_execution, load_config, preflight,
    group_id, run_training, selections, rooted, write_json)
from experiments.harness_test import run_test


LIVE = os.environ.get("MEDEVOLVE_HARNESS_LIVE") == "1"
pytestmark = pytest.mark.skipif(not LIVE, reason="需要 MEDEVOLVE_HARNESS_LIVE=1，默认不启动模型或医疗服务")


@pytest.mark.parametrize("harness", ["openclaw", "hermes", "ace", "reflexion"])
def test_live_train_and_frozen_test_for_configured_group(harness):
    config=load_config(ROOT/"configs/harness-evolution.json")
    if config.get("execute") is not True:
        pytest.skip("configs/harness-evolution.json 的 execute 仍为 false；真实接入未验证")
    environment=os.environ.get("MEDEVOLVE_HARNESS_ENVIRONMENT","PhysicianBench")
    source=os.environ.get("MEDEVOLVE_HARNESS_SOURCE","medevolve")
    combo=(environment,harness,source)
    report=preflight(config,[combo])[0]
    if not report["ok"]:
        pytest.skip("本机上游程序、依赖、锁定版本、数据或评审配置未就绪；真实接入未验证")
    try:
        _require_execution(config)
    except ValueError as exc:
        pytest.skip(f"模型授权或预算未就绪；真实接入未验证：{exc}")
    trained=asyncio.run(run_training(config,*combo))
    assert len(trained["train_task_ids"])==7
    assert all(row.get("tool_session",{}).get("closed") for row in trained["attempts"])
    group_path=rooted(group_id(config,*combo,config["seeds"][0]))
    write_json(group_path/"train.json",trained,config)
    tested=asyncio.run(run_test(config,*combo))
    assert tested["planned_denominator"]==3
    assert tested["snapshots_unchanged"]
    write_json(group_path/"test.json",tested,config)
