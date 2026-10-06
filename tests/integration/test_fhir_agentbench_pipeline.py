"""FHIR-AgentBench 公共任务执行链：核验 → 质量 → 解题 → 完成。

判分沿用公共 ``evaluation.evaluate_task``，不为本环境建立单独算法。全部使用
httpx.MockTransport 与固定模型输出，不连接 Docker、PostgreSQL、http://127.0.0.1:18086
或任何模型服务。
"""

import json
from collections import Counter

import httpx
import pytest
from agents import FunctionTool
from agents.tool_context import ToolContext

import runtime
from contracts import (
    Budget,
    CompletionReview,
    EnvironmentConfig,
    EvaluationConfig,
    Evidence,
    ModelConfig,
    PatientTrajectory,
    QualityReview,
    RunResult,
    RuntimeConfig,
    SolverAnswer,
    Task,
    Verification,
)
from environments import fhir_agentbench
from evaluation import evaluate_task

MANAGER_URL = "http://manager.test"
ENDPOINT = "http://instance-one/fhir/"
TRAJECTORY_MARKER = "trajectory-text-marker"


def record(resource_type, identifier, **fields):
    resource = {"resourceType": resource_type, "id": identifier}
    resource.update(fields)
    return resource


def stored():
    return {
        "Patient/a": record("Patient", "a", gender="female"),
        "Patient/b": record("Patient", "b", gender="male"),
        "Observation/o1": record(
            "Observation",
            "o1",
            subject={"reference": "Patient/a"},
            status="final",
            code={"text": "heart rate"},
            effectiveDateTime="2136-04-22T18:47:00-04:00",
            valueQuantity={"value": 80, "unit": "beats/min"},
        ),
        "Observation/ob": record(
            "Observation",
            "ob",
            subject={"reference": "Patient/b"},
            status="final",
            effectiveDateTime="2136-04-22T19:00:00-04:00",
        ),
    }


def bundle(resources):
    return {
        "resourceType": "Bundle",
        "total": len(resources),
        "entry": [{"resource": resource} for resource in resources],
    }


class FakeServices:
    """按主机名服务的 fake 管理服务与 fake FHIR 服务，记录每次请求。"""

    def __init__(self):
        self.store = stored()
        self.requests = []
        self.created = []

    @staticmethod
    def owner(resource):
        return resource.get("subject", {}).get("reference")

    def handle(self, request):
        self.requests.append(request)
        host, path = request.url.host, request.url.path
        if host == "manager.test":
            if request.method == "POST":
                instance_id = f"i{len(self.created)}"
                self.created.append(instance_id)
                return httpx.Response(
                    201,
                    json={
                        "instance_id": instance_id,
                        "environment": "FHIR-AgentBench",
                        "endpoint": ENDPOINT,
                    },
                )
            return httpx.Response(204)
        assert host == "instance-one", f"unexpected host: {host}"
        local = path.removeprefix("/fhir/")
        if local.endswith("/$everything"):
            identifier = local.split("/")[1]
            resources = [
                resource
                for resource in self.store.values()
                if (
                    resource["resourceType"] == "Patient"
                    and resource["id"] == identifier
                )
                or self.owner(resource) == f"Patient/{identifier}"
            ]
            return httpx.Response(
                200,
                json=bundle(sorted(resources, key=lambda r: (r["resourceType"], r["id"]))),
            )
        if "/" in local:
            if local not in self.store:
                return httpx.Response(404, json={"resourceType": "OperationOutcome"})
            return httpx.Response(200, json=self.store[local])
        resources = [
            resource
            for resource in self.store.values()
            if resource["resourceType"] == local
        ]
        params = request.url.params
        if params.get("_id"):
            wanted = set(params["_id"].split(","))
            resources = [r for r in resources if r["id"] in wanted]
        if params.get("patient"):
            wanted = {f"Patient/{p}" for p in params["patient"].split(",")}
            resources = [r for r in resources if self.owner(r) in wanted]
        return httpx.Response(
            200, json=bundle(sorted(resources, key=lambda r: r["id"]))
        )

    @property
    def everything_calls(self):
        return [
            request
            for request in self.requests
            if request.url.path.endswith("/$everything")
        ]


class FixedModels:
    """固定模型输出与固定 solver 检索，记录公共链实际收到的输入。"""

    def __init__(self, services):
        self.services = services
        self.calls = Counter()
        self.verifier_payloads = []
        self.solver_payloads = []
        self.requests_before_state = []

    async def call_llm(self, *, instructions, input_text, output_type, model):
        self.calls[output_type.__name__] += 1
        if output_type is Verification:
            payload = json.loads(input_text)
            if "resource" in payload:
                self.verifier_payloads.append(payload["resource"])
            output = {"valid": True, "reason": "原始记录支持该题"}
        elif output_type is QualityReview:
            output = {
                "dimensions": {
                    dimension: {"judgment": "合格", "reason": "满足要求"}
                    for dimension in (
                        "clinical_value",
                        "correctness",
                        "longitudinal_integration",
                        "clarity",
                    )
                }
            }
        elif output_type is CompletionReview:
            output = {"completed": True, "reason": "检索记录支持答案"}
        else:
            raise AssertionError(f"unexpected llm schema: {output_type.__name__}")
        return RunResult(output=output, trace=[], usage={}, status="completed")

    async def run_agent(self, *, instructions, input_text, tools, output_type, model, budget):
        assert output_type is SolverAnswer
        # 公共运行通道只接受 SDK FunctionTool，本环境必须落在同一条通道上。
        assert all(isinstance(tool, FunctionTool) for tool in tools)
        payload = json.loads(input_text)
        self.solver_payloads.append(payload)
        # 解题输入只含题目、患者身份与决策点；参考答案、出题证据与缓存轨迹都不传入。
        assert set(payload) == {"instruction", "patient_id", "decision_point"}
        for leaked in ("reference_answer", "evidence", "sources", "trajectory"):
            assert leaked not in input_text
        assert TRAJECTORY_MARKER not in input_text

        tool = next(t for t in tools if t.name == "get_resources_by_patient_fhir_id")
        arguments = json.dumps(
            {"resource_type": "Observation", "patient_fhir_id": payload["patient_id"]}
        )
        context = ToolContext(
            context=None,
            tool_name=tool.name,
            tool_call_id="c1",
            tool_arguments=arguments,
        )
        records = json.loads(await tool.on_invoke_tool(context, arguments))
        assert [resource["id"] for resource in records["Observation"]] == ["o1"]
        # 解题结束时的 FHIR 请求数：状态读取与后续复核都不应再访问 FHIR 服务。
        self.requests_before_state.append(
            sum(1 for request in self.services.requests if request.url.host == "instance-one")
        )
        return RunResult(
            output={"answer": "最近一次心率 80 次/分"},
            trace=[
                {
                    "type": "tool_start",
                    "name": tool.name,
                    "call_id": "c1",
                    "arguments": json.loads(arguments),
                },
                {
                    "type": "tool_output",
                    "name": tool.name,
                    "call_id": "c1",
                    "output": records,
                },
            ],
            usage={},
            status="completed",
        )


@pytest.fixture
def services(monkeypatch):
    services = FakeServices()
    transport = httpx.MockTransport(services.handle)
    real_client = httpx.AsyncClient

    def client(**kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    monkeypatch.setattr(fhir_agentbench, "_READERS", {})
    return services


def environment():
    return EnvironmentConfig(
        name="FHIR-AgentBench", manager_url=MANAGER_URL, seed_data="/seed"
    )


def evaluation_config():
    roles = ("verifier", "quality", "solver", "completion")
    return EvaluationConfig(
        runtime=RuntimeConfig(
            models={role: ModelConfig(model=role, api_key_env="UNUSED") for role in roles},
            budgets={role: Budget(max_model_turns=2, max_tool_calls=2) for role in roles},
        )
    )


def prepare(monkeypatch, services, source_ids):
    trajectory = PatientTrajectory(
        patient_id="a",
        environment="FHIR-AgentBench",
        text=f"固定轨迹 {TRAJECTORY_MARKER}",
        sources=[
            {
                "source_id": source_id,
                "statement": json.dumps(services.store[source_id], sort_keys=True),
            }
            for source_id in source_ids
        ],
    )
    task = Task(
        id="pipeline:1",
        patient_id="a",
        environment="FHIR-AgentBench",
        round_index=0,
        source_method="medevolve",
        instruction="报告患者最近一次心率并说明依据",
        task_type="检查与风险分层",
        reference_answer="最近一次心率 80 次/分",
        evidence=[
            {"source_id": source_id, "statement": "题目依据"}
            for source_id in source_ids
        ],
        decision_point=None,
    )
    models = FixedModels(services)
    monkeypatch.setattr(runtime, "call_llm", models.call_llm)
    monkeypatch.setattr(runtime, "run_agent", models.run_agent)
    return task, trajectory, models


@pytest.mark.asyncio
async def test_public_pipeline_runs_the_existing_chain_with_read_only_state(
    services, monkeypatch
):
    task, trajectory, models = prepare(monkeypatch, services, ["Observation/o1"])
    result = await evaluate_task(task, trajectory, environment(), evaluation_config())

    assert result.status == "completed"
    assert result.verification.valid is True
    assert result.completion.completed is True
    # 公共链按「每条证据核验 + 整体核验」读取原始记录，读到的就是服务端原始资源。
    assert models.calls["Verification"] == 2
    assert [resource["valueQuantity"] for resource in models.verifier_payloads] == [
        {"value": 80, "unit": "beats/min"}
    ]
    assert models.solver_payloads == [
        {
            "instruction": task.instruction,
            "patient_id": "a",
            "decision_point": None,
        }
    ]

    # read_state 返回只读语义，不套用 PhysicianBench 的可写资源筛选。
    assert result.observed_state == {
        "patient_id": "a",
        "environment": "FHIR-AgentBench",
        "operation_semantics": {"read_only": True},
    }
    assert "resources" not in result.observed_state

    # 只读环境的完成判断不读全量病历：状态读取不发起任何 FHIR 请求。
    assert len(models.requests_before_state) == 1
    fhir_requests = [
        request for request in services.requests if request.url.host == "instance-one"
    ]
    assert len(fhir_requests) == models.requests_before_state[0]
    assert services.everything_calls == []


@pytest.mark.asyncio
async def test_absent_evidence_source_fails_before_any_environment_read(
    services, monkeypatch
):
    task, trajectory, models = prepare(monkeypatch, services, ["Observation/o1"])
    task = task.model_copy(
        update={
            "evidence": [
                Evidence(source_id="Observation/absent", statement="题目依据")
            ]
        }
    )
    result = await evaluate_task(task, trajectory, environment(), evaluation_config())

    assert result.status == "invalid"
    assert result.failure_category == "insufficient_evidence"
    assert result.verification.valid is False
    assert "Observation/absent" in result.verification.reason
    assert not models.calls
    assert services.requests == []


@pytest.mark.asyncio
async def test_out_of_scope_evidence_source_is_refused_not_silently_accepted(
    services, monkeypatch
):
    task, trajectory, models = prepare(
        monkeypatch, services, ["Observation/o1", "Observation/ob"]
    )
    task = task.model_copy(
        update={"evidence": [Evidence(source_id="Observation/ob", statement="题目依据")]}
    )
    result = await evaluate_task(task, trajectory, environment(), evaluation_config())

    assert result.status == "failed"
    assert result.failure_category == "environment_error"
    assert result.verification is None
    assert result.completion is None
    assert not models.calls
    assert "outside the selected patient scope" in result.observed_state["evaluation_error"]
    assert any(
        request.url.path == "/fhir/Observation/ob" for request in services.requests
    )
