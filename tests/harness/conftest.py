"""Shared harness fixtures: fake Docker backend, isolation rig and tool sessions.

The rig runs without Docker and without any paid model call. Docker is replaced by
`FakeDocker`, which materialises containers, volumes and bind mounts on the local
filesystem and enforces the same mount visibility a real container would: a path
that is not mounted in a container cannot be reached from it.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from agents import function_tool
from contracts import EnvironmentConfig, EnvironmentInstance, SolverInput
from environments.errors import PatientAccessError
from harnesses.contracts import GroupKey, HarnessName, Snapshot
from harnesses.isolation import DockerCommandError, HarnessIsolation
from harnesses.medical_bridge import ToolSessions


class FakeDocker:
    """Docker CLI stand-in recording every command and enforcing mount scope."""

    def __init__(self, root: Path):
        self.root = root
        self.commands: list[tuple] = []
        self.containers: dict[str, dict] = {}
        self.volumes: dict[str, Path] = {}
        self.fail_run: str | None = None

    async def __call__(self, *args, stdin=None):
        self.commands.append(args)
        if args[0] == "volume":
            return self._volume(args)
        if args[0] == "run":
            return self._run(args)
        if args[:2] == ("container", "inspect"):
            return self._inspect(args[-1])
        if args[:2] == ("container", "rm"):
            return self._remove(args[-1])
        if args[0] == "exec":
            return self._exec(args)
        raise AssertionError(args)

    # -- docker vocabulary -------------------------------------------------
    def _volume(self, args: tuple) -> str:
        name = args[-1]
        if args[1] == "create":
            directory = self.root / "volumes" / name
            directory.mkdir(parents=True, exist_ok=True)
            self.volumes[name] = directory
            return name
        if args[1] == "rm":
            if name not in self.volumes:
                raise DockerCommandError(f"No such volume: {name}")
            directory = self.volumes.pop(name)
            for child in sorted(directory.rglob("*"), reverse=True):
                child.rmdir() if child.is_dir() else child.unlink()
            directory.rmdir()
            return name
        raise AssertionError(args)

    def _labels(self, args: tuple) -> dict:
        labels = {}
        for index, value in enumerate(args):
            if value == "--label" and "=" in args[index + 1]:
                key, _, label = args[index + 1].partition("=")
                labels[key] = label
        return labels

    def _run(self, args: tuple) -> str:
        if self.fail_run is not None:
            raise DockerCommandError(self.fail_run)
        name = args[args.index("--name") + 1]
        mounts = []
        env = {}
        network = None
        for index, value in enumerate(args):
            if value == "--mount":
                spec = dict(
                    part.split("=", 1) for part in args[index + 1].split(",") if "=" in part
                )
                source = spec["source"]
                if spec["type"] == "volume":
                    source = str(self.volumes[spec["source"]])
                mounts.append({
                    "Type": spec["type"],
                    "Source": source,
                    "Destination": spec["target"],
                    "RW": "readonly" not in args[index + 1].split(","),
                })
            if value == "--env":
                key, _, item = args[index + 1].partition("=")
                env[key] = item
            if value == "--network":
                network = args[index + 1]
        if name in self.containers:
            raise DockerCommandError(
                f"Conflict. The container name {name} is already in use"
            )
        self.containers[name] = {
            "Config": {"Labels": self._labels(args)},
            "Mounts": mounts,
            "NetworkSettings": {"Networks": {network: {}} if network else {}},
            "env": env,
            "network": network,
        }
        return name

    def _inspect(self, name: str) -> str:
        if name not in self.containers:
            raise DockerCommandError(f"No such container: {name}")
        return json.dumps([self.containers[name]])

    def _remove(self, name: str) -> str:
        if name not in self.containers:
            raise DockerCommandError(f"No such container: {name}")
        self.containers.pop(name)
        return name

    def _exec(self, args: tuple) -> str:
        _, _flag, name, *command = args
        if name not in self.containers:
            raise DockerCommandError(f"No such container: {name}")
        if command[:1] == ["cat"]:
            return self.read(name, command[1])
        if command[:1] == ["write"]:
            self.write(name, command[1], command[2])
            return ""
        return ""

    # -- container-visible filesystem -------------------------------------
    def _resolve(self, name: str, container_path: str) -> tuple[Path, bool]:
        if name not in self.containers:
            raise DockerCommandError(f"No such container: {name}")
        for mount in self.containers[name]["Mounts"]:
            destination = mount["Destination"].rstrip("/")
            if container_path == destination or container_path.startswith(destination + "/"):
                relative = container_path[len(destination):].lstrip("/")
                return Path(mount["Source"]) / relative, mount["RW"]
        raise PermissionError(f"{container_path} is not mounted in {name}")

    def read(self, name: str, container_path: str) -> str:
        path, _ = self._resolve(name, container_path)
        return path.read_text(encoding="utf-8")

    def write(self, name: str, container_path: str, text: str) -> None:
        path, writable = self._resolve(name, container_path)
        if not writable:
            raise PermissionError(f"{container_path} is read-only in {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def mounts(self, name: str) -> dict[str, dict]:
        return {
            mount["Destination"]: dict(mount)
            for mount in self.containers[name]["Mounts"]
        }

    def container_names(self) -> set[str]:
        return set(self.containers)


class FakeProvider:
    """OpenAI-compatible Chat Completions backend for gateway tests."""

    def __init__(self, *, usage=None, emit_usage=True, tool_calls=0, status=200,
                 error=None, content="ok"):
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.usage = usage if usage is not None else {
            "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        }
        self.emit_usage = emit_usage
        self.tool_calls = tool_calls
        self.status = status
        self.error = error
        self.content = content

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(json.loads(request.content))
        self.headers.append(dict(request.headers))
        if self.status != 200:
            return httpx.Response(self.status, json={"error": {"message": self.error or "boom"}})
        message = {"role": "assistant", "content": self.content, "tool_calls": [
            {"id": f"call-{index}", "type": "function",
             "function": {"name": "medical_action", "arguments": "{}"}}
            for index in range(self.tool_calls)
        ]}
        payload = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": self.requests[-1]["model"],
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls" if self.tool_calls else "stop",
                "message": message,
            }],
        }
        if self.emit_usage:
            payload["usage"] = dict(self.usage)
        return httpx.Response(200, json=payload)


class FakeFHIRService:
    """In-process FHIR backend for bridge tests; records every request."""

    def __init__(self, *, owner: str = "p1"):
        self.owner = owner
        self.requests: list[httpx.Request] = []
        self.routes: list[tuple[str, object]] = []

    def route(self, fragment: str, response) -> None:
        self.routes.append((fragment, response))

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        for fragment, response in reversed(self.routes):
            if fragment in str(request.url):
                return response() if callable(response) else response
        if request.method == "POST":
            payload = json.loads(request.content or b"{}")
            return httpx.Response(201, json={**payload, "id": "created-1"})
        if "Condition" in str(request.url):
            return httpx.Response(200, json={"resourceType": "Bundle", "total": 1, "entry": [
                {"resource": {
                    "resourceType": "Condition", "id": "cond-1",
                    "clinicalStatus": {"coding": [{"code": "active"}]},
                    "subject": {"reference": f"Patient/{self.owner}"},
                }}
            ]})
        return httpx.Response(200, json={"resourceType": "Bundle", "total": 0, "entry": []})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)

    def methods(self) -> list[str]:
        return [request.method for request in self.requests]

    def urls(self) -> list[str]:
        return [str(request.url) for request in self.requests]


class FakeEnvironments:
    """Module-shaped stand-in for `environments` with per-attempt storage."""

    def __init__(self, *, tool_factory=None):
        self.instances: dict[str, dict] = {}
        self.created: list[EnvironmentInstance] = []
        self.released: list[EnvironmentInstance] = []
        self.tool_calls: list[tuple] = []
        self.tool_sessions: list[tuple] = []
        self.state_reads: list[tuple] = []
        self.tool_factory = tool_factory
        self.fail_get_tools = False
        self._counter = 0

    async def create_environment(self, config: EnvironmentConfig) -> EnvironmentInstance:
        self._counter += 1
        instance = EnvironmentInstance(
            instance_id=f"instance-{self._counter}",
            endpoint=f"http://fhir-{self._counter}/fhir/",
            environment=config.name,
        )
        self.instances[instance.instance_id] = {"messages": []}
        self.created.append(instance)
        return instance

    async def release_environment(self, config, instance) -> None:
        self.instances.pop(instance.instance_id, None)
        self.released.append(instance)

    async def get_tools(
        self, instance, patient_ids, decision_point=None, *, tool_result_limit=10000
    ):
        self.tool_sessions.append((instance, tuple(patient_ids), decision_point))
        if self.fail_get_tools:
            raise RuntimeError("tool setup failed")
        if self.tool_factory is not None:
            return self.tool_factory(instance, patient_ids, decision_point, tool_result_limit)
        return self._default_tools(instance, patient_ids)

    def _default_tools(self, instance, patient_ids):
        store = self.instances[instance.instance_id]
        allowed = set(patient_ids)

        @function_tool
        async def write_message(patient: str, text: str) -> str:
            """Append a message to this attempt's patient record."""
            if patient not in allowed:
                raise PatientAccessError(f"Patient {patient} is outside the allowed scope")
            self.tool_calls.append((instance.instance_id, "write_message", patient))
            store["messages"].append(text)
            return json.dumps({"patient": patient, "message_count": len(store["messages"])})

        @function_tool
        async def read_messages(patient: str) -> str:
            """Read this attempt's patient messages."""
            if patient not in allowed:
                raise PatientAccessError(f"Patient {patient} is outside the allowed scope")
            self.tool_calls.append((instance.instance_id, "read_messages", patient))
            return json.dumps({"messages": list(store["messages"])})

        return [write_message, read_messages]

    async def read_state(self, instance, task):
        self.state_reads.append((instance, task))
        store = self.instances[instance.instance_id]
        return {
            "patient_id": task.patient_id,
            "environment": instance.environment,
            "messages": list(store["messages"]),
            "reference_answer": task.reference_answer,
            "evidence": [item.model_dump() for item in task.evidence],
        }

    @property
    def active(self) -> int:
        return len(self.instances)


class IsolationRig:
    """Composed group isolation plus per-attempt medical tool session."""

    def __init__(self, root: Path, *, snapshot: Snapshot, environment: str = "fixture",
                 harness: HarnessName = "ace"):
        self.temp_dir = root
        self.docker = FakeDocker(root / "docker")
        self.env = FakeEnvironments()
        self.isolation = HarnessIsolation(
            run_id="run-1",
            image="harness:test",
            network="medevolve-run-1",
            state_root=root / "states",
            bridge_url="http://bridge:8080",
            gateway_url="http://gateway:8081",
            runner=self.docker,
        )
        self.sessions = ToolSessions(environments=self.env)
        self.group = GroupKey(
            run_id="run-1", environment=environment, harness=harness,
            source_method="medevolve", seed=42,
        )
        self.config = EnvironmentConfig(
            name=environment, manager_url="http://unused", seed_data="seed"
        )
        self.snapshot = snapshot
        self.workspace = self.isolation.prepare_group(self.group, snapshot)
        self.patients: dict[str, str] = {}

    @property
    def group_state(self) -> Path:
        return self.workspace.state_dir / "memory.txt"

    @property
    def active_instances(self) -> int:
        return self.env.active

    @property
    def container(self) -> str:
        return self.workspace.container_name

    async def open_session(self, *, patient: str = "p1", client_id: str | None = "container-1"):
        await self.isolation.start_attempt(self.workspace)
        session_id = await self.sessions.open(
            SolverInput(instruction="核对患者用药", patient_id=patient),
            self.config,
            client_id=client_id,
        )
        self.patients[session_id] = patient
        return session_id

    async def write_message(self, session_id: str, text: str) -> dict:
        return await self.sessions.invoke(
            session_id, "write_message",
            {"patient": self.patients[session_id], "text": text},
        )

    async def read_messages(self, session_id: str) -> list[str]:
        result = await self.sessions.invoke(
            session_id, "read_messages", {"patient": self.patients[session_id]}
        )
        return result["output"]["messages"]

    async def close_session(self, session_id: str) -> dict:
        return await self.sessions.close(session_id)

    async def close(self) -> None:
        await self.sessions.aclose()
        await self.isolation.close()


def make_snapshot(root: Path, text: str = "基座初始经验") -> Snapshot:
    """Create a base experience snapshot directory for one harness."""
    directory = root / "snapshot"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "memory.txt").write_text(text, encoding="utf-8")
    return Snapshot(path=directory, digest="sha256:base")


@pytest.fixture
def temp_dir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture
def fake_docker(tmp_path) -> FakeDocker:
    return FakeDocker(tmp_path / "docker")


@pytest.fixture
def base_snapshot(tmp_path) -> Snapshot:
    return make_snapshot(tmp_path)


@pytest.fixture
def group_factory():
    def make(name: HarnessName = "ace", **overrides) -> GroupKey:
        values = {
            "run_id": "run-1", "environment": "fixture", "harness": name,
            "source_method": "medevolve", "seed": 42,
        }
        values.update(overrides)
        return GroupKey(**values)

    return make


@pytest.fixture
def isolation(tmp_path, fake_docker) -> HarnessIsolation:
    return HarnessIsolation(
        run_id="run-1",
        image="harness:test",
        network="medevolve-run-1",
        state_root=tmp_path / "states",
        bridge_url="http://bridge:8080",
        gateway_url="http://gateway:8081",
        runner=fake_docker,
    )


@pytest.fixture
def rig(tmp_path, base_snapshot) -> IsolationRig:
    return IsolationRig(tmp_path, snapshot=base_snapshot)


@pytest.fixture
def fake_environments() -> FakeEnvironments:
    return FakeEnvironments()


@pytest.fixture
def fake_environments_factory():
    def make(**kwargs) -> FakeEnvironments:
        return FakeEnvironments(**kwargs)

    return make


@pytest.fixture
def fake_fhir() -> FakeFHIRService:
    return FakeFHIRService()


@pytest.fixture
def fhir_service(monkeypatch, fake_fhir) -> FakeFHIRService:
    """Route every FHIRClient HTTP call in this test to `FakeFHIRService`."""
    real_client = httpx.AsyncClient
    transport = fake_fhir.transport()

    def client_factory(*args, **kwargs):
        kwargs.setdefault("transport", transport)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client_factory)
    return fake_fhir


@pytest.fixture
def tool_sessions(fake_environments) -> ToolSessions:
    return ToolSessions(environments=fake_environments)


@pytest.fixture
def real_physicianbench(fhir_service, fake_environments):
    """Real PhysicianBench tool schemas and patient checks over a fake FHIR service."""
    from environments import get_tools

    environments = SimpleNamespace(
        create_environment=fake_environments.create_environment,
        release_environment=fake_environments.release_environment,
        read_state=fake_environments.read_state,
        get_tools=get_tools,
    )
    return SimpleNamespace(
        environments=environments, env=fake_environments, fhir=fhir_service,
        config=EnvironmentConfig(
            name="PhysicianBench", manager_url="http://unused", seed_data="seed"
        ),
    )


@pytest.fixture
def real_medagentbench(fhir_service, fake_environments):
    """Real MedAgentBench native action tools over a fake FHIR service."""
    from environments import get_tools

    environments = SimpleNamespace(
        create_environment=fake_environments.create_environment,
        release_environment=fake_environments.release_environment,
        read_state=fake_environments.read_state,
        get_tools=get_tools,
    )
    return SimpleNamespace(
        environments=environments, env=fake_environments, fhir=fhir_service,
        config=EnvironmentConfig(
            name="MedAgentBench", manager_url="http://unused", seed_data="seed"
        ),
    )


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def api_key(monkeypatch) -> str:
    monkeypatch.setenv("GATEWAY_API_KEY", "secret-key")
    return "secret-key"


class BridgeRig:
    """Tool-session fixture used by bridge, isolation and adapter tests."""

    def __init__(self, root: Path, environments):
        self.registry = ToolSessions(environments=environments)
        self.env = environments
        self.temp_dir = root
        self.config = EnvironmentConfig(
            name="fixture", manager_url="http://unused", seed_data="seed"
        )
        self.task = SolverInput(instruction="核对患者用药", patient_id="p1")
        self.session_id: str | None = None
        self._learning_model_calls = 0

    @property
    def learning_model_calls(self) -> int:
        return self._learning_model_calls

    def record_model_call(self, phase: str) -> None:
        if phase == "learning":
            self._learning_model_calls += 1

    async def open(self, **kwargs) -> str:
        self.session_id = await self.registry.open(self.task, self.config, **kwargs)
        return self.session_id

    async def invoke(self, name: str, arguments: dict, **kwargs) -> dict:
        return await self.registry.invoke(self.session_id, name, arguments, **kwargs)

    async def list_tools(self, **kwargs) -> list[dict]:
        return await self.registry.list(self.session_id, **kwargs)

    async def close(self, **kwargs) -> dict:
        return await self.registry.close(self.session_id, **kwargs)


@pytest.fixture
def bridge(tmp_path, fake_environments) -> BridgeRig:
    return BridgeRig(tmp_path, fake_environments)
