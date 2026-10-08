"""Two-layer isolation for harness runs.

Layer one is the group: every (run, environment, harness, source method, seed)
gets its own host state directory, seeded once from the base snapshot and mounted
at `/state`. Group state survives task attempts; other groups never see it.

Layer two is the attempt: every task attempt gets a new scratch volume mounted at
`/scratch`, a new container and a new medical FHIR instance. The medical instance
belongs to `harnesses.medical_bridge`, which creates it before the attempt and
releases it when the tool session closes.

Nothing outside the run is mounted: no host home directory, no project dataset,
no medical database and no Docker socket. Containers and volumes carry this run's
labels and are removed by the concrete IDs this instance recorded, never by a
global prune.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Mapping, Sequence

from harnesses.contracts import GroupKey, Snapshot

MANAGED_LABEL = "medevolve.managed"
RUN_LABEL = "medevolve.run_id"
GROUP_LABEL = "medevolve.group"
HARNESS_LABEL = "medevolve.harness"
CONTAINER_PREFIX = "medevolve-harness"
STATE_MOUNT = "/state"
SCRATCH_MOUNT = "/scratch"
# Explicit command appended to `docker run`: an image whose default CMD exits
# immediately would make every later `docker exec` fail, so the shared plumbing
# never relies on the image's own command.
KEEPALIVE_COMMAND = ("sleep", "infinity")
DOCKER_SOCKETS = (Path("/var/run/docker.sock"), Path("/run/docker.sock"))
PERSONAL_DIRECTORIES = (
    ".openclaw", ".hermes", ".claude", ".codex", ".agents", ".config", ".ssh",
    ".aws", ".gnupg",
)
STATE_PATHS: dict[str, str] = {
    "openclaw": "/state/openclaw",
    "hermes": "/state/hermes",
    "ace": "/state/ace",
    "reflexion": "/state/reflexion",
}
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
IMAGE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@-]*\Z")


class IsolationError(RuntimeError):
    """Harness isolation could not be established or torn down."""


class DockerCommandError(IsolationError):
    """The Docker CLI failed."""


async def docker_command(*args: str, stdin: str | None = None, timeout: float = 120.0) -> str:
    """Run one Docker CLI call with an argument array; never a shell string."""
    try:
        process = await asyncio.create_subprocess_exec(
            "docker",
            *args,
            stdin=asyncio.subprocess.PIPE if stdin is not None else None,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise DockerCommandError("Docker CLI is not installed") from exc
    try:
        stdout, stderr = await asyncio.wait_for(
            process.communicate(stdin.encode() if stdin is not None else None),
            timeout=timeout,
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        process.kill()
        await process.communicate()
        raise
    if process.returncode:
        raise DockerCommandError(stderr.decode().strip() or f"docker {' '.join(args)} failed")
    return stdout.decode().strip()


def state_env(harness: str) -> dict[str, str]:
    """Container environment variables the harness uses to find its own state."""
    if harness == "openclaw":
        return {
            "OPENCLAW_STATE_DIR": STATE_PATHS["openclaw"],
            "OPENCLAW_CONFIG_PATH": STATE_PATHS["openclaw"] + "/openclaw.json",
        }
    if harness == "hermes":
        return {"HERMES_HOME": STATE_PATHS["hermes"]}
    if harness not in STATE_PATHS:
        raise IsolationError(f"未知基座：{harness}")
    return {}


def _copy_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, dirs_exist_ok=True)


@dataclass
class GroupWorkspace:
    """Container, mounts and paths for one isolated group."""

    group: GroupKey
    state_dir: Path
    state_mount_source: Path
    state_readonly: bool
    scratch_dir: Path
    container_name: str
    volume_name: str
    attempt: int
    env: dict[str, str]
    labels: dict[str, str]
    frozen: bool = False


class HarnessIsolation:
    """Materialise and tear down isolated harness containers for one run."""

    def __init__(
        self,
        *,
        run_id: str,
        image: str,
        network: str,
        state_root: Path,
        bridge_url: str,
        gateway_url: str,
        runner: Callable[..., Awaitable[str]] | None = None,
        copier: Callable[[Path, Path], None] | None = None,
        extra_env: Mapping[str, str] | None = None,
        forbidden_paths: Sequence[Path] = (),
        keepalive: Sequence[str] = KEEPALIVE_COMMAND,
    ):
        if not RUN_ID_PATTERN.fullmatch(run_id or ""):
            raise IsolationError(f"非法 run_id：{run_id!r}")
        if not IMAGE_PATTERN.fullmatch(image or ""):
            raise IsolationError(f"非法镜像引用：{image!r}")
        if not IMAGE_PATTERN.fullmatch(network or ""):
            raise IsolationError(f"非法网络名：{network!r}")
        if not bridge_url or not gateway_url:
            raise IsolationError("工具桥和模型网关地址为必需项")
        self.run_id = run_id
        self.image = image
        self.network = network
        self.state_root = Path(state_root)
        self.bridge_url = bridge_url
        self.gateway_url = gateway_url
        self._runner = runner or docker_command
        self._copier = copier or _copy_tree
        self.extra_env = dict(extra_env or {})
        self.forbidden_paths = tuple(Path(path) for path in forbidden_paths)
        self.keepalive = tuple(keepalive)
        self._workspaces: dict[str, GroupWorkspace] = {}

    # -- paths and names ---------------------------------------------------
    def state_dir_for(self, group: GroupKey) -> Path:
        return self.state_root / self.group_name(group)

    def group_name(self, group: GroupKey) -> str:
        readable = "-".join(
            re.sub(r"[^A-Za-z0-9]+", "_", str(value))
            for value in (group.environment, group.harness, group.source_method, group.seed)
        )
        digest = hashlib.sha1("|".join((
            group.run_id, group.environment, group.harness, group.source_method,
            str(group.seed),
        )).encode()).hexdigest()[:10]
        return f"{readable}-{digest}"

    def _container_name(self, group: GroupKey, attempt: int) -> str:
        digest = hashlib.sha1(self.group_name(group).encode()).hexdigest()[:10]
        return f"{CONTAINER_PREFIX}-{self.run_id[:16]}-{digest}-{attempt}"

    def _volume_name(self, group: GroupKey, attempt: int) -> str:
        digest = hashlib.sha1(self.group_name(group).encode()).hexdigest()[:10]
        return f"{CONTAINER_PREFIX}-{self.run_id[:16]}-{digest}-scratch-{attempt}"

    def _scratch_dir(self, group: GroupKey, attempt: int) -> Path:
        return self.state_root.parent / "scratch" / self.group_name(group) / f"attempt-{attempt}"

    # -- group lifecycle ---------------------------------------------------
    def prepare_group(self, group: GroupKey, initial: Snapshot, *, frozen: bool = False) -> GroupWorkspace:
        """Seed group state and allocate paths; no container is created yet."""
        self._validate_group(group)
        source = Path(initial.path)
        self._assert_allowed_source(source)
        if not source.is_dir():
            raise IsolationError(f"初始快照不存在：{source}")
        if frozen:
            state_source, state_readonly = source, True
        else:
            state_dir = self.state_dir_for(group)
            self._assert_allowed_source(state_dir)
            self._seed(source, state_dir)
            state_source, state_readonly = state_dir, False
        return GroupWorkspace(
            group=group,
            state_dir=source if frozen else self.state_dir_for(group),
            state_mount_source=state_source,
            state_readonly=state_readonly,
            scratch_dir=Path(),
            container_name="",
            volume_name="",
            attempt=0,
            env=self._env(group, frozen=frozen),
            labels=self._labels(group),
            frozen=frozen,
        )

    async def start_group(self, group: GroupKey, initial: Snapshot, *, frozen: bool = False) -> GroupWorkspace:
        workspace = self.prepare_group(group, initial, frozen=frozen)
        return await self.start_attempt(workspace)

    async def start_attempt(self, workspace: GroupWorkspace) -> GroupWorkspace:
        """Start a fresh container and scratch volume for the next task attempt."""
        await self._stop_attempt(workspace)
        workspace.attempt += 1
        workspace.container_name = self._container_name(workspace.group, workspace.attempt)
        workspace.volume_name = self._volume_name(workspace.group, workspace.attempt)
        workspace.scratch_dir = self._scratch_dir(workspace.group, workspace.attempt)
        try:
            await self._runner("volume", "create", *self._flat_labels(workspace), workspace.volume_name)
            await self._run_container(workspace)
        except BaseException:
            await self._cleanup_attempt(workspace)
            raise
        self._workspaces[workspace.container_name] = workspace
        return workspace

    async def stop_group(self, workspace: GroupWorkspace) -> None:
        await self._stop_attempt(workspace)

    async def close(self) -> None:
        for workspace in reversed(list(self._workspaces.values())):
            await self._stop_attempt(workspace)

    async def exec(self, workspace: GroupWorkspace, *command: str, stdin: str | None = None) -> str:
        if not command:
            raise IsolationError("exec 需要命令")
        return await self._runner(
            "exec", "-i", workspace.container_name, *command, stdin=stdin
        )

    # -- internals ---------------------------------------------------------
    def _validate_group(self, group: GroupKey) -> None:
        if group.run_id != self.run_id:
            raise IsolationError(
                f"分组 run_id={group.run_id} 不属于本运行 {self.run_id}，拒绝创建容器"
            )
        if group.harness not in STATE_PATHS:
            raise IsolationError(f"未知基座：{group.harness}")

    def _seed(self, source: Path, destination: Path) -> None:
        if destination.is_dir() and any(destination.iterdir()):
            return
        destination.mkdir(parents=True, exist_ok=True)
        self._copier(source, destination)

    def _env(self, group: GroupKey, *, frozen: bool) -> dict[str, str]:
        environment = {
            "MEDEVOLVE_RUN_ID": self.run_id,
            "MEDEVOLVE_GROUP": self.group_name(group),
            "MEDEVOLVE_HARNESS": group.harness,
            "MEDEVOLVE_ENVIRONMENT": group.environment,
            "MEDEVOLVE_TOOL_BRIDGE": self.bridge_url,
            "MEDEVOLVE_MODEL_GATEWAY": self.gateway_url,
            "MEDEVOLVE_WRITABLE_DIR": SCRATCH_MOUNT,
            "MEDEVOLVE_STATE_DIR": STATE_MOUNT,
            "MEDEVOLVE_FROZEN": "1" if frozen else "0",
        }
        environment.update(state_env(group.harness))
        environment.update(self.extra_env)
        return environment

    def _labels(self, group: GroupKey) -> dict[str, str]:
        return {
            MANAGED_LABEL: "true",
            RUN_LABEL: self.run_id,
            GROUP_LABEL: self.group_name(group),
            HARNESS_LABEL: group.harness,
        }

    @staticmethod
    def _flat_labels(workspace: GroupWorkspace) -> tuple[str, ...]:
        args: list[str] = []
        for key, value in workspace.labels.items():
            args.extend(("--label", f"{key}={value}"))
        return tuple(args)

    @staticmethod
    def _mount(kind: str, source: str, target: str, readonly: bool) -> str:
        spec = f"type={kind},source={source},target={target}"
        return spec + ",readonly" if readonly else spec

    async def _run_container(self, workspace: GroupWorkspace) -> None:
        args = [
            "run", "--detach", "--pull=never", "--name", workspace.container_name,
            *self._flat_labels(workspace),
            "--network", self.network,
            "--workdir", "/workspace",
            "--mount", self._mount(
                "bind", str(workspace.state_mount_source), STATE_MOUNT,
                workspace.state_readonly,
            ),
            "--mount", self._mount("volume", workspace.volume_name, SCRATCH_MOUNT, False),
        ]
        for key, value in workspace.env.items():
            args.extend(("--env", f"{key}={value}"))
        args.append(self.image)
        args.extend(self.keepalive)
        await self._runner(*args)

    async def _stop_attempt(self, workspace: GroupWorkspace) -> None:
        await self._remove_container(workspace.container_name)
        await self._remove_volume(workspace.volume_name)
        self._workspaces.pop(workspace.container_name, None)

    async def _cleanup_attempt(self, workspace: GroupWorkspace) -> None:
        """Remove whatever the failed attempt created; safe to call twice."""
        await self._remove_container(workspace.container_name)
        await self._remove_volume(workspace.volume_name)
        self._workspaces.pop(workspace.container_name, None)

    async def _remove_container(self, name: str) -> None:
        if not name:
            return
        try:
            await self._runner("container", "inspect", name)
        except DockerCommandError as exc:
            if "No such container" in str(exc) or "No such object" in str(exc):
                return
            raise
        await self._runner("container", "rm", "--force", "--volumes", name)

    async def _remove_volume(self, name: str) -> None:
        if not name:
            return
        try:
            await self._runner("volume", "rm", name)
        except DockerCommandError as exc:
            if "No such volume" in str(exc) or "No such object" in str(exc):
                return
            raise

    def _forbidden_entries(self) -> list[tuple[Path, bool]]:
        """(path, also-forbid-children) for every source that must stay on the host."""
        home = Path.home()
        entries = [(home, False)]
        entries.extend((home / name, True) for name in PERSONAL_DIRECTORIES)
        entries.extend((socket, False) for socket in DOCKER_SOCKETS)
        entries.extend((Path(path), True) for path in self.forbidden_paths)
        return entries

    def _assert_allowed_source(self, source: Path) -> None:
        resolved = Path(source).expanduser().resolve()
        for path, with_children in self._forbidden_entries():
            forbidden = path.expanduser().resolve()
            if resolved == forbidden or (with_children and forbidden in resolved.parents):
                raise IsolationError(
                    f"不允许挂载 {resolved}：个人目录、项目数据集、医疗数据库和 "
                    "Docker socket 都不能成为 Agent 容器的挂载源"
                )
