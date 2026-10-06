"""Bounded evaluation environment pool with read-only instance reuse."""

import asyncio
import os
from contextlib import asynccontextmanager
from importlib import import_module

from contracts import EnvironmentConfig, EnvironmentInstance

# 核验阶段只读患者原始记录，任何环境都可以整轮共用一个实例。
VERIFY_SHARING = {"PhysicianBench", "MedAgentBench", "FHIR-AgentBench"}
# 解题会写库的环境不能共用实例：写后状态由 read_state 读回，跨任务累积会污染完成判断。
# PhysicianBench 的解题会创建 MedicationRequest、ServiceRequest 等资源；
# MedAgentBench 的原生 POST 只校验请求、不回写数据库；FHIR-AgentBench 只读。
SOLVE_SHARING = {"MedAgentBench", "FHIR-AgentBench"}

DEFAULT_MAX_INSTANCES = 3


def max_instances() -> int:
    """Concurrent container budget for one run; override with MEDEVOLVE_MAX_INSTANCES."""
    try:
        value = int(os.environ.get("MEDEVOLVE_MAX_INSTANCES", ""))
    except ValueError:
        return DEFAULT_MAX_INSTANCES
    return value if value > 0 else DEFAULT_MAX_INSTANCES


class EvaluationEnvironments:
    """Reuse read-only instances inside one run and cap concurrent containers.

    A container budget is held for as long as its instance lives, so the process never
    runs more than ``max_instances`` containers at once. Shared instances are released
    by :meth:`close` when the run ends.
    """

    def __init__(self, instance_budget: int | None = None, sharing: bool = True):
        self._slots = asyncio.Semaphore(instance_budget or max_instances())
        self._lock = asyncio.Lock()
        self._shared: dict[str, tuple[EnvironmentConfig, EnvironmentInstance]] = {}
        self._sharing = sharing

    def _shareable(self, environment: EnvironmentConfig, phase: str) -> bool:
        if not self._sharing:
            return False
        table = VERIFY_SHARING if phase == "verify" else SOLVE_SHARING
        return environment.name in table

    @asynccontextmanager
    async def instance(self, environment: EnvironmentConfig, *, phase: str):
        if self._shareable(environment, phase):
            async with self._lock:
                entry = self._shared.get(environment.name)
                if entry is None:
                    await self._slots.acquire()
                    try:
                        created = await self._create(environment)
                    except BaseException:
                        self._slots.release()
                        raise
                    entry = (environment, created)
                    self._shared[environment.name] = entry
            yield entry[1]
            return
        await self._slots.acquire()
        try:
            created = await self._create(environment)
        except BaseException:
            self._slots.release()
            raise
        try:
            yield created
        finally:
            try:
                await self._release(environment, created)
            finally:
                self._slots.release()

    async def close(self) -> None:
        async with self._lock:
            shared, self._shared = self._shared, {}
        failures = []
        for environment, instance in shared.values():
            try:
                await self._release(environment, instance)
            except Exception as exc:  # 逐个尝试回收，最后统一报告
                failures.append(f"{environment.name}:{instance.instance_id}: {exc}")
        if failures:
            raise RuntimeError("Evaluation instance release failed: " + "; ".join(failures))

    @staticmethod
    async def _create(environment: EnvironmentConfig) -> EnvironmentInstance:
        return await import_module("environments").create_environment(environment)

    @staticmethod
    async def _release(
        environment: EnvironmentConfig, instance: EnvironmentInstance
    ) -> None:
        await import_module("environments").release_environment(environment, instance)
