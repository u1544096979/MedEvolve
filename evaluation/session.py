"""Bounded per-run environment instances shared by every experiment phase."""

import asyncio
import os
from contextlib import asynccontextmanager
from importlib import import_module

from contracts import EnvironmentConfig, EnvironmentInstance

DEFAULT_MAX_INSTANCES = 3


def max_instances() -> int:
    """Concurrent logical-instance budget for one run."""
    try:
        value = int(os.environ.get("MEDEVOLVE_MAX_INSTANCES", ""))
    except ValueError:
        return DEFAULT_MAX_INSTANCES
    return value if value > 0 else DEFAULT_MAX_INSTANCES


class EvaluationEnvironments:
    """Reuse one configured instance across all phases of a run."""

    def __init__(self, instance_budget: int | None = None, sharing: bool = True,
                 create=None, release=None):
        self._instance_budget = instance_budget or max_instances()
        self._slots = asyncio.Semaphore(self._instance_budget)
        self._condition = asyncio.Condition()
        self._close_lock = asyncio.Lock()
        self._shared: dict[tuple[str, str, str], tuple[EnvironmentConfig, EnvironmentInstance, int]] = {}
        self._sharing = sharing
        self._closing = False
        self._create_fn = create
        self._release_fn = release

    @staticmethod
    def _key(environment: EnvironmentConfig) -> tuple[str, str, str]:
        return environment.name, environment.manager_url, environment.seed_data

    @asynccontextmanager
    async def instance(self, environment: EnvironmentConfig, *, phase: str = "solve"):
        """Borrow the run's fixed-config instance; all phase labels share it."""
        if not self._sharing:
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
            return

        key = self._key(environment)
        async with self._condition:
            if self._closing:
                raise RuntimeError("Evaluation environment session is closing")
            entry = self._shared.get(key)
            if entry is not None:
                self._shared[key] = (entry[0], entry[1], entry[2] + 1)
                instance = entry[1]
            else:
                if len(self._shared) >= self._instance_budget:
                    raise RuntimeError(
                        "Evaluation environment session exhausted its logical instance budget"
                    )
                # Hold the condition while creating so concurrent first borrowers cannot
                # create duplicate instances. The slot counts creation as a live instance.
                await self._slots.acquire()
                try:
                    instance = await self._create(environment)
                except BaseException:
                    self._slots.release()
                    raise
                self._shared[key] = (environment, instance, 1)
        try:
            yield instance
        finally:
            async with self._condition:
                current = self._shared.get(key)
                if current is not None and current[1] is instance:
                    self._shared[key] = (current[0], current[1], current[2] - 1)
                    self._condition.notify_all()

    async def close(self) -> None:
        async with self._close_lock:
            async with self._condition:
                self._closing = True
                await self._condition.wait_for(
                    lambda: all(entry[2] == 0 for entry in self._shared.values())
                )
            failures = []
            for key, (environment, instance, borrowers) in list(self._shared.items()):
                if borrowers:
                    continue
                try:
                    await self._release(environment, instance)
                except Exception as exc:
                    failures.append(f"{environment.name}:{instance.instance_id}: {exc}")
                else:
                    self._shared.pop(key, None)
                    self._slots.release()
        if failures:
            raise RuntimeError("Evaluation instance release failed: " + "; ".join(failures))

    async def _create(self, environment: EnvironmentConfig) -> EnvironmentInstance:
        create = self._create_fn or import_module("environments").create_environment
        return await create(environment)

    async def _release(self, environment: EnvironmentConfig, instance: EnvironmentInstance) -> None:
        release = self._release_fn or import_module("environments").release_environment
        await release(environment, instance)
