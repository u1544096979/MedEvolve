"""Evaluation instance reuse and container budget."""

import asyncio
import sys
from types import SimpleNamespace

from contracts import EnvironmentConfig, EnvironmentInstance
from evaluation.session import EvaluationEnvironments


class FakeManager:
    def __init__(self, budget=None):
        self.created, self.released, self.live, self.peak = [], [], set(), 0
        self.fail_release = False

    async def create_environment(self, config):
        instance = EnvironmentInstance(
            instance_id=f"i{len(self.created)}", endpoint="http://fake", environment=config.name
        )
        self.created.append(instance.instance_id)
        self.live.add(instance.instance_id)
        self.peak = max(self.peak, len(self.live))
        return instance

    async def release_environment(self, config, instance):
        if self.fail_release:
            raise RuntimeError("release failed")
        self.released.append(instance.instance_id)
        self.live.discard(instance.instance_id)


def install(monkeypatch, manager):
    monkeypatch.setitem(sys.modules, "environments", SimpleNamespace(
        create_environment=manager.create_environment,
        release_environment=manager.release_environment,
    ))


def environment(name):
    return EnvironmentConfig(name=name, manager_url="http://manager.test", seed_data="image")


def test_all_phases_share_one_fixed_configuration_instance(monkeypatch):
    manager = FakeManager()
    install(monkeypatch, manager)

    async def main():
        session = EvaluationEnvironments(instance_budget=3)
        verify_ids = []
        for _ in range(3):
            async with session.instance(environment("PhysicianBench"), phase="verify") as instance:
                verify_ids.append(instance.instance_id)
        solve_ids = []
        for _ in range(2):
            async with session.instance(environment("PhysicianBench"), phase="solve") as instance:
                solve_ids.append(instance.instance_id)
        assert manager.live == {"i0"}
        await session.close()
        return verify_ids, solve_ids

    verify_ids, solve_ids = asyncio.run(main())
    assert verify_ids == ["i0", "i0", "i0"]
    assert solve_ids == ["i0", "i0"]
    assert manager.released == ["i0"]
    assert manager.live == set()


def test_read_only_environments_share_the_solve_instance(monkeypatch):
    manager = FakeManager()
    install(monkeypatch, manager)

    async def main():
        session = EvaluationEnvironments(instance_budget=3)
        for phase in ("verify", "solve"):
            for _ in range(2):
                async with session.instance(environment("MedAgentBench"), phase=phase) as instance:
                    assert instance.instance_id == "i0"
        await session.close()

    asyncio.run(main())
    assert manager.created == ["i0"]
    assert manager.released == ["i0"]


def test_container_budget_caps_concurrent_instances(monkeypatch):
    manager = FakeManager()
    install(monkeypatch, manager)

    async def hold(session):
        async with session.instance(environment("PhysicianBench"), phase="generate"):
            await asyncio.sleep(0.02)

    async def main():
        session = EvaluationEnvironments(instance_budget=2)
        await asyncio.gather(*(hold(session) for _ in range(5)))
        await session.close()

    asyncio.run(main())
    assert len(manager.created) == 1
    assert manager.peak == 1
    assert manager.live == set()


def test_close_reports_release_failure(monkeypatch):
    manager = FakeManager()
    install(monkeypatch, manager)

    async def main():
        session = EvaluationEnvironments(instance_budget=3)
        async with session.instance(environment("PhysicianBench"), phase="verify"):
            pass
        manager.fail_release = True
        try:
            await session.close()
        except RuntimeError as exc:
            assert "release failed" in str(exc)
        else:
            raise AssertionError("close must surface release failures")
        assert manager.live == {"i0"}
        manager.fail_release = False
        await session.close()

    asyncio.run(main())


def test_configuration_key_includes_manager_and_seed(monkeypatch):
    manager = FakeManager()
    install(monkeypatch, manager)

    async def main():
        session = EvaluationEnvironments(instance_budget=3)
        base = environment("PhysicianBench")
        for config in (base, base.model_copy(update={"manager_url": "http://other"}),
                       base.model_copy(update={"seed_data": "other-image"})):
            async with session.instance(config, phase="prepare"):
                pass
        await session.close()

    asyncio.run(main())
    assert manager.created == ["i0", "i1", "i2"]
    assert manager.released == ["i0", "i1", "i2"]


def test_close_waits_for_active_borrower(monkeypatch):
    manager = FakeManager()
    install(monkeypatch, manager)

    async def main():
        session = EvaluationEnvironments()
        entered, finish = asyncio.Event(), asyncio.Event()

        async def borrower():
            async with session.instance(environment("PhysicianBench"), phase="solve"):
                entered.set()
                await finish.wait()

        task = asyncio.create_task(borrower())
        await entered.wait()
        closing = asyncio.create_task(session.close())
        await asyncio.sleep(0)
        assert not closing.done()
        finish.set()
        await task
        await closing

    asyncio.run(main())
    assert manager.released == ["i0"]


def test_cancelled_creation_releases_its_logical_slot(monkeypatch):
    started = asyncio.Event()
    count = 0

    async def create(config):
        nonlocal count
        count += 1
        if count == 1:
            started.set()
            await asyncio.Event().wait()
        return EnvironmentInstance(
            instance_id=f"i{count}", endpoint="http://fake", environment=config.name
        )

    async def release(config, instance):
        pass

    install(monkeypatch, SimpleNamespace(
        create_environment=create, release_environment=release))

    async def main():
        session = EvaluationEnvironments(instance_budget=1)

        async def borrow():
            async with session.instance(environment("PhysicianBench"), phase="prepare"):
                pass

        first = asyncio.create_task(borrow())
        await started.wait()
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        async with session.instance(environment("PhysicianBench"), phase="solve") as instance:
            assert instance.instance_id == "i2"
        await session.close()

    asyncio.run(main())
