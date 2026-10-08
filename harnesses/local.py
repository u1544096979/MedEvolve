"""本地 Harness 子进程与实验目录的轻量级辅助函数。"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from harnesses.snapshots import collect_files, materialize_files


async def run_command(
    command: Sequence[str], *, group_dir: Path, group_env: Mapping[str, str], timeout: float
) -> tuple[int, str, str]:
    """在组目录运行命令，继承宿主环境并覆盖组级环境变量。"""
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(group_dir),
        env={**os.environ, **group_env},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        process.kill()
        await process.communicate()
        raise
    return process.returncode, stdout.decode("utf-8", errors="replace"), stderr.decode(
        "utf-8", errors="replace"
    )


def prepare_group_directory(template: Path, destination: Path) -> Path:
    """从统一模板创建空的实验状态目录；目标目录必须尚不存在。"""
    template, destination = Path(template), Path(destination)
    if destination.exists():
        raise FileExistsError(f"实验目录已存在：{destination}")
    if not template.is_dir():
        raise FileNotFoundError(f"初始状态模板不存在：{template}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(template, destination)
    return destination


@contextmanager
def temporary_experience(snapshot: Path) -> Iterator[Path]:
    """将冻结经验复制到临时目录，退出时无论成功或异常都清理。"""
    with tempfile.TemporaryDirectory(prefix="medevolve-experience-") as temporary:
        working = Path(temporary) / "experience"
        shutil.copytree(snapshot, working)
        yield working


def export_experience(source: Path, destination: Path, *, include: Sequence[str]) -> list[str]:
    """按显式相对路径导出经验文件，原始会话和答案缓存由快照过滤器排除。"""
    files = collect_files(Path(source), include=include)
    return materialize_files(Path(destination), files)
