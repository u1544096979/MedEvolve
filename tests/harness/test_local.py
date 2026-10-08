"""Local process and experience directory helpers."""

import asyncio

import pytest

from harnesses.local import (export_experience, prepare_group_directory, run_command,
                             temporary_experience)


def test_group_directory_copies_template_without_reusing_existing_destination(tmp_path):
    template = tmp_path / "template"
    template.mkdir()
    (template / "settings.json").write_text("{}")
    destination = prepare_group_directory(template, tmp_path / "group")
    assert (destination / "settings.json").read_text() == "{}"
    with pytest.raises(FileExistsError):
        prepare_group_directory(template, destination)


def test_test_experience_is_temporary_and_export_is_explicit(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "skills.md").write_text("skill")
    (snapshot / "sessions" / "answer_cache").mkdir(parents=True)
    (snapshot / "sessions" / "answer_cache" / "answer.json").write_text("secret")
    with temporary_experience(snapshot) as working:
        assert (working / "skills.md").read_text() == "skill"
        working_path = working
    assert not working_path.exists()
    exported = export_experience(snapshot, tmp_path / "export", include=("skills.md", "sessions"))
    assert exported == ["skills.md"]
    assert list((tmp_path / "export").rglob("*")) == [tmp_path / "export" / "skills.md"]


@pytest.mark.asyncio
async def test_run_command_uses_group_directory_env_and_returns_output(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDEVOLVE_TEST_HOST_VALUE", "inherited")
    code, stdout, stderr = await run_command(
        ["/bin/sh", "-c", "printf '%s:%s' \"$MEDEVOLVE_TEST_HOST_VALUE\" \"$GROUP_VALUE\""],
        group_dir=tmp_path,
        group_env={"GROUP_VALUE": "group"},
        timeout=3,
    )
    assert (code, stdout, stderr) == (0, "inherited:group", "")


@pytest.mark.asyncio
async def test_run_command_kills_timed_out_child(tmp_path):
    with pytest.raises(asyncio.TimeoutError):
        await run_command(["/bin/sh", "-c", "sleep 10"], group_dir=tmp_path,
                          group_env={}, timeout=0.01)
