"""本机原生 Agent runner 工厂。"""
import os
from pathlib import Path
from importlib import import_module

RUNNERS = {
    "openclaw": ("harnesses.runners.openclaw", "OpenClawRunner"),
    "hermes": ("harnesses.runners.hermes", "HermesRunner"),
    "ace": ("harnesses.runners.ace", "ACERunner"),
    "reflexion": ("harnesses.runners.reflexion", "ReflexionRunner"),
}


def create_runner(name, *, config, group_dir, bridge_url):
    try:
        module_name, class_name = RUNNERS[name]
    except KeyError as exc:
        raise ValueError(f"不支持的本地 Harness：{name}") from exc
    runner_config = dict(config)
    if "execution_model" not in runner_config:
        models = runner_config.get("models", {})
        runner_config["execution_model"] = models.get("execution", {})
        runner_config["learning_model"] = models.get("learning", {})
        limits = runner_config.get("limits", {})
        runner_config["execution_budget"] = limits.get("execution", {})
        runner_config["learning_budget"] = limits.get("learning", {})
        runner_config["timeout_seconds"] = limits.get("timeout_seconds", 900)
        runner_config["learning_timeout_seconds"] = limits.get("learning_timeout_seconds", 300)
        local = runner_config.get("local_harnesses", {}).get(name, {})
        for key in ("source_dir", "program_path", "python_path"):
            value = local.get(key)
            if value:
                runner_config[key] = (os.path.abspath(Path(value).expanduser()) if key == "python_path"
                                      else str(Path(value).expanduser().resolve()))
        if name == "openclaw" and runner_config.get("program_path"):
            runner_config["openclaw_bin"] = runner_config["program_path"]
    cls = getattr(import_module(module_name), class_name)
    return cls(config=runner_config, group_dir=group_dir, bridge_url=bridge_url)
