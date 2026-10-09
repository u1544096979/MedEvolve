"""Run an experiment from a JSON configuration."""

import argparse
import asyncio
import json
from pathlib import Path

from contracts import ExperimentConfig
from .runner import METHOD_MODULES, run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=tuple(METHOD_MODULES))
    parser.add_argument("--run-id")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = ExperimentConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
    overrides = {key: value for key, value in {"method": args.method, "run_id": args.run_id,
                                               "output_dir": args.output_dir}.items() if value is not None}
    config = config.model_copy(update=overrides)
    asyncio.run(run_experiment(config))
    manifest = Path(config.output_dir) / "manifest.json"
    return 1 if manifest.exists() and json.loads(manifest.read_text(encoding="utf-8")).get("status") == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
