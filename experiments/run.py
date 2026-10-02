"""Run an experiment from a JSON configuration."""

import argparse
import asyncio
from pathlib import Path

from contracts import ExperimentConfig
from .runner import run_experiment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = ExperimentConfig.model_validate_json(Path(args.config).read_text(encoding="utf-8"))
    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
