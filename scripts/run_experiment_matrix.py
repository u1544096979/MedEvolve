"""从 JSON 配置运行模型隔离的全量实验矩阵。"""

import argparse
import asyncio
import json
from pathlib import Path

from experiments.matrix import MatrixConfig, run_matrix


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="矩阵 JSON 配置")
    args = parser.parse_args(argv)
    config = MatrixConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    result = asyncio.run(run_matrix(config))
    print(json.dumps({"status": result["status"], "jobs": len(result["jobs"]),
                      "errors": len(result["errors"])}, ensure_ascii=False))
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
