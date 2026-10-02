"""Print a saved experiment's summary without model or environment calls."""

import argparse
import json

from .summary import summarize_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(summarize_run(args.run_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
