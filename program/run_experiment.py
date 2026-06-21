from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


COMMANDS = {
    "prepare": ["prepare_results.py"],
    "tables": ["generate_tables.py"],
    "figures": ["generate_figures.py"],
    "assets": ["build_publication_assets.py"],
    "all": ["build_publication_assets.py"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the dispatch matching publication pipeline.")
    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=sorted(COMMANDS),
        help="Pipeline step to run. Default: all.",
    )
    args = parser.parse_args()

    for script in COMMANDS[args.mode]:
        path = SCRIPTS / script
        print(f"[run_experiment] {script}")
        subprocess.run([sys.executable, str(path)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
