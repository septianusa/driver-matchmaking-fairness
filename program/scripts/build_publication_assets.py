from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


PIPELINE = [
    "prepare_results.py",
    "generate_tables.py",
    "generate_figures.py",
]


def main() -> int:
    for script in PIPELINE:
        path = SCRIPT_DIR / script
        print(f"[build] running {path.name}")
        subprocess.run([sys.executable, str(path)], check=True)
    print("[build] publication assets are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
