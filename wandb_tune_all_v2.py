"""Launch W&B tuning for all four V2 scripts.

By default this starts the four sweep agents in parallel so --hours is a wall-clock
budget for all datasets together. Pass --sequential to run one after another.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    "wandb_tune_trainV2_congress.py",
    "wandb_tune_trainV2_DBLP_CC.py",
    "wandb_tune_trainV2_house_bills.py",
    "wandb_tune_trainV2_walmart.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W&B sweeps for all V2 scripts.")
    parser.add_argument("--hours", type=float, default=10.0)
    parser.add_argument("--entity", default=None)
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--devices", default=None, help="Comma-separated device overrides, one per script.")
    return parser.parse_args()


def command(script: str, args: argparse.Namespace, device: Optional[str]) -> list[str]:
    cmd = [sys.executable, str(ROOT / script), "--hours", str(args.hours), "--wandb-mode", args.wandb_mode]
    if args.entity:
        cmd.extend(["--entity", args.entity])
    if device:
        cmd.extend(["--device", device])
    return cmd


def main() -> None:
    args = parse_args()
    devices = args.devices.split(",") if args.devices else [None] * len(SCRIPTS)
    if len(devices) != len(SCRIPTS):
        raise SystemExit("--devices must provide exactly four comma-separated values.")

    commands = [command(script, args, device) for script, device in zip(SCRIPTS, devices)]
    if args.sequential:
        for cmd in commands:
            subprocess.run(cmd, cwd=ROOT, check=True)
        return

    procs = [subprocess.Popen(cmd, cwd=ROOT) for cmd in commands]
    failures = []
    for proc, cmd in zip(procs, commands):
        code = proc.wait()
        if code:
            failures.append((code, cmd))
    if failures:
        for code, cmd in failures:
            print(f"failed ({code}): {' '.join(cmd)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
