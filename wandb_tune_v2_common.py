"""Shared W&B sweep runner for the V2 conformal HGNN scripts.

The runner keeps dataset/split/calibration defaults inside each trainV2_*.py file.
Only conformal-method hyperparameters are tuned, and --num_runs is fixed at 20.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
import statistics
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent

SCRIPT_SETTINGS: Dict[str, Dict[str, Any]] = {
    "congress": {
        "program": "trainV2_congress.py",
        "project": "cont-conf-hgnn-congress",
        "edge_topk_values": [250, 500, 1000],
        "warmup_values": [50, 100, 200],
        "coverage_warmup_values": [200, 300, 500],
    },
    "dblp": {
        "program": "trainV2_DBLP_CC.py",
        "project": "cont-conf-hgnn-dblp",
        "edge_topk_values": [1000, 2500, 5000, 10000],
        "warmup_values": [50, 100, 200],
        "coverage_warmup_values": [200, 300, 500],
    },
    "house_bills": {
        "program": "trainV2_house_bills.py",
        "project": "cont-conf-hgnn-house-bills",
        "edge_topk_values": [1000, 2500, 5000, 10000],
        "warmup_values": [50, 100, 200],
        "coverage_warmup_values": [200, 300, 500],
    },
    "walmart": {
        "program": "trainV2_walmart.py",
        "project": "cont-conf-hgnn-walmart",
        "edge_topk_values": [1000, 2500, 5000, 10000],
        "warmup_values": [500, 1000, 1500],
        "coverage_warmup_values": [1500, 3000, 4000],
    },
}

PROTECTED_ARGS = {
    "num_runs",
    "train_fraction",
    "valid_fraction",
    "max_calib_size",
    "calib_fraction",
    "data_seed",
    "valid_split_seed",
}

BOOL_FLAGS = {"cond_cov_loss"}


def _wandb():
    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "wandb is not installed. Install it in this environment, for example: "
            "pip install wandb, then run wandb login."
        ) from exc
    return wandb


def sweep_config(script_key: str) -> Dict[str, Any]:
    settings = SCRIPT_SETTINGS[script_key]
    return {
        "method": "bayes",
        "metric": {"name": "eff_valid", "goal": "minimize"},
        "parameters": {
            "conf_correct_model": {"values": ["hnn", "mlp"]},
            "confgnn_lr": {"distribution": "log_uniform_values", "min": 1e-5, "max": 5e-3},
            "confgnn_weight_decay": {"distribution": "log_uniform_values", "min": 1e-7, "max": 1e-3},
            "confgnn_dropout": {"values": [0.2, 0.3, 0.5, 0.7]},
            "confnn_hidden_dim": {"values": [32, 64, 128, 256]},
            "confgnn_num_layers": {"values": [1, 2, 3, 4]},
            "tau": {"values": [0.03, 0.05, 0.1, 0.2, 0.5]},
            "target_size": {"values": [0, 1, 2, 3]},
            "size_loss_weight": {"values": [0.5, 1.0, 2.0, 3.0, 4.0, 6.0]},
            "temperature": {"values": [0.1, 0.2, 0.3, 0.5, 1.0]},
            "aug_ratio": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
            "edge_topk": {"values": settings["edge_topk_values"]},
            "warmup_epochs": {"values": settings["warmup_values"]},
            "coverage_warmup_epochs": {"values": settings["coverage_warmup_values"]},
            "contrastive_loss_weight": {"values": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]},
            "late_contrastive_loss_weight": {"values": [0.0, 0.001, 0.01, 0.05, 0.1]},
            "non_conftr_size_loss_weight": {"values": [0.5, 1.0, 2.0, 4.0]},
            "raps_lam_reg": {"distribution": "log_uniform_values", "min": 1e-4, "max": 1e-1},
            "raps_k": {"values": [1, 2, 5]},
            "cond_cov_loss": {"values": [False, True]},
        },
    }


def parse_args(script_key: str) -> argparse.Namespace:
    settings = SCRIPT_SETTINGS[script_key]
    parser = argparse.ArgumentParser(description=f"Run W&B tuning for {settings['program']}.")
    parser.add_argument("--project", default=settings["project"])
    parser.add_argument("--entity", default=None)
    parser.add_argument("--sweep-id", default=None, help="Resume an existing sweep instead of creating one.")
    parser.add_argument("--hours", type=float, default=10.0, help="Wall-clock tuning budget for this script.")
    parser.add_argument("--count", type=int, default=None, help="Optional maximum number of W&B trials.")
    parser.add_argument("--device", default=None, help="Optional device override passed to the V2 script.")
    parser.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    parser.add_argument("--extra-arg", action="append", default=[], help="Extra raw arg passed through to the V2 script. Repeat as needed.")
    return parser.parse_args()


def build_command(script_key: str, config: Dict[str, Any], args: argparse.Namespace) -> List[str]:
    program = ROOT / SCRIPT_SETTINGS[script_key]["program"]
    cmd = [sys.executable, str(program), "--num_runs", "20"]
    if args.device:
        cmd.extend(["--device", args.device])

    for key, value in sorted(config.items()):
        if key in PROTECTED_ARGS:
            continue
        if key in BOOL_FLAGS:
            if value:
                cmd.append(f"--{key}")
            continue
        cmd.extend([f"--{key}", str(value)])

    cmd.extend(args.extra_arg)
    return cmd


def clean_literal(line: str) -> str:
    line = re.sub(r"np\.float\d*\(([^()]+)\)", r"\1", line)
    line = re.sub(r"np\.int\d*\(([^()]+)\)", r"\1", line)
    return line


def parse_trial_metrics(stdout: str) -> Dict[str, float]:
    runs: List[Dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line.startswith("{'cont_conf_hgnn'"):
            continue
        try:
            parsed = ast.literal_eval(clean_literal(line))
        except (SyntaxError, ValueError):
            continue
        if "cont_conf_hgnn" in parsed:
            runs.append(parsed["cont_conf_hgnn"])

    if not runs:
        return {}

    buckets: Dict[str, List[float]] = {
        "aps_cov": [],
        "aps_eff": [],
        "raps_cov": [],
        "raps_eff": [],
        "eff_valid": [],
        "eff_valid_raps": [],
    }
    for run in runs:
        aps = run.get("APS")
        if isinstance(aps, (tuple, list)) and len(aps) == 2:
            buckets["aps_cov"].append(float(aps[0]))
            buckets["aps_eff"].append(float(aps[1]))
        raps = run.get("RAPS")
        if isinstance(raps, (tuple, list)) and len(raps) == 2:
            buckets["raps_cov"].append(float(raps[0]))
            buckets["raps_eff"].append(float(raps[1]))
        if "eff_valid" in run:
            buckets["eff_valid"].append(float(run["eff_valid"]))
        if "eff_valid_raps" in run:
            buckets["eff_valid_raps"].append(float(run["eff_valid_raps"]))

    metrics = {name: statistics.fmean(values) for name, values in buckets.items() if values}
    metrics["completed_runs"] = float(len(runs))
    return metrics


def objective(script_key: str, args: argparse.Namespace) -> None:
    wandb = _wandb()
    with wandb.init() as run:
        config = dict(run.config)
        cmd = build_command(script_key, config, args)
        start = time.monotonic()
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={**os.environ, "WANDB_MODE": args.wandb_mode},
        )
        duration = time.monotonic() - start
        metrics = parse_trial_metrics(proc.stdout)
        metrics.update({"returncode": float(proc.returncode), "duration_sec": duration})
        wandb.log(metrics)
        wandb.save(str(ROOT / SCRIPT_SETTINGS[script_key]["program"]), policy="now")
        if proc.returncode != 0:
            print(proc.stdout)
            raise RuntimeError(f"Training command failed with exit code {proc.returncode}: {' '.join(cmd)}")
        if not metrics.get("completed_runs"):
            print(proc.stdout)
            raise RuntimeError("Could not parse cont_conf_hgnn metrics from training output.")


def main(script_key: str) -> None:
    args = parse_args(script_key)
    wandb = _wandb()
    os.environ["WANDB_MODE"] = args.wandb_mode

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_id = wandb.sweep(sweep_config(script_key), project=args.project, entity=args.entity)

    deadline = time.monotonic() + args.hours * 3600
    launched = 0
    while time.monotonic() < deadline:
        if args.count is not None and launched >= args.count:
            break
        wandb.agent(
            sweep_id,
            function=partial(objective, script_key, args),
            count=1,
            project=args.project,
            entity=args.entity,
        )
        launched += 1
