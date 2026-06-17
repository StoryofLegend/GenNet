#!/usr/bin/env python3
"""Summarise one GenNet grid-search folder into a single CSV report.

Point this at a folder containing ``GenNet_experiment_<ID>_/`` experiment
directories (e.g. ``results/tanh``). It reads the hyperparameters
(``train_args.json``) and the AUCs (``pd_summary_results.csv``) of every
experiment found underneath, and writes one CSV — sorted by validation AUC — into
a report folder meant to be committed to git (``results/`` itself is gitignored
because of the multi-GB weight files, so the small CSV in ``reports/`` is the
publishable artefact).

The output CSV is named after the input folder by default
(``results/tanh`` -> ``reports/gridsearch/tanh_gridsearch.csv``); override with
``--name``.

Usage:
    python pipeline/04_report/summarize_gridsearch.py results/tanh
    python pipeline/04_report/summarize_gridsearch.py results/relu --out-dir reports/gridsearch
    python pipeline/04_report/summarize_gridsearch.py results/tanh --name tanh_seed42
"""

import argparse
import csv
import json
import re
from pathlib import Path

# Columns written to the CSV, in order.
FIELDS = [
    "ID",
    "activation",
    "seed",
    "learning_rate",
    "L1",
    "L1_act",
    "batch_size",
    "epochs",
    "patience",
    "epochs_trained",
    "best_val_loss",
    "auc_val",
    "auc_test",
    "problem_type",
    "slurm_job_id",
    "experiment_dir",
]


def default_activation(problem_type: str) -> str:
    """GenNet's default hidden activation when none is given (GenNet.py:261)."""
    return "relu" if problem_type == "regression" else "tanh"


def parse_seed(datapath: str) -> str:
    """Extract the seed number from a path like 'processed_data/seed_42/'."""
    if not datapath:
        return ""
    m = re.search(r"seed_(\d+)", datapath)
    return m.group(1) if m else ""


def read_aucs(exp_dir: Path) -> tuple[str, str]:
    """Read validation/test AUC from pd_summary_results.csv (key,value rows)."""
    summary = exp_dir / "pd_summary_results.csv"
    auc_val, auc_test = "", ""
    if not summary.exists():
        return auc_val, auc_test
    with summary.open() as fh:
        for row in csv.reader(fh):
            if not row:
                continue
            key = row[0].strip()
            val = row[1].strip() if len(row) > 1 else ""
            if key == "AUC validation":
                auc_val = val
            elif key == "AUC test":
                auc_test = val
    return auc_val, auc_test


def read_train_log(exp_dir: Path) -> tuple[str, str]:
    """Return (epochs_trained, best_val_loss) from train_log.csv if present."""
    log = exp_dir / "train_log.csv"
    if not log.exists():
        return "", ""
    with log.open() as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or "val_loss" not in reader.fieldnames:
            return "", ""
        best = None
        n = 0
        for row in reader:
            n += 1
            try:
                vl = float(row["val_loss"])
            except (ValueError, KeyError, TypeError):
                continue
            if best is None or vl < best:
                best = vl
    best_val_loss = f"{best:.6f}" if best is not None else ""
    return str(n), best_val_loss


def collect(exp_dir: Path) -> dict | None:
    """Build one summary row from an experiment folder, or None if unreadable."""
    args_path = exp_dir / "train_args.json"
    if not args_path.exists():
        return None
    try:
        args = json.loads(args_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    problem_type = args.get("problem_type", "")
    activation = args.get("hidden_activation") or default_activation(problem_type)
    auc_val, auc_test = read_aucs(exp_dir)
    epochs_trained, best_val_loss = read_train_log(exp_dir)

    return {
        "ID": args.get("ID", ""),
        "activation": activation,
        "seed": parse_seed(args.get("datapath", "") or args.get("path", "")),
        "learning_rate": args.get("learning_rate", ""),
        "L1": args.get("L1", ""),
        "L1_act": args.get("L1_act", ""),
        "batch_size": args.get("batch_size", ""),
        "epochs": args.get("epochs", ""),
        "patience": args.get("patience", ""),
        "epochs_trained": epochs_trained,
        "best_val_loss": best_val_loss,
        "auc_val": auc_val,
        "auc_test": auc_test,
        "problem_type": problem_type,
        "slurm_job_id": args.get("SlURM_JOB_ID", ""),
        "experiment_dir": str(exp_dir),
    }


def sort_key(row: dict):
    """Sort by validation AUC descending; runs without an AUC go last."""
    try:
        return (0, -float(row["auc_val"]))
    except (ValueError, TypeError):
        return (1, 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "results_path",
        help="Folder of GenNet_experiment_<ID>_ dirs to summarise, e.g. results/tanh",
    )
    ap.add_argument(
        "--out-dir",
        default="reports/gridsearch",
        help="Folder for the CSV report (default: reports/gridsearch)",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="Output CSV basename (default: input folder name, e.g. 'tanh')",
    )
    args = ap.parse_args()

    results_path = Path(args.results_path)
    if not results_path.is_dir():
        raise SystemExit(f"Not a directory: {results_path}")

    # rglob so it works whether the experiment dirs sit directly in the given
    # folder or one level down.
    exp_dirs = sorted(results_path.rglob("GenNet_experiment_*_"),
                      key=lambda p: p.name)
    rows = [r for d in exp_dirs if d.is_dir() for r in (collect(d),) if r]

    if not rows:
        raise SystemExit(f"No readable experiments found under {results_path}/")

    rows.sort(key=sort_key)

    name = args.name or results_path.name
    out_path = Path(args.out_dir) / f"{name}_gridsearch.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} experiments -> {out_path}")
    best = rows[0]
    if best["auc_val"]:
        print(f"Best: exp {best['ID']} ({best['activation']}, "
              f"lr={best['learning_rate']}, L1={best['L1']}) "
              f"val AUC={best['auc_val']}, test AUC={best['auc_test']}")


if __name__ == "__main__":
    main()
