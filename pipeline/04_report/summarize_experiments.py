#!/usr/bin/env python3
"""Summarise a folder of GenNet experiments into a single CSV report.

Reads the hyperparameters (``train_args.json``) and AUCs
(``pd_summary_results.csv``) of every ``GenNet_experiment_<ID>_/`` under the
given folder. Output goes to ``reports/``, which is committed, unlike the
multi-GB ``results/``.

Two modes:

* ``gridsearch`` (default) — sorted by validation AUC, so the winner is row 1.
* ``multiseed`` — one config across ``seed_N`` splits, sorted by seed, with
  mean/std/min/max rows appended over the AUC columns.

Activation comes from ``hidden_activation`` in the JSON rather than the folder
name, so the same command works for tanh, relu and softplus.

Usage:
    python pipeline/04_report/summarize_experiments.py results/relu
    python pipeline/04_report/summarize_experiments.py results/tanh --mode multiseed
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
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
    if not datapath:
        return ""
    m = re.search(r"seed_(\d+)", datapath)
    return m.group(1) if m else ""


def read_aucs(exp_dir: Path) -> tuple[str, str]:
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


MODES = {
    "gridsearch": {"out_dir": "reports/gridsearch", "suffix": "gridsearch",
                   "sort": "auc", "aggregate": False},
    "multiseed": {"out_dir": "reports/multiseed", "suffix": "multiseed",
                  "sort": "seed", "aggregate": True},
}

AGG_FIELDS = ["auc_val", "auc_test", "best_val_loss"]


def _num(row: dict, field: str, default: float):
    try:
        return float(row[field])
    except (ValueError, TypeError, KeyError):
        return default


def sort_rows(rows: list[dict], how: str) -> None:
    """Sort in place; entries without the sort field go last."""
    if how == "auc":
        rows.sort(key=lambda r: (0, -_num(r, "auc_val", 0.0))
                  if r.get("auc_val") else (1, 0.0))
    elif how == "seed":
        rows.sort(key=lambda r: (0, int(r["seed"]))
                  if str(r.get("seed", "")).isdigit() else (1, 0))
    elif how == "id":
        rows.sort(key=lambda r: (0, int(r["ID"]))
                  if str(r.get("ID", "")).isdigit() else (1, 0))


def parse_ids(spec: str) -> set:
    """Parse '142-146,150' into a set of string IDs."""
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(str(i) for i in range(int(lo), int(hi) + 1))
        else:
            ids.add(part)
    return ids


def config_key(row: dict) -> tuple:
    """The hyperparameters that define 'the same config' across seeds."""
    return (row["activation"], row["learning_rate"], row["L1"], row["L1_act"])


def dominant_config_rows(rows: list[dict]) -> list[dict]:
    """Rows for the config shared across the most distinct seeds.

    A stability folder also holds the grid points it was selected from, which
    must not pollute the aggregates.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault(config_key(r), []).append(r)
    return max(groups.values(),
               key=lambda g: (len({r["seed"] for r in g}), len(g)))


def aggregate_rows(rows: list[dict]) -> list[dict]:
    stats = {label: {} for label in ("mean", "std", "min", "max")}
    for field in AGG_FIELDS:
        vals = [v for r in rows if (v := _num(r, field, None)) is not None]
        if not vals:
            continue
        stats["mean"][field] = f"{statistics.mean(vals):.6f}"
        stats["std"][field] = (f"{statistics.stdev(vals):.6f}"
                               if len(vals) > 1 else "0.000000")
        stats["min"][field] = f"{min(vals):.6f}"
        stats["max"][field] = f"{max(vals):.6f}"

    out = []
    for label in ("mean", "std", "min", "max"):
        if not stats[label]:
            continue
        row = {f: "" for f in FIELDS}
        row["ID"] = label
        row["seed"] = f"n={len(rows)}"
        row.update(stats[label])
        out.append(row)
    return out


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
        "--mode",
        choices=sorted(MODES),
        default="gridsearch",
        help="Report mode (default: gridsearch). 'multiseed' sorts by seed and "
             "appends mean/std/min/max rows.",
    )
    ap.add_argument(
        "--sort",
        choices=["auc", "seed", "id"],
        default=None,
        help="Row order (default: per --mode; gridsearch=auc, multiseed=seed)",
    )
    ap.add_argument(
        "--aggregate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Append mean/std/min/max rows (default: on for --mode multiseed)",
    )
    ap.add_argument(
        "--ids",
        default=None,
        help="Restrict to these experiment IDs, e.g. '142-146' or '105,143,144'. "
             "Overrides the automatic same-config filtering in multiseed mode.",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Folder for the CSV report (default: per --mode under reports/)",
    )
    ap.add_argument(
        "--name",
        default=None,
        help="Output CSV basename (default: input folder name, e.g. 'tanh')",
    )
    args = ap.parse_args()

    cfg = MODES[args.mode]
    sort_how = args.sort or cfg["sort"]
    aggregate = cfg["aggregate"] if args.aggregate is None else args.aggregate
    out_dir = args.out_dir or cfg["out_dir"]

    results_path = Path(args.results_path)
    if not results_path.is_dir():
        raise SystemExit(f"Not a directory: {results_path}")

    # rglob: experiment dirs may sit one level down.
    exp_dirs = sorted(results_path.rglob("GenNet_experiment_*_"),
                      key=lambda p: p.name)
    rows = [r for d in exp_dirs if d.is_dir() for r in (collect(d),) if r]

    if not rows:
        raise SystemExit(f"No readable experiments found under {results_path}/")

    # --ids wins; otherwise multiseed keeps only the config shared across seeds.
    if args.ids:
        want = parse_ids(args.ids)
        rows = [r for r in rows if str(r["ID"]) in want]
        if not rows:
            raise SystemExit(f"No experiments matched --ids {args.ids}")
    elif aggregate:
        rows = dominant_config_rows(rows)

    sort_rows(rows, sort_how)
    out_rows = rows + aggregate_rows(rows) if aggregate else rows

    name = args.name or results_path.name
    out_path = Path(out_dir) / f"{name}_{cfg['suffix']}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(rows)} experiments -> {out_path}")
    if aggregate:
        agg = {r["ID"]: r for r in out_rows if r["ID"] in ("mean", "std")}
        m, s = agg.get("mean", {}), agg.get("std", {})
        if m:
            print(f"test AUC = {m.get('auc_test','?')} ± {s.get('auc_test','?')}  |  "
                  f"val AUC = {m.get('auc_val','?')} ± {s.get('auc_val','?')}")
    else:
        best = rows[0]
        if best["auc_val"]:
            print(f"Best: exp {best['ID']} ({best['activation']}, "
                  f"lr={best['learning_rate']}, L1={best['L1']}) "
                  f"val AUC={best['auc_val']}, test AUC={best['auc_test']}")


if __name__ == "__main__":
    main()
