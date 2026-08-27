"""Aggregate per-object MAE/MSE mean±std from an eval summary and export a paper-style table figure."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


NAME_MAP = {
    "grasp_logs_beike": "beike",
    "grasp_logs_mosha": "mosha",
    "grasp_logs_mosilian": "mosilian",
    "grasp_logs_paomo": "paomo",
    "grasp_logs_qiu": "qiu",
    "grasp_logs_ruanzhu": "ruanzhu",
    "grasp_logs_suliaoping": "suliaoping",
    "grasp_logs_taocibei": "taocibei",
    "grasp_logs_touming": "touming",
    "grasp_logs_yingzhu": "yingzhu",
}


def fmt(m: float, s: float) -> str:
    return f"{m:.4f}±{s:.4f}"


def parse_summary(path: Path) -> list[tuple[str, str, float, float]]:
    text = path.read_text(encoding="utf-8")
    rows = []
    for line in text.splitlines():
        m = re.match(
            r"(grasp_logs_\w+)/([\w.]+): MAE=([0-9.]+) MSE=([0-9.]+)",
            line,
        )
        if m:
            rows.append((m.group(1), m.group(2), float(m.group(3)), float(m.group(4))))
    return rows


def summarize(rows: list[tuple[str, str, float, float]]) -> tuple[list[dict], dict]:
    objs = sorted(set(r[0] for r in rows))
    table = []
    for obj in objs:
        maes = [r[2] for r in rows if r[0] == obj]
        mses = [r[3] for r in rows if r[0] == obj]
        table.append(
            {
                "object": NAME_MAP.get(obj, obj),
                "folder": obj,
                "mse_mean": float(np.mean(mses)),
                "mse_std": float(np.std(mses, ddof=0)),
                "mae_mean": float(np.mean(maes)),
                "mae_std": float(np.std(maes, ddof=0)),
                "n": len(maes),
            }
        )
    all_mae = [r[2] for r in rows]
    all_mse = [r[3] for r in rows]
    all_row = {
        "object": "All",
        "mse_mean": float(np.mean(all_mse)),
        "mse_std": float(np.std(all_mse, ddof=0)),
        "mae_mean": float(np.mean(all_mae)),
        "mae_std": float(np.std(all_mae, ddof=0)),
        "n": len(all_mae),
    }
    return table, all_row


def save_png(table: list[dict], all_row: dict, out: Path, title: str) -> None:
    labels = [t["object"] for t in table] + ["All"]
    mse_cells = [fmt(t["mse_mean"], t["mse_std"]) for t in table] + [
        fmt(all_row["mse_mean"], all_row["mse_std"])
    ]
    mae_cells = [fmt(t["mae_mean"], t["mae_std"]) for t in table] + [
        fmt(all_row["mae_mean"], all_row["mae_std"])
    ]
    best_i = int(np.argmin([t["mae_mean"] for t in table]))

    fig_h = 0.45 * (len(labels) + 2) + 0.5
    fig, ax = plt.subplots(figsize=(6.8, fig_h))
    ax.axis("off")
    cell_text = [[labels[i], mse_cells[i], mae_cells[i]] for i in range(len(labels))]
    table_m = ax.table(
        cellText=cell_text,
        colLabels=["Object type", "MSE", "MAE"],
        loc="center",
        cellLoc="center",
        colWidths=[0.28, 0.36, 0.36],
    )
    table_m.auto_set_font_size(False)
    table_m.set_fontsize(11)
    table_m.scale(1.15, 1.6)

    for (r, c), cell in table_m.get_celld().items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "BT"
            cell.set_linewidth(1.3)
        elif r == len(labels):
            if c == 0:
                cell.set_text_props(weight="bold")
            cell.visible_edges = "B"
            cell.set_linewidth(1.3)
        elif r - 1 == best_i:
            cell.set_text_props(weight="bold")
            cell.visible_edges = ""
        else:
            cell.visible_edges = ""
        if c == 0 and r > 0:
            cell._loc = "left"
            cell.set_text_props(ha="left")

    ax.set_title(title, fontsize=12, pad=10)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parents[1]
            / "checkpoints"
            / "test_eval_rollout"
        ),
    )
    parser.add_argument("--title", type=str, default="Object type metrics (trial split, exit+1s)")
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    rows = parse_summary(eval_dir / "summary.txt")
    if not rows:
        raise SystemExit(f"no per-trial rows in {eval_dir / 'summary.txt'}")

    table, all_row = summarize(rows)
    print("Object type          MSE                 MAE")
    print("-" * 58)
    best = min(t["mae_mean"] for t in table)
    for t in table:
        star = "  *" if t["mae_mean"] == best else ""
        print(
            f"{t['object']:12s}  {fmt(t['mse_mean'], t['mse_std']):20s}  "
            f"{fmt(t['mae_mean'], t['mae_std']):20s}{star}"
        )
    print(
        f"{'All':12s}  {fmt(all_row['mse_mean'], all_row['mse_std']):20s}  "
        f"{fmt(all_row['mae_mean'], all_row['mae_std']):20s}"
    )

    payload = {"per_object": table, "all": all_row, "note": "mean±std over test trials per object; All over all test trials"}
    (eval_dir / "metrics_table.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    out_png = eval_dir / "metrics_table_object.png"
    save_png(table, all_row, out_png, args.title)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
