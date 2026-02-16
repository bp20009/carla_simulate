#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# -------------------------------------------------------
# フォント設定（macOS向け・IPAexフォールバック付き）
# -------------------------------------------------------
cand_fonts = ["BIZ UDPGothic", "Hiragino Sans", "YuGothic", "Osaka", "BIZ UDGothic", "IPAexGothic"]
use_font = None
for f in cand_fonts:
    if f in {font.name for font in fm.fontManager.ttflist}:
        use_font = f
        break
if use_font:
    plt.rcParams["font.family"] = use_font
else:
    print("[WARN] 適切な日本語フォントが見つからず fallback します")
    plt.rcParams["font.family"] = "sans-serif"

plt.rcParams["font.size"] = 14
plt.rcParams["axes.labelsize"] = 16
plt.rcParams["xtick.labelsize"] = 14
plt.rcParams["ytick.labelsize"] = 14
plt.rcParams["legend.fontsize"] = 14

# -------------------------------------------------------
# load
# -------------------------------------------------------
def load_events(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["x", "y", "method", "lead_sec", "rep"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing columns {missing} in {csv_path}")

    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["lead_sec"] = pd.to_numeric(df["lead_sec"], errors="coerce")
    df["rep"] = pd.to_numeric(df["rep"], errors="coerce")
    df = df.dropna(subset=["x", "y", "method", "lead_sec", "rep"]).copy()

    df["method"] = df["method"].astype(str).str.lower()
    df["lead_sec"] = df["lead_sec"].astype(int)
    df["rep"] = df["rep"].astype(int)

    if "event_type" in df.columns:
        df["event_type"] = df["event_type"].astype(str).str.lower()

    return df

# -------------------------------------------------------
# plot helpers
# -------------------------------------------------------
def compute_limits(df: pd.DataFrame, pad_ratio: float = 0.03) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    xmin, xmax = float(df["x"].min()), float(df["x"].max())
    ymin, ymax = float(df["y"].min()), float(df["y"].max())
    if xmax - xmin < 1e-9:
        xmax += 1.0
        xmin -= 1.0
    if ymax - ymin < 1e-9:
        ymax += 1.0
        ymin -= 1.0
    xr = xmax - xmin
    yr = ymax - ymin
    return (xmin - xr * pad_ratio, xmax + xr * pad_ratio), (ymin - yr * pad_ratio, ymax + yr * pad_ratio)

def setup_axes(ax):
    ax.set_xlabel("x (world)")
    ax.set_ylabel("y (world)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", direction="in")

def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(df) > max_points:
        return df.sample(max_points, random_state=0)
    return df

# -------------------------------------------------------
# legend-only pdf
# -------------------------------------------------------
def save_legend_pdf(out_pdf: Path, labels: List[str]) -> None:
    """
    Create a PDF that contains only a legend.
    Colors are default matplotlib cycle (no explicit color setting).
    """
    if not labels:
        return

    fig = plt.figure(figsize=(6, 1.0), dpi=300)
    ax = fig.add_subplot(111)
    ax.axis("off")

    handles = []
    for lab in labels:
        h = ax.scatter([], [], s=30, label=lab)  # dummy handle
        handles.append(h)

    leg = ax.legend(
        handles=handles,
        labels=labels,
        loc="center",
        ncol=min(len(labels), 4),
        framealpha=1.0,
        facecolor="white",
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote legend: {out_pdf}")

# -------------------------------------------------------
# plots
# -------------------------------------------------------
def plot_single_scatter(df: pd.DataFrame, out_pdf: Path, s: float, alpha: float, max_points: int) -> None:
    if df.empty:
        return
    df = downsample(df, max_points)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    ax.scatter(df["x"], df["y"], s=s, alpha=alpha)
    setup_axes(ax)

    (xmin, xmax), (ymin, ymax) = compute_limits(df)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

def plot_merged_methods(df: pd.DataFrame, out_pdf: Path, s: float, alpha: float, max_points: int) -> None:
    if df.empty:
        return
    df = downsample(df, max_points)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    methods = sorted(df["method"].unique().tolist())
    for m in methods:
        sub = df[df["method"] == m]
        ax.scatter(sub["x"], sub["y"], s=s, alpha=alpha)  # legendは外に出すので labelしない

    setup_axes(ax)
    (xmin, xmax), (ymin, ymax) = compute_limits(df)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

def plot_all_eventtype_colored(df: pd.DataFrame, out_pdf: Path, s: float, alpha: float, max_points: int) -> None:
    if df.empty:
        return
    if "event_type" not in df.columns:
        raise SystemExit("events_all.csv must have event_type column for color split.")
    df = downsample(df, max_points)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)

    # order fixed
    order = ["near_miss", "collision"]
    keys = [k for k in order if k in set(df["event_type"].unique())]
    # include any extras (should not, but safe)
    extras = sorted([k for k in df["event_type"].unique() if k not in set(keys)])
    keys += extras

    for k in keys:
        sub = df[df["event_type"] == k]
        ax.scatter(sub["x"], sub["y"], s=s, alpha=alpha)  # legendは外に出すので labelしない

    setup_axes(ax)
    (xmin, xmax), (ymin, ymax) = compute_limits(df)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

# -------------------------------------------------------
# main
# -------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", type=str, required=True, help="events_*.csv があるディレクトリ")
    ap.add_argument("--outdir", type=str, default="", help="未指定なら events-dir/out_scatter_pdf_by_run_v3")

    ap.add_argument("--max-points", type=int, default=200000)
    ap.add_argument("--s", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.15)

    ap.add_argument("--only-lead", type=int, default=-1, help="lead_sec filter. -1=all")
    ap.add_argument("--only-rep", type=int, default=-1, help="rep filter. -1=all")

    ap.add_argument("--make-merged-methods", action="store_true",
                    help="autopilotとlstmを同一図に重ねたPDFも出力する（lead,repごと）")
    ap.add_argument("--merged-target", type=str, default="all",
                    choices=["nearmiss", "collision", "all"],
                    help="mergedに使う元CSV（通常はall）")
    ap.add_argument("--merged-color", type=str, default="method",
                    choices=["method", "event_type"],
                    help="merged図の色分け（method または event_type）")

    ap.add_argument("--legend-separate", action="store_true",
                    help="凡例を別PDFに出す（図本体には凡例を入れない）")
    args = ap.parse_args()

    events_dir = Path(args.events_dir).expanduser().resolve()
    if not events_dir.exists():
        raise SystemExit(f"events-dir not found: {events_dir}")

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (events_dir / "out_scatter_pdf_by_run_v3")
    outdir.mkdir(parents=True, exist_ok=True)

    tag_to_file: Dict[str, Path] = {
        "nearmiss": events_dir / "events_nearmiss.csv",
        "collision": events_dir / "events_collision.csv",
        "all": events_dir / "events_all.csv",
    }

    # 1) per-run outputs (tagごと)
    for tag, csv_path in tag_to_file.items():
        if not csv_path.exists():
            print(f"[WARN] skip (not found): {csv_path}")
            continue

        df = load_events(csv_path)

        if args.only_lead >= 0:
            df = df[df["lead_sec"] == int(args.only_lead)]
        if args.only_rep >= 0:
            df = df[df["rep"] == int(args.only_rep)]

        if df.empty:
            print(f"[WARN] no rows after filters: {csv_path}")
            continue

        for (method, lead, rep), sub in df.groupby(["method", "lead_sec", "rep"], sort=True):
            out_pdf = outdir / tag / method / f"lead_{lead}" / f"scatter_rep_{rep}.pdf"
            if tag == "all":
                plot_all_eventtype_colored(sub, out_pdf, s=args.s, alpha=args.alpha, max_points=args.max_points)
            else:
                plot_single_scatter(sub, out_pdf, s=args.s, alpha=args.alpha, max_points=args.max_points)

        print(f"[OK] wrote per-run tag={tag} -> {outdir / tag}")

        # legend-only for tag=all (event_type)
        if args.legend_separate and tag == "all":
            legend_dir = outdir / tag
            save_legend_pdf(legend_dir / "legend_event_type.pdf", ["near_miss", "collision"])

    # 2) merged methods outputs
    if args.make_merged_methods:
        base_tag = args.merged_target
        base_csv = tag_to_file[base_tag]
        if not base_csv.exists():
            raise SystemExit(f"merged-target CSV not found: {base_csv}")

        dfm = load_events(base_csv)

        if args.only_lead >= 0:
            dfm = dfm[dfm["lead_sec"] == int(args.only_lead)]
        if args.only_rep >= 0:
            dfm = dfm[dfm["rep"] == int(args.only_rep)]

        if dfm.empty:
            raise SystemExit("No rows for merged after filters.")

        for (lead, rep), sub in dfm.groupby(["lead_sec", "rep"], sort=True):
            out_pdf = outdir / "merged_methods" / base_tag / f"lead_{lead}" / f"scatter_rep_{rep}.pdf"

            if args.merged_color == "event_type":
                plot_all_eventtype_colored(sub, out_pdf, s=args.s, alpha=args.alpha, max_points=args.max_points)
            else:
                plot_merged_methods(sub, out_pdf, s=args.s, alpha=args.alpha, max_points=args.max_points)

        print(f"[OK] wrote merged_methods -> {outdir / 'merged_methods' / base_tag}")

        # legend-only for merged
        if args.legend_separate:
            legend_dir = outdir / "merged_methods" / base_tag
            if args.merged_color == "event_type":
                save_legend_pdf(legend_dir / "legend_event_type.pdf", ["near_miss", "collision"])
            else:
                save_legend_pdf(legend_dir / "legend_method.pdf", ["autopilot", "lstm"])

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
