#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# -------------------------------------------------------
# フォント設定（提示スクリプトと同一）
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
# 集計
# -------------------------------------------------------
def load_events_csv(path: Path, event_name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{event_name} CSV not found: {path}")

    df = pd.read_csv(path)

    required = ["method", "lead_sec", "rep"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"{event_name} CSV schema mismatch: {path}\n"
            f"missing={missing}\n"
            f"columns:\n  {', '.join(df.columns.astype(str))}"
        )

    df = df.copy()
    df["method"] = df["method"].astype(str).str.lower()
    df["lead_sec"] = pd.to_numeric(df["lead_sec"], errors="coerce")
    df["rep"] = pd.to_numeric(df["rep"], errors="coerce")
    df = df.dropna(subset=["method", "lead_sec", "rep"]).copy()
    df["lead_sec"] = df["lead_sec"].astype(int)
    df["rep"] = df["rep"].astype(int)

    return df


def summarize_mean_events_by_method_lead(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: events_* の生イベント（行数=イベント数）
    1) rep内のイベント数を数える: count(method,lead,rep)
    2) rep平均を計算: mean_count(method,lead)
    """
    # rep内イベント数
    per_rep = (
        df.groupby(["method", "lead_sec", "rep"], as_index=False)
          .size()
          .rename(columns={"size": "n_events"})
    )

    # leadごとrep平均
    mean_by_lead = (
        per_rep.groupby(["method", "lead_sec"], as_index=False)["n_events"]
               .mean()
               .rename(columns={"n_events": "mean_events"})
    )

    return mean_by_lead


# -------------------------------------------------------
# 描画（棒グラフ：lead軸に並べてmethodを比較）
# -------------------------------------------------------
def plot_mean_events_bar(
    mean_df: pd.DataFrame,
    out_pdf: Path,
    y_label: str,
    methods_order: List[str],
    lead_order: Optional[List[int]] = None,
):
    if mean_df.empty:
        raise SystemExit("No data to plot (mean_df is empty).")

    df = mean_df.copy()

    # method順序
    df["method"] = df["method"].astype(str).str.lower()
    df["method"] = pd.Categorical(df["method"], categories=methods_order, ordered=True)

    # lead順序
    if lead_order is None:
        lead_order = sorted(df["lead_sec"].unique().tolist())
    df["lead_sec"] = pd.Categorical(df["lead_sec"], categories=lead_order, ordered=True)

    df = df.sort_values(["lead_sec", "method"]).copy()

    # pivot: rows=lead, cols=method, values=mean_events
    pv = df.pivot_table(index="lead_sec", columns="method", values="mean_events", aggfunc="mean").fillna(0.0)

    leads = [int(x) for x in pv.index.tolist()]
    x = np.arange(len(leads), dtype=float)

    # 棒の幅
    n_methods = len(methods_order)
    width = 0.38 if n_methods == 2 else 0.8 / max(1, n_methods)

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=300)

    for i, m in enumerate(methods_order):
        if m not in pv.columns:
            continue
        vals = pv[m].to_numpy(dtype=float)
        # 2系列想定で左右にずらす
        offset = (i - (n_methods - 1) / 2.0) * width
        label = "Autopilot" if m == "autopilot" else ("LSTM" if m == "lstm" else m)
        ax.bar(x + offset, vals, width=width, label=label)

    ax.set_xlabel("予測開始時刻 [s]")
    ax.set_ylabel(y_label)

    # x軸ラベルは lead_sec を負で出す運用（事故フレーム=0に合わせる）
    # lead=1..10 を -1..-10 表示
    ax.set_xticks(x)
    ax.set_xticklabels([f"{-int(l)}" for l in leads])

    ax.grid(True, axis="y", linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", direction="in")
    ax.set_axisbelow(True)

    # タイトル無し（要求）
    ax.legend(framealpha=1.0, facecolor="white")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_pdf}")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", type=str, required=True,
                    help="events_nearmiss.csv / events_collision.csv があるディレクトリ")
    ap.add_argument("--outdir", type=str, default="", help="出力先（未指定ならevents-dir配下out_mean_events_YYYYmmdd_HHMMSS）")

    ap.add_argument("--nearmiss-csv", type=str, default="events_nearmiss.csv")
    ap.add_argument("--collision-csv", type=str, default="events_collision.csv")

    ap.add_argument("--methods-order", type=str, nargs="*", default=["autopilot", "lstm"])
    ap.add_argument("--lead-min", type=int, default=-1, help="leadの下限（-1で無効）")
    ap.add_argument("--lead-max", type=int, default=-1, help="leadの上限（-1で無効）")
    args = ap.parse_args()

    events_dir = Path(args.events_dir).expanduser().resolve()
    if not events_dir.exists():
        raise SystemExit(f"events-dir not found: {events_dir}")

    nearmiss_path = events_dir / args.nearmiss_csv
    collision_path = events_dir / args.collision_csv

    # 出力dir
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = events_dir / f"out_mean_events_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)

    methods_order = [m.lower() for m in args.methods_order]

    # -------- Near-miss --------
    nm = load_events_csv(nearmiss_path, "near_miss")
    if args.lead_min >= 0:
        nm = nm[nm["lead_sec"] >= args.lead_min]
    if args.lead_max >= 0:
        nm = nm[nm["lead_sec"] <= args.lead_max]
    nm_mean = summarize_mean_events_by_method_lead(nm)

    # lead順序の確定（両方の図で揃えるなら collision からも union して揃えても良いが，今回は各図ごと）
    nm_leads = sorted(nm_mean["lead_sec"].unique().tolist())
    plot_mean_events_bar(
        mean_df=nm_mean,
        out_pdf=outdir / "fig_nearmiss_mean_events_by_lead.pdf",
        y_label="ニアミス回数（平均）",
        methods_order=methods_order,
        lead_order=nm_leads,
    )

    # -------- Collision --------
    co = load_events_csv(collision_path, "collision")
    if args.lead_min >= 0:
        co = co[co["lead_sec"] >= args.lead_min]
    if args.lead_max >= 0:
        co = co[co["lead_sec"] <= args.lead_max]
    co_mean = summarize_mean_events_by_method_lead(co)
    co_leads = sorted(co_mean["lead_sec"].unique().tolist())

    plot_mean_events_bar(
        mean_df=co_mean,
        out_pdf=outdir / "fig_collision_mean_events_by_lead.pdf",
        y_label="衝突回数（平均）",
        methods_order=methods_order,
        lead_order=co_leads,
    )

    # 集計表も一応出す（後で表にしたい場合用）
    nm_mean.to_csv(outdir / "mean_nearmiss_events_by_method_lead.csv", index=False)
    co_mean.to_csv(outdir / "mean_collision_events_by_method_lead.csv", index=False)

    print(f"[OK] 出力先: {outdir}")
    return 0


if __name__ == "__main__":
    from datetime import datetime
    raise SystemExit(main())
