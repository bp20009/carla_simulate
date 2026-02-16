#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Optional

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
# 探索: results_grid_accident/{method}/lead_*/rep_*/logs/collisions.csv
# -------------------------------------------------------
LEAD_RE = re.compile(r"lead_(\d+)$")
REP_RE = re.compile(r"rep_(\d+)$")


@dataclass
class RunRef:
    method: str
    lead: int
    rep: int
    collisions_csv: Path


def discover_runs(root: Path, methods: Optional[List[str]] = None) -> List[RunRef]:
    runs: List[RunRef] = []
    if methods is None:
        methods = ["autopilot", "lstm"]

    for method in methods:
        mdir = root / method
        if not mdir.exists():
            continue

        for lead_dir in sorted(mdir.glob("lead_*")):
            m = LEAD_RE.search(lead_dir.name)
            if not m:
                continue
            lead = int(m.group(1))

            for rep_dir in sorted(lead_dir.glob("rep_*")):
                r = REP_RE.search(rep_dir.name)
                if not r:
                    continue
                rep = int(r.group(1))

                cpath = rep_dir / "logs" / "collisions.csv"
                if cpath.exists():
                    runs.append(RunRef(method=method, lead=lead, rep=rep, collisions_csv=cpath))
    return runs


# -------------------------------------------------------
# 重複排除
# -------------------------------------------------------
def canonical_event_key(df: pd.DataFrame) -> pd.Series:
    """
    同一衝突が (A,B) と (B,A) の2行で出るケースを落とす．
    key = (payload_frame, min(id), max(id), round(intensity, 3))
    """
    a = pd.to_numeric(df["actor_id"], errors="coerce").astype("Int64")
    b = pd.to_numeric(df["other_id"], errors="coerce").astype("Int64")
    lo = a.where(a <= b, b)
    hi = b.where(a <= b, a)

    inten_round = pd.to_numeric(df["intensity"], errors="coerce").fillna(-1).round(3)

    return (
        df["payload_frame"].astype("Int64").astype(str)
        + "|"
        + lo.astype("Int64").astype(str)
        + "|"
        + hi.astype("Int64").astype(str)
        + "|"
        + inten_round.astype(str)
    )


def norm(s: object) -> str:
    return "" if s is None else str(s).strip().lower()


# -------------------------------------------------------
# other_type の集約ルール
# -------------------------------------------------------
def other_bucket(other_type: object) -> str:
    """
    - vehicle.* -> vehicle
    - static.car + static.bicycle -> static.mobility
    - static.* (それ以外) -> static.other
    - walker.* -> walker
    - else/欠損 -> unknown
    """
    ot = norm(other_type)
    if not ot or ot in ["none", "nan", "null"]:
        return "unknown"

    if ot.startswith("vehicle."):
        return "vehicle"

    if ot in ["static.car", "static.bicycle"]:
        return "static.mobility"

    if ot.startswith("static."):
        return "static.other"

    if ot.startswith("walker."):
        return "walker"

    return "unknown"


# -------------------------------------------------------
# 描画ユーティリティ
# -------------------------------------------------------
def common_axes_style(ax):
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", direction="in")
    ax.set_axisbelow(True)


def plot_total_overall(method_total: pd.DataFrame, out_pdf: Path):
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    ax.bar(method_total["method"].astype(str), method_total["total"].astype(float))
    ax.set_xlabel("手法")
    ax.set_ylabel("総衝突回数（rep平均）")
    common_axes_style(ax)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存 -> {out_pdf}")


def plot_rate_overall_bar(df_rates: pd.DataFrame, out_pdf: Path, show_legend: bool = False):
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
    ax.bar(df_rates["bucket"].astype(str), df_rates["rate"].astype(float))
    ax.set_xlabel("衝突相手カテゴリ")
    ax.set_ylabel("割合 [%]")
    common_axes_style(ax)
    ax.set_ylim(0.0, max(5.0, float(df_rates["rate"].max()) * 1.15) if not df_rates.empty else 5.0)
    if show_legend:
        ax.legend(framealpha=1.0, facecolor="white", ncol=2)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存 -> {out_pdf}")


def save_legend_pdf(handles, labels, out_pdf: Path, ncol: int = 2):
    fig = plt.figure(figsize=(6, 1.2), dpi=300)
    fig.legend(handles, labels, loc="center", ncol=ncol, framealpha=1.0, facecolor="white")
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer())
    fig.savefig(out_pdf, format="pdf", bbox_inches=bbox)
    plt.close(fig)
    print(f"[OK] 凡例を保存 -> {out_pdf}")


def plot_rate_by_lead_stacked_bars(
    df_lead_rates: pd.DataFrame,
    bucket_order: List[str],
    out_pdf: Path,
    legend_pdf: Optional[Path] = None,
    show_legend_in_plot: bool = False,
):
    """
    100%積み上げ棒（leadごとに1本）
    legend_pdf を指定すると凡例だけ別PDFで保存する．
    図にタイトルは入れない．
    """
    df_lead_rates = df_lead_rates.sort_values("lead_sec_plot").copy()
    x = df_lead_rates["lead_sec_plot"].astype(int)
    x_labels = x.astype(str).tolist()

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    bottom = pd.Series([0.0] * len(df_lead_rates), index=df_lead_rates.index)
    bar_handles = []
    labels = []

    for b in bucket_order:
        if b not in df_lead_rates.columns:
            continue
        y = df_lead_rates[b].astype(float)
        cont = ax.bar(x_labels, y, bottom=bottom, label=b)
        bottom = bottom + y
        bar_handles.append(cont[0])
        labels.append(b)

    ax.set_xlabel("予測開始時刻 [s]")
    ax.set_ylabel("割合 [%]")
    ax.set_ylim(0.0, 100.0)

    common_axes_style(ax)

    if show_legend_in_plot:
        ax.legend(framealpha=1.0, facecolor="white", ncol=2)

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 保存 -> {out_pdf}")

    if legend_pdf is not None:
        save_legend_pdf(bar_handles, labels, legend_pdf, ncol=2)


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="results_grid_accident")
    ap.add_argument("--outdir", type=str, default=None)
    ap.add_argument("--methods", type=str, nargs="*", default=None)

    ap.add_argument("--base-payload-frame", type=int, default=25411)
    ap.add_argument("--dt", type=float, default=0.1)

    ap.add_argument("--post-window-sec", type=float, default=None,
                    help="switch以降の何秒間だけ集計するか．未指定ならswitch以降全部．")
    ap.add_argument("--until-base", action="store_true",
                    help="switchから base_payload_frame までに限定する．")

    ap.add_argument("--only-accident", action="store_true",
                    help="is_accident==1 のみ集計する．")
    ap.add_argument("--min-intensity", type=float, default=None,
                    help="intensity >= 閾値 のみ集計する（重複排除後に適用）．")

    ap.add_argument("--legend-mode", type=str, default="per_method",
                    choices=["none", "per_method", "shared"],
                    help="凡例PDFの出し方: none=出さない, per_method=手法ごと, shared=共通で1枚")

    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir) if args.outdir else (root / "out_collisions_total_and_rates_v5_legend_notitle")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "csv").mkdir(exist_ok=True)

    runs = discover_runs(root, methods=args.methods)
    if not runs:
        raise SystemExit(f"No runs found under {root}/<method>/lead_*/rep_*/logs/collisions.csv")

    fps = 1.0 / args.dt
    frames_per_sec = int(round(fps))
    if abs(frames_per_sec - fps) > 1e-6:
        raise SystemExit(f"dt={args.dt} does not yield integer frames/sec. Got {fps}.")

    bucket_order = ["vehicle", "static.mobility", "static.other", "walker", "unknown"]

    totals = []  # method, lead, rep, total
    rows = []    # method, lead, rep, bucket, count

    for run in runs:
        df = pd.read_csv(run.collisions_csv)

        required = ["payload_frame", "actor_id", "actor_type", "other_id", "other_type", "intensity", "is_accident"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise SystemExit(f"{run.collisions_csv} missing columns: {missing}")

        switch_pf = args.base_payload_frame - run.lead * frames_per_sec

        df["payload_frame"] = pd.to_numeric(df["payload_frame"], errors="coerce")
        df = df[df["payload_frame"].notna()].copy()

        # switch直後から
        df = df[df["payload_frame"] >= switch_pf]

        # 範囲上限
        if args.until_base:
            df = df[df["payload_frame"] <= args.base_payload_frame]
        elif args.post_window_sec is not None:
            df = df[df["payload_frame"] <= switch_pf + args.post_window_sec * frames_per_sec]

        # 主体は車両だけ
        df["actor_type"] = df["actor_type"].astype(str)
        df = df[df["actor_type"].str.startswith("vehicle")]

        # 事故フラグ
        if args.only_accident:
            df["is_accident"] = pd.to_numeric(df["is_accident"], errors="coerce").fillna(0).astype(int)
            df = df[df["is_accident"] == 1]

        # 重複排除
        df["event_key"] = canonical_event_key(df)
        df = df.drop_duplicates(subset=["event_key"]).copy()

        # intensity閾値
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        if args.min_intensity is not None:
            df = df[df["intensity"].fillna(-1) >= args.min_intensity]

        total_n = int(len(df))
        totals.append({"method": run.method, "lead": run.lead, "rep": run.rep, "total": total_n})

        if total_n == 0:
            continue

        df["bucket"] = df["other_type"].map(other_bucket)
        cnt = df.groupby("bucket").size().reset_index(name="count")
        for _, r in cnt.iterrows():
            rows.append(
                {
                    "method": run.method,
                    "lead": run.lead,
                    "rep": run.rep,
                    "bucket": r["bucket"],
                    "count": int(r["count"]),
                }
            )

        if args.debug:
            print(f"[INFO] {run.method} lead={run.lead} rep={run.rep} events={total_n}")

    totals_df = pd.DataFrame(totals)

    # 出力名サフィックス
    suffix = []
    suffix.append("acc1" if args.only_accident else "all")
    if args.min_intensity is not None:
        suffix.append(f"int{args.min_intensity:g}")
    if args.until_base:
        suffix.append("untilbase")
    elif args.post_window_sec is not None:
        suffix.append(f"win{args.post_window_sec:g}s")
    suffix.append(f"base{args.base_payload_frame}")
    suffix = "_".join(suffix)

    # -------------------------------------------------------
    # 表: leadごとの総衝突回数（rep平均）※図にしない
    # -------------------------------------------------------
    total_by_lead = (
        totals_df.groupby(["method", "lead"])["total"].mean()
        .reset_index()
        .sort_values(["method", "lead"])
    )
    total_by_lead["lead_sec_plot"] = -total_by_lead["lead"].astype(int)
    total_by_lead.to_csv(outdir / "csv" / f"total_collisions_by_lead_{suffix}.csv", index=False)

    # -------------------------------------------------------
    # 図: 総衝突回数（全lead統合，手法比較）
    # -------------------------------------------------------
    total_overall = (
        totals_df.groupby(["method", "rep"])["total"].sum()
        .reset_index()
        .groupby(["method"])["total"].mean()
        .reset_index()
        .sort_values("method")
    )
    total_overall.to_csv(outdir / "csv" / f"total_collisions_overall_{suffix}.csv", index=False)

    plot_total_overall(
        total_overall,
        out_pdf=outdir / f"fig_total_collisions_overall_{suffix}.pdf",
    )

    # -------------------------------------------------------
    # bucket別カウントのrep平均（lead別, overall）
    # -------------------------------------------------------
    if not rows:
        print("[WARN] No bucket rows after filtering. Done.")
        print(f"[OK] 出力先 -> {outdir}")
        return

    rows_df = pd.DataFrame(rows)

    piv_run = (
        rows_df.groupby(["method", "lead", "rep", "bucket"])["count"].sum()
        .reset_index()
        .pivot_table(index=["method", "lead", "rep"], columns="bucket", values="count", fill_value=0)
        .reset_index()
    )
    for b in bucket_order:
        if b not in piv_run.columns:
            piv_run[b] = 0

    # lead別 rep平均
    mean_by_lead = (
        piv_run.groupby(["method", "lead"])[bucket_order].mean()
        .reset_index()
        .sort_values(["method", "lead"])
    )
    mean_by_lead["lead_sec_plot"] = -mean_by_lead["lead"].astype(int)
    mean_by_lead.to_csv(outdir / "csv" / f"bucket_counts_mean_by_lead_{suffix}.csv", index=False)

    # overall rep平均（rep単位でlead合算）
    run_overall = piv_run.groupby(["method", "rep"])[bucket_order].sum().reset_index()
    mean_overall = run_overall.groupby(["method"])[bucket_order].mean().reset_index().sort_values("method")
    mean_overall.to_csv(outdir / "csv" / f"bucket_counts_mean_overall_{suffix}.csv", index=False)

    # -------------------------------------------------------
    # 図: overall割合（methodごと，棒）
    # -------------------------------------------------------
    for method in ["autopilot", "lstm"]:
        sub = mean_overall[mean_overall["method"] == method]
        if sub.empty:
            continue
        counts = sub.iloc[0][bucket_order].astype(float)
        total = float(counts.sum())
        rates = (100.0 * counts / total) if total > 0 else (counts * 0.0)

        df_rates = pd.DataFrame({"bucket": bucket_order, "rate": rates.values})
        df_rates = df_rates.sort_values("rate", ascending=False)

        plot_rate_overall_bar(
            df_rates,
            out_pdf=outdir / f"fig_bucket_rate_overall_{method}_{suffix}.pdf",
            show_legend=False,
        )

    # -------------------------------------------------------
    # 図: lead別割合（methodごと，100%積み上げ棒で1枚）+ 凡例PDF
    # -------------------------------------------------------
    legend_written = False
    for method in ["autopilot", "lstm"]:
        subM = mean_by_lead[mean_by_lead["method"] == method].copy()
        if subM.empty:
            continue

        denom = subM[bucket_order].sum(axis=1).astype(float)
        denom = denom.where(denom > 0, 1.0)

        rates_df = subM[["lead_sec_plot"]].copy()
        for b in bucket_order:
            rates_df[b] = 100.0 * subM[b].astype(float) / denom

        rates_df.to_csv(outdir / "csv" / f"bucket_rates_by_lead_{method}_{suffix}.csv", index=False)

        legend_pdf = None
        if args.legend_mode == "per_method":
            legend_pdf = outdir / f"legend_bucket_rate_by_lead_stacked_{method}_{suffix}.pdf"
        elif args.legend_mode == "shared":
            if not legend_written:
                legend_pdf = outdir / f"legend_bucket_rate_by_lead_stacked_shared_{suffix}.pdf"
                legend_written = True

        plot_rate_by_lead_stacked_bars(
            rates_df,
            bucket_order=bucket_order,
            out_pdf=outdir / f"fig_bucket_rate_by_lead_stacked_{method}_{suffix}.pdf",
            legend_pdf=legend_pdf,
            show_legend_in_plot=False,
        )

    print(f"[OK] 出力先 -> {outdir}")


if __name__ == "__main__":
    main()
