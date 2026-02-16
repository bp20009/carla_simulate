#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

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
# 読み込みと正規化
# -------------------------------------------------------
def detect_mode_col(df: pd.DataFrame) -> str:
    """
    期待:
      - mode 列がある，または
      - rendering 列があり，その中身が rendering/no_rendering
    """
    # まずは素直に mode
    if "mode" in df.columns:
        return "mode"

    # あなたのケース: 列名が rendering
    if "rendering" in df.columns:
        return "rendering"

    # それっぽい列名も一応見る（保険）
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in ("run_mode", "mode_name", "render_mode"):
            return c

    raise SystemExit(
        "mode列が見つかりません．\n"
        "期待する列名は mode または rendering です．\n"
        f"columns:\n  {', '.join(df.columns.astype(str))}"
    )


def load_one_summary(csv_path: Path, label: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    mode_col = detect_mode_col(df)

    required = [mode_col, "requested_actors", "speedup_median"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(
            f"schema mismatch: {csv_path}\n"
            f"missing={missing}\n"
            f"columns:\n  {', '.join(df.columns.astype(str))}"
        )

    out = df[required].copy()
    out = out.rename(columns={mode_col: "mode"})

    out["mode"] = out["mode"].astype(str).str.lower()
    out["requested_actors"] = pd.to_numeric(out["requested_actors"], errors="coerce")
    out["speedup_median"] = pd.to_numeric(out["speedup_median"], errors="coerce")
    out = out.dropna(subset=["mode", "requested_actors", "speedup_median"]).copy()

    out["requested_actors"] = out["requested_actors"].astype(int)
    out["run_tag"] = str(label)
    return out


def discover_default_files(root: Path) -> Dict[str, Path]:
    """
    root直下に以下がある想定:
      time_accel_benchmark_005_summary.csv
      time_accel_benchmark_010_summary.csv
      time_accel_benchmark_020_summary.csv
      time_accel_benchmark_050_summary.csv
    """
    out: Dict[str, Path] = {}
    for tag in ["005", "010", "020", "050"]:
        p = root / f"time_accel_benchmark_{tag}_summary.csv"
        if p.exists():
            out[tag] = p
    return out


# -------------------------------------------------------
# 描画
# -------------------------------------------------------
TAG_LABEL = {
    "005": "0.05",
    "010": "0.10",
    "020": "0.20",
    "050": "0.50",
}

def plot_speedup_lines(
    df: pd.DataFrame,
    out_pdf: Path,
    mode: str,
    tag_order: List[str],
    title: str = "",
):
    mode = mode.lower()
    sub = df[df["mode"] == mode].copy()
    if sub.empty:
        raise SystemExit(f"No rows for mode={mode}. available modes={sorted(df['mode'].unique().tolist())}")

    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=300)

    for tag in tag_order:
        ss = sub[sub["run_tag"] == tag].copy()
        if ss.empty:
            continue
        ss = ss.sort_values("requested_actors")
        ax.plot(
            ss["requested_actors"],
            ss["speedup_median"],
            marker="o",
            linewidth=2,
            label=TAG_LABEL.get(tag, tag),
        )

    ax.set_xlabel("車両数")
    ax.set_ylabel("平均加速倍率")
    ax.set_ylim(bottom=0) 
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", direction="in")
    ax.set_axisbelow(True)

    if title:
        ax.set_title(title)

    ax.legend(framealpha=1.0, facecolor="white")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_pdf}")



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="summary CSVがあるディレクトリ")
    ap.add_argument("--outdir", type=str, default="", help="出力先（未指定ならroot/out_speedup_plots）")

    ap.add_argument("--csv-005", type=str, default="")
    ap.add_argument("--csv-010", type=str, default="")
    ap.add_argument("--csv-020", type=str, default="")
    ap.add_argument("--csv-050", type=str, default="")

    ap.add_argument("--title", action="store_true", help="タイトルを入れる（デフォルトは無し）")
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    # 入力ファイル解決
    files: Dict[str, Path] = {}
    if args.csv_005:
        files["005"] = Path(args.csv_005).expanduser().resolve()
    if args.csv_010:
        files["010"] = Path(args.csv_010).expanduser().resolve()
    if args.csv_020:
        files["020"] = Path(args.csv_020).expanduser().resolve()
    if args.csv_050:
        files["050"] = Path(args.csv_050).expanduser().resolve()

    if not files:
        files = discover_default_files(root)

    if not files:
        raise SystemExit(
            "No input CSVs found. Provide --csv-005 ... or place files like "
            "time_accel_benchmark_005_summary.csv in --root."
        )

    tag_order = ["005", "010", "020", "050"]
    available_tags = [t for t in tag_order if t in files]
    if not available_tags:
        raise SystemExit("No expected tags found among inputs.")

    # 読み込み統合
    all_df: List[pd.DataFrame] = []
    for tag in available_tags:
        all_df.append(load_one_summary(files[tag], label=tag))
    df = pd.concat(all_df, ignore_index=True)

    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else (root / "out_speedup_plots")
    outdir.mkdir(parents=True, exist_ok=True)

    t1 = "rendering: speedup_median vs actors" if args.title else ""
    t2 = "no_rendering: speedup_median vs actors" if args.title else ""

    plot_speedup_lines(
        df=df,
        out_pdf=outdir / "fig_speedup_median_rendering.pdf",
        mode="rendering",
        tag_order=available_tags,
        title=t1,
    )
    plot_speedup_lines(
        df=df,
        out_pdf=outdir / "fig_speedup_median_no_rendering.pdf",
        mode="no_rendering",
        tag_order=available_tags,
        title=t2,
    )

    print(f"[OK] 出力先: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
