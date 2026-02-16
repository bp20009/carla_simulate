#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# -------------------------------------------------------
# フォント設定（macOS向け・IPAexフォールバック付き）
# -------------------------------------------------------
cand_fonts = ["BIZ UDPGothic","Hiragino Sans", "YuGothic", "Osaka", "BIZ UDGothic", "IPAexGothic"]
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

plt.rcParams["font.size"] = 14          # 基本の文字サイズ
plt.rcParams["axes.labelsize"] = 16     # x,y軸ラベル
plt.rcParams["xtick.labelsize"] = 14    # x軸目盛ラベル
plt.rcParams["ytick.labelsize"] = 14    # y軸目盛ラベル
plt.rcParams["legend.fontsize"] = 14    # 凡例


# -------------------------------------------------------
# 図作成関数
# -------------------------------------------------------
def plot_lane_deviation(csv_path: Path, out_pdf: Path):
    df = pd.read_csv(csv_path)

    required = ["method", "lead_sec", "lane_over_th_rate_mean"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{csv_path} に必須列 {c} がありません")

    # null/空欄を0にする
    df["lane_over_th_rate_mean"] = pd.to_numeric(df["lane_over_th_rate_mean"], errors="coerce")
    df["lane_over_th_rate_mean"] = df["lane_over_th_rate_mean"].fillna(0.0)

    # 並び順制御
    method_order = ["autopilot", "lstm"]
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["lead_sec", "method"])

    # ★ プロット用に「残り時間 = -lead_sec」を作る（事故フレームを0とする）
    df["lead_sec_plot"] = -df["lead_sec"]

    # 図
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    for m in method_order:
        sub = df[df["method"] == m]
        if sub.empty:
            continue
        label = "Autopilot" if m == "autopilot" else "LSTM"
        ax.plot(
            sub["lead_sec_plot"],
            sub["lane_over_th_rate_mean"],
            marker="o",
            linewidth=2,
            label=label,
        )

    # x軸：-10〜0 にして，左右に0.5ずつゆとり
    ax.set_xlim(-10.5, 0.5)
    ax.set_xticks(list(range(-10, 1, 1)))

    ax.set_xlabel("予測開始時刻 [s]")
    ax.set_ylabel("車線逸脱率")

    # y軸は 0〜1.1 にして他の図と縦スケールを合わせる
    ax.set_ylim(-0.05, 1.1)

    # グリッド
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", direction="in")

    # ★ 事故フレームライン（x=0 に赤い縦線）
    ax.axvline(
        x=0,
        color="red",
        linewidth=1.8,
        linestyle="-",
    )

    # ★ 事故フレームラベル（位置はいい感じに調整）
    ax.text(
        -0.6,        # x位置（少し左にずらす）
        0.7,         # y位置
        "事故発生フレーム",
        ha="left",
        va="center",
        fontsize=14,
        color="black",
        rotation=90,
    )

    # 凡例：右下寄り，背景は白で不透明
    ax.legend(
        framealpha=1.0,
        facecolor="white",
    )

    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] 図を保存しました -> {out_pdf}")


# -------------------------------------------------------
# CLI
# -------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-csv", required=True, help="actor_metrics_by_method_lead.csv")
    ap.add_argument("--out-pdf", required=True, help="出力PDFのパス")
    args = ap.parse_args()

    plot_lane_deviation(Path(args.metrics_csv), Path(args.out_pdf))


if __name__ == "__main__":
    main()
