#!/usr/bin/env python3
"""
実験結果から以下の2枚の図をPDFで出力するスクリプト．

図1: 衝突強度別の事故発生回数（methodごとの総数）の比較
      - 入力: near_miss_and_collisions_per_method.csv
      - y軸: 事故回数（lead_sec 全体で合計）
      - x軸: method (autopilot / lstm)
      - 棒: intensity 1000–4999, 5000–9999, >=10000 のスタック棒グラフ

図2: 危険領域への侵入率の比較
      - 入力: predicted_risk_summary.csv
      - x軸: lead_sec
      - y軸: risk_rate[%]
      - 線: method ごとの侵入率 (autopilot / lstm)

フォントは MS ゴシックを指定（環境にインストールされていない場合は
別フォントにフォールバックする可能性あり）．
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd


def setup_japanese_font() -> None:
    import matplotlib

    # YuGothic を明示的に指定（一覧に出ていたので確実に存在）
    matplotlib.rcParams["font.sans-serif"] = [
        "BIZ UDPGothic",
        "Hiragino Sans",
        "BIZ UDGothic",
        "Osaka",
        "Apple SD Gothic Neo",
        "AppleGothic",
    ]

    matplotlib.rcParams["font.size"] = 12          # 基本の文字サイズ
    matplotlib.rcParams["axes.labelsize"] = 14     # x,y軸ラベル
    matplotlib.rcParams["xtick.labelsize"] = 12    # x軸目盛ラベル
    matplotlib.rcParams["ytick.labelsize"] = 12    # y軸目盛ラベル
    matplotlib.rcParams["legend.fontsize"] = 12    # 凡例
    # PDF への埋め込み設定（TrueType のまま埋め込む）
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["axes.unicode_minus"] = False

    print("[INFO] Using Japanese font: YuGothic")

def plot_collision_intensity_summary(
    csv_path: Path,
    out_pdf: Path,
) -> None:
    """
    near_miss_and_collisions_per_method.csv から，
    強度レンジ別の事故回数を method ごとに集計し，
    スタック棒グラフで描画する．

    想定カラム:
      method,lead_sec,
      accident_1000_4999,accident_5000_9999,accident_ge_10000,...
    """
    df = pd.read_csv(csv_path)

    required_cols = [
        "method",
        "lead_sec",
        "accident_1000_4999",
        "accident_5000_9999",
        "accident_ge_10000",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} に必須列 '{c}' がありません．")

    # method ごとに lead_sec をまたいで合計
    agg = (
        df.groupby("method", as_index=False)[
            ["accident_1000_4999", "accident_5000_9999", "accident_ge_10000"]
        ]
        .sum()
    )

    methods = agg["method"].tolist()
    x = range(len(methods))

    fig, ax = plt.subplots(figsize=(6, 4))

    bottom = [0.0] * len(methods)
    labels = ["1000–4999", "5000–9999", "10000以上"]
    cols = ["accident_1000_4999", "accident_5000_9999", "accident_ge_10000"]

    for label, col in zip(labels, cols):
        values = agg[col].tolist()
        ax.bar(x, values, bottom=bottom, label=label)
        bottom = [b + v for b, v in zip(bottom, values)]

    ax.set_xticks(list(x))
    ax.set_xticklabels(methods)
    ax.set_ylabel("事故回数")
    ax.set_xlabel("手法")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_predicted_risk_summary(
    csv_path: Path,
    out_pdf: Path,
) -> None:
    df = pd.read_csv(csv_path)

    required_cols = ["method", "lead_sec", "risk_rate"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{csv_path} に必須列 '{c}' がありません．")

    df = df.copy()
    df["lead_sec"] = pd.to_numeric(df["lead_sec"], errors="coerce")
    df = df.dropna(subset=["lead_sec"])
    df = df.sort_values(["method", "lead_sec"])

    # ★ -10 → 0（事故フレーム）
    df["lead_sec_plot"] = -df["lead_sec"]

    methods = sorted(df["method"].unique())

    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)

    for method in methods:
        df_m = df[df["method"] == method]
        ax.plot(
            df_m["lead_sec_plot"],
            df_m["risk_rate"],
            marker="o",
            label = "LSTM" if method.lower()=="lstm" else method.capitalize()
            )

    ax.set_xlabel("予測開始時刻 [s]")
    ax.set_ylabel("事故リスク発生割合")

    # 事故フレームの明示
    ax.axvline(x=0, color="black", linewidth=1)
    # ax.text(0, 1.05, "事故フレーム", ha="center", va="bottom", fontsize=14)

    # y上限（1.05〜1.1）
    ax.set_ylim(-0.05, 1.1)

    # -10 → 0
    ax.set_xlim(-10.5, 0.5)
    ax.set_xticks(list(range(-10, 1, 1)))
    # === 事故フレームライン（赤） ===
    ax.axvline(
    x=0,
    color="red",      # ★ 赤の縦線
    linewidth=1.8,    # 少し太め
    linestyle="-"
    )
    # 事故フレーム表記も赤系に合わせて調整（任意）
    ax.text(
    -0.6, 0.7,
    "事故発生フレーム",
    ha="left",
    va="center",
    fontsize=12,
    color="black",
    rotation=90
    )


    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend(loc="lower right",
              framealpha=1.0,
              bbox_to_anchor=(0.95, 0.05)
              )

    ax.tick_params(axis="both", direction="in")
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Autopilot表記に統一しました -> {out_pdf}")
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--near-miss-collisions",
        type=Path,
        required=False,
        help="near_miss_and_collisions_per_method.csv のパス",
    )
    parser.add_argument(
        "--risk-summary",
        type=Path,
        required=False,
        help="predicted_risk_summary.csv のパス",
    )
    parser.add_argument(
        "--out-collision-pdf",
        type=Path,
        required=False,
        help="衝突強度別グラフの出力先 PDF パス",
    )
    parser.add_argument(
        "--out-risk-pdf",
        type=Path,
        required=False,
        help="危険領域侵入率グラフの出力先 PDF パス",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_japanese_font()

    # 図1: 衝突強度別の事故回数
    if args.near_miss_collisions and args.out_collision_pdf:
        print("[INFO] plot_collision_intensity_summary を実行します．")
        plot_collision_intensity_summary(
            csv_path=args.near_miss_collisions,
            out_pdf=args.out_collision_pdf,
        )

    # 図2: 危険領域侵入率
    if args.risk_summary and args.out_risk_pdf:
        print("[INFO] plot_predicted_risk_summary を実行します．")
        plot_predicted_risk_summary(
            csv_path=args.risk_summary,
            out_pdf=args.out_risk_pdf,
        )


if __name__ == "__main__":
    main()
