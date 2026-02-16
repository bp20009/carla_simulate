#!/usr/bin/env python3
"""
ニアミス + コリジョン集計結果の可視化・比較スクリプト

入力:
- results_grid_accident/near_miss_and_collisions_per_method.csv
- results_grid_accident/near_miss_and_collisions_per_run.csv

出力:
1) lead_sec × intensityカテゴリのヒートマップ (method ごと)
2) heavy accident (intensity >= 10000) 件数 vs lead_sec の折れ線グラフ
3) LSTM vs Autopilot の比較テーブル (CSV)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ===== 共通設定 =====

BASE_DIR = Path("results_grid_accident")
PER_METHOD_PATH = BASE_DIR / "near_miss_and_collisions_per_method.csv"
PER_RUN_PATH = BASE_DIR / "near_miss_and_collisions_per_run.csv"

# intensity カテゴリ列の順番
INTENSITY_CATEGORIES = [
    "contact_lt_1000",
    "accident_1000_4999",
    "accident_5000_9999",
    "accident_ge_10000",
]


# ===== 1. ヒートマップ: lead_sec × intensityカテゴリ =====

def plot_intensity_heatmap(per_method: pd.DataFrame, method: str, normalize: bool = False) -> None:
    """
    指定した method (autopilot / lstm) について、
    lead_sec × intensityカテゴリ のヒートマップを描画し PNG で保存する。

    normalize=True の場合は各 lead_sec 行ごとに割合(%)に正規化。
    """
    df = per_method[per_method["method"] == method].copy()
    if df.empty:
        print(f"[WARN] method={method} のデータがありません。ヒートマップをスキップします。")
        return

    # lead_sec でソート
    df = df.sort_values("lead_sec")

    # 欠けているカテゴリ列があれば 0 で追加
    for col in INTENSITY_CATEGORIES:
        if col not in df.columns:
            df[col] = 0

    # 行: lead_sec, 列: intensityカテゴリ の行列を作成
    heat_df = df.set_index("lead_sec")[INTENSITY_CATEGORIES]

    if normalize:
        # 行ごとに合計で割って割合に（0除算対策込み）
        row_sums = heat_df.sum(axis=1).replace(0, 1)
        heat_df = heat_df.div(row_sums, axis=0) * 100.0

    # 描画
    fig, ax = plt.subplots()
    im = ax.imshow(heat_df.values, aspect="auto")

    # 軸ラベル
    ax.set_xticks(range(len(INTENSITY_CATEGORIES)))
    ax.set_xticklabels(INTENSITY_CATEGORIES, rotation=45, ha="right")
    ax.set_yticks(range(len(heat_df.index)))
    ax.set_yticklabels(heat_df.index)
    ax.set_xlabel("intensity category")
    ax.set_ylabel("lead_sec")
    title_suffix = " (normalized %)" if normalize else " (count)"
    ax.set_title(f"{method} - collisions heatmap{title_suffix}")

    # カラーバー
    fig.colorbar(im, ax=ax)

    # 値をセル内に表示（オプション）
    for i, lead in enumerate(heat_df.index):
        for j, cat in enumerate(INTENSITY_CATEGORIES):
            value = heat_df.iloc[i, j]
            if normalize:
                text = f"{value:.1f}"
            else:
                text = f"{int(value)}"
            ax.text(j, i, text, ha="center", va="center")

    # 保存
    out_name = f"heatmap_{method}{'_norm' if normalize else ''}.png"
    out_path = BASE_DIR / out_name
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Saved heatmap for {method} -> {out_path}")


# ===== 2. heavy accident (>=10000) 件数 vs lead_sec 折れ線グラフ =====

def plot_heavy_accidents_vs_lead(per_method: pd.DataFrame) -> None:
    """
    method ごとに heavy accident (accident_ge_10000) 件数を lead_sec ごとに折れ線で描画。
    """
    fig, ax = plt.subplots()

    for method in sorted(per_method["method"].unique()):
        df = per_method[per_method["method"] == method].copy()
        if "accident_ge_10000" not in df.columns:
            continue
        df = df.sort_values("lead_sec")
        ax.plot(df["lead_sec"], df["accident_ge_10000"], marker="o", label=method)

    ax.set_xlabel("lead_sec (switch time)")
    ax.set_ylabel("num heavy accidents (intensity >= 10000)")
    ax.set_title("Heavy accidents vs lead_sec")
    ax.legend()

    out_path = BASE_DIR / "heavy_accidents_vs_lead.png"
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[OK] Saved heavy accidents vs lead plot -> {out_path}")


# ===== 3. LSTM vs Autopilot の比較テーブル作成 =====

def build_method_comparison_table(
    per_method: pd.DataFrame,
    per_run: pd.DataFrame,
    out_path: Path,
) -> None:
    """
    LSTM / Autopilot を比較するテーブルを生成して CSV に書き出す。

    出力テーブル (例):
    method,
      total_runs,
      total_near_miss_dist,
      total_contact_lt_1000,
      total_accident_1000_4999,
      total_accident_5000_9999,
      total_accident_ge_10000,
      total_is_accident_flag,
      avg_near_miss_dist_per_run,
      avg_heavy_accidents_per_run
    """
    rows = []

    for method in sorted(per_method["method"].unique()):
        pm = per_method[per_method["method"] == method]
        pr = per_run[per_run["method"] == method]

        total_runs = len(pr[["lead_sec", "rep"]].drop_duplicates())

        total_near_miss_dist = pm["n_near_miss_dist"].sum()
        total_contact_lt_1000 = pm.get("contact_lt_1000", pd.Series([0])).sum()
        total_acc_1000_4999 = pm.get("accident_1000_4999", pd.Series([0])).sum()
        total_acc_5000_9999 = pm.get("accident_5000_9999", pd.Series([0])).sum()
        total_acc_ge_10000 = pm.get("accident_ge_10000", pd.Series([0])).sum()
        total_is_accident_flag = pm.get("n_is_accident_flag", pd.Series([0])).sum()

        # run 単位での平均（荒めの比較用）
        avg_near_miss_dist_per_run = (
            pr["n_near_miss_dist"].mean() if "n_near_miss_dist" in pr.columns else 0.0
        )
        avg_heavy_accidents_per_run = (
            pr.get("accident_ge_10000", pd.Series([0])).mean()
            if "accident_ge_10000" in pr.columns
            else 0.0
        )

        rows.append(
            {
                "method": method,
                "total_runs": total_runs,
                "total_near_miss_dist": int(total_near_miss_dist),
                "total_contact_lt_1000": int(total_contact_lt_1000),
                "total_accident_1000_4999": int(total_acc_1000_4999),
                "total_accident_5000_9999": int(total_acc_5000_9999),
                "total_accident_ge_10000": int(total_acc_ge_10000),
                "total_is_accident_flag": int(total_is_accident_flag),
                "avg_near_miss_dist_per_run": float(avg_near_miss_dist_per_run),
                "avg_heavy_accidents_per_run": float(avg_heavy_accidents_per_run),
            }
        )

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(out_path, index=False)
    print(f"[OK] Saved method comparison table -> {out_path}")


# ===== メイン関数 =====

def main() -> None:
    if not PER_METHOD_PATH.exists():
        raise FileNotFoundError(PER_METHOD_PATH)
    if not PER_RUN_PATH.exists():
        raise FileNotFoundError(PER_RUN_PATH)

    per_method = pd.read_csv(PER_METHOD_PATH)
    per_run = pd.read_csv(PER_RUN_PATH)

    # 1. ヒートマップ (method ごと)
    for m in sorted(per_method["method"].unique()):
        plot_intensity_heatmap(per_method, method=m, normalize=False)
        plot_intensity_heatmap(per_method, method=m, normalize=True)

    # 2. heavy accidents vs lead_sec
    plot_heavy_accidents_vs_lead(per_method)

    # 3. LSTM vs Autopilot 比較テーブル
    out_comp = BASE_DIR / "method_comparison_summary.csv"
    build_method_comparison_table(per_method, per_run, out_comp)


if __name__ == "__main__":
    main()
