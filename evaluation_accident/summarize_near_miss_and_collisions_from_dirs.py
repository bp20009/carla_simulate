#!/usr/bin/env python3
"""
results_grid_accident/ 以下のディレクトリ構成

  results_grid_accident/
    autopilot/
      lead_1/
        rep_1/
          logs/collisions.csv
        rep_2/
          logs/collisions.csv
      ...
    lstm/
      lead_1/
        rep_1/
          logs/collisions.csv
      ...

をたどって collisions.csv を集約し，
near_miss_events_d3_noCollisionFrames.csv とマージして
method, lead_sec, rep ごとのサマリを作るスクリプト．

collisions.csv のカラム想定:
time_sec,payload_frame,payload_frame_source,carla_frame,
actor_id,actor_type,other_id,other_type,other_class,
x,y,z,intensity,is_accident
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


def classify_intensity(intensity: float) -> str:
    """コリジョン強度をカテゴリに分ける."""
    if pd.isna(intensity) or intensity <= 0:
        return "no_collision_or_zero"
    if intensity < 1000:
        return "contact_lt_1000"        # 軽微接触・ニアミス寄り
    if intensity < 5000:
        return "accident_1000_4999"     # 中程度
    if intensity < 10000:
        return "accident_5000_9999"     # やや重い
    return "accident_ge_10000"          # かなり重い事故


def find_collision_files(base_dir: Path) -> List[Tuple[str, int, int, Path]]:
    """
    base_dir 以下から collisions.csv を探し，
    (method, lead_sec, rep, path) のリストを返す．
    期待するパス構造:
      base_dir / <method> / lead_<N> / rep_<M> / logs / collisions.csv
    """
    results: List[Tuple[str, int, int, Path]] = []
    for path in base_dir.rglob("collisions.csv"):
        try:
            logs_dir = path.parent          # logs
            rep_dir = logs_dir.parent       # rep_Y
            lead_dir = rep_dir.parent       # lead_X
            method_dir = lead_dir.parent    # method

            method = method_dir.name          # "autopilot" / "lstm"
            lead_str = lead_dir.name.split("_", 1)[1]  # "lead_1" -> "1"
            rep_str = rep_dir.name.split("_", 1)[1]    # "rep_1" -> "1"

            lead_sec = int(lead_str)
            rep = int(rep_str)
        except Exception:
            # 想定外の構造は無視
            continue

        results.append((method, lead_sec, rep, path))
    return results


def load_collisions(base_dir: Path) -> pd.DataFrame:
    """ディレクトリから collisions.csv を集約して DataFrame にする."""
    records = find_collision_files(base_dir)
    if not records:
        raise RuntimeError(f"No collisions.csv found under {base_dir}")

    dfs: List[pd.DataFrame] = []
    for method, lead_sec, rep, path in records:
        df = pd.read_csv(path)

        # 想定カラムチェック（最低限 intensity と is_accident があるか）
        if "intensity" not in df.columns or "is_accident" not in df.columns:
            raise ValueError(f"{path} に 'intensity' または 'is_accident' 列がありません。")

        # 型を軽くそろえておく
        df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")
        df["is_accident"] = pd.to_numeric(df["is_accident"], errors="coerce").fillna(0).astype(int)

        # ディレクトリから取得した情報を列として付加
        df["method"] = method
        df["lead_sec"] = lead_sec
        df["rep"] = rep

        dfs.append(df)

    all_collisions = pd.concat(dfs, ignore_index=True)
    return all_collisions


def summarize(
    base_dir: Path,
    near_miss_events_path: Path,
    out_per_run: Path,
    out_per_method: Path,
) -> None:
    # === 1. near miss 読み込み ===
    near = pd.read_csv(near_miss_events_path)

    # near_miss_events_d3_noCollisionFrames.csv が
    # method, lead_sec, rep を持っている前提
    key_cols = ["method", "lead_sec", "rep"]

    near_summary_per_run = (
        near.groupby(key_cols)
        .size()
        .reset_index(name="n_near_miss_dist")
    )

    near_summary_per_method = (
        near.groupby(["method", "lead_sec"])
        .size()
        .reset_index(name="n_near_miss_dist")
    )

    # === 2. collisions 読み込み & intensity 分類 ===
    coll = load_collisions(base_dir)

    coll["intensity_severity"] = coll["intensity"].apply(classify_intensity)

    # is_accident のカウント（ログ上で事故フラグが立ったフレーム数）
    # ※ intensity>=1000 とほぼ一致するはずだが，別指標として残す
    coll["is_accident_flag"] = coll["is_accident"].astype(int)

    # === 3. コリジョン集計（per run） ===
    # intensity のカテゴリ別件数
    coll_counts_per_run = (
        coll.groupby(key_cols)["intensity_severity"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    # is_accident (0/1) の合計
    acc_flag_per_run = (
        coll.groupby(key_cols)["is_accident_flag"]
        .sum()
        .reset_index(name="n_is_accident_flag")
    )

    coll_summary_per_run = pd.merge(
        coll_counts_per_run, acc_flag_per_run, on=key_cols, how="left"
    )

    # === 4. コリジョン集計（per method, lead_sec） ===
    coll_counts_per_method = (
        coll.groupby(["method", "lead_sec"])["intensity_severity"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    acc_flag_per_method = (
        coll.groupby(["method", "lead_sec"])["is_accident_flag"]
        .sum()
        .reset_index(name="n_is_accident_flag")
    )

    coll_summary_per_method = pd.merge(
        coll_counts_per_method, acc_flag_per_method, on=["method", "lead_sec"], how="left"
    )

    # === 5. near miss + collisions マージ（per run） ===
    per_run = pd.merge(
        near_summary_per_run,
        coll_summary_per_run,
        on=key_cols,
        how="outer",
    ).fillna(0)

    for c in per_run.columns:
        if c not in key_cols:
            per_run[c] = per_run[c].astype(int)

    # === 6. near miss + collisions マージ（per method, lead_sec） ===
    per_method = pd.merge(
        near_summary_per_method,
        coll_summary_per_method,
        on=["method", "lead_sec"],
        how="outer",
    ).fillna(0)

    for c in per_method.columns:
        if c not in ["method", "lead_sec"]:
            per_method[c] = per_method[c].astype(int)

    # === 7. 書き出し ===
    out_per_run.parent.mkdir(parents=True, exist_ok=True)
    out_per_method.parent.mkdir(parents=True, exist_ok=True)

    per_run.to_csv(out_per_run, index=False)
    per_method.to_csv(out_per_method, index=False)

    print(f"[OK] per-run summary    -> {out_per_run}")
    print(f"[OK] per-method summary -> {out_per_method}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize near-miss (distance) and collisions (intensity) from results_grid_accident/*/*/logs/collisions.csv"
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="results_grid_accident ディレクトリへのパス",
    )
    p.add_argument(
        "--near-miss-events",
        type=Path,
        required=True,
        help="near_miss_events_d3_noCollisionFrames.csv へのパス",
    )
    p.add_argument(
        "--out-per-run",
        type=Path,
        default=Path("near_miss_and_collisions_per_run.csv"),
        help="method, lead_sec, rep ごとのサマリ CSV 出力先",
    )
    p.add_argument(
        "--out-per-method",
        type=Path,
        default=Path("near_miss_and_collisions_per_method.csv"),
        help="method, lead_sec ごとのサマリ CSV 出力先",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summarize(
        base_dir=args.base_dir,
        near_miss_events_path=args.near_miss_events,
        out_per_run=args.out_per_run,
        out_per_method=args.out_per_method,
    )


if __name__ == "__main__":
    main()
