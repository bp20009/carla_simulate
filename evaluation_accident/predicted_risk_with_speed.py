#!/usr/bin/env python3
"""
results_grid_accident 以下の logs/actor.csv を用いて，
「予測された車両挙動が基準事故の危険領域に侵入した確率」を
method × lead_sec ごとに評価するスクリプト．

更に，risk_flag==1 の run について，
リスク成立区間に危険領域へ進入していた車両（actor_id）全てを対象に，
検出フレームを中心とした前後Nフレームの速度[km/h]を lead ごとにプロットする．

前提:
  - 基準事故: collisions_exp_accident.csv
      frame, x, y, z, intensity, is_accident を含む（少なくとも）
  - 予測挙動: <base_dir>/<method>/lead_<N>/rep_<M>/logs/actor.csv
      frame, id, location_x, location_y, location_z を含む
  - 切り替え情報: 同じ logs ディレクトリ内の meta.json
      switch_payload_frame_observed を持つと仮定（オプション）

危険領域の定義:
  - 時間:  以下のいずれか
      (A) デフォルト:
          frame ∈ [base_frame - frame_window_before, base_frame + frame_window]
      (B) --use-switch-window 指定時:
          frame ∈ [switch_frame, switch_frame + window_from_switch_sec / dt]
  - 空間:  基準事故座標 (x0, y0, z0) から半径 radius[m] 以内

1 run (method, lead_sec, rep) について，
  - 危険領域に「連続して min_consecutive_frames フレーム以上」
    かつ「同一フレーム内で min_actor_inside 台以上」
    侵入した場合に risk_flag = 1
  - そうでなければ risk_flag = 0

速度プロット:
  - risk_flag==1 の run について，
    リスク成立の最初の連続区間 [segment_start, segment_end] を取り，
    その区間で危険領域内にいた actor_id を全て抽出する．
  - 各 actor_id について，detect_frame=segment_end を中心に
    前後 speed_window_frames の速度[km/h]を算出し，
    lead ごとに重ね描きする．
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm


# -------------------------------------------------------
# フォント設定（他スクリプトと同一）
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
plt.rcParams["legend.fontsize"] = 12


# ========= 基準事故の読み込み =========

def load_baseline_accident(
    path: Path,
    base_frame: int | None = None,
    intensity_threshold: float = 1000.0,
) -> dict:
    """
    collisions_exp_accident.csv から
      frame, x, y, z, intensity
    を取得する．

    - base_frame が指定されていれば:
        frame == base_frame かつ is_accident == 1 の行から平均を取る
    - 指定されていなければ:
        is_accident == 1 かつ intensity >= threshold の中で
        intensity 最大の frame を基準とする
    """
    df = pd.read_csv(path)

    required = ["frame", "x", "y", "z", "intensity", "is_accident"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"baseline CSV に '{c}' 列が必要です．")

    df["is_accident"] = pd.to_numeric(df["is_accident"], errors="coerce").fillna(0).astype(int)
    df["intensity"] = pd.to_numeric(df["intensity"], errors="coerce")

    if base_frame is not None:
        cand = df[(df["frame"] == base_frame) & (df["is_accident"] == 1)]
        if cand.empty:
            raise ValueError(f"frame == {base_frame} かつ is_accident==1 の行が見つかりません．")
    else:
        cand = df[(df["is_accident"] == 1) & (df["intensity"] >= intensity_threshold)]
        if cand.empty:
            raise ValueError("is_accident==1 かつ intensity>=threshold の行が見つかりません．")
        max_row = cand.loc[cand["intensity"].idxmax()]
        base_frame = int(max_row["frame"])
        cand = df[df["frame"] == base_frame]

    baseline_intensity = cand["intensity"].mean()
    bx = cand["x"].mean()
    by = cand["y"].mean()
    bz = cand["z"].mean()

    return {
        "frame": int(base_frame),
        "x": float(bx),
        "y": float(by),
        "z": float(bz),
        "intensity": float(baseline_intensity),
    }


# ========= actor.csv の探索・読み込み =========

def find_actor_files(base_dir: Path) -> Dict[Tuple[str, int, int], Path]:
    """
    base_dir 以下から logs/actor.csv を探し，
      key = (method, lead_sec, rep)
      val = actor.csv のパス
    の dict を作る．

    想定ディレクトリ構造:
      base_dir / <method> / lead_<N> / rep_<M> / logs / actor.csv
    """
    result: Dict[Tuple[str, int, int], Path] = {}

    for path in base_dir.rglob("actor.csv"):
        try:
            logs_dir = path.parent
            rep_dir = logs_dir.parent
            lead_dir = rep_dir.parent
            method_dir = lead_dir.parent

            method = method_dir.name
            lead_sec = int(lead_dir.name.split("_", 1)[1])
            rep = int(rep_dir.name.split("_", 1)[1])
        except Exception:
            continue

        result[(method, lead_sec, rep)] = path

    if not result:
        raise RuntimeError(f"No actor.csv found under {base_dir}")

    return result


def load_actor_csv_as_positions(path: Path) -> pd.DataFrame:
    """
    logs/actor.csv を読み込み，
    内部表現: frame, actor_id, x, y, z に正規化する．
    """
    df = pd.read_csv(path)

    required = ["frame", "id", "location_x", "location_y", "location_z"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"{path} に必須列 '{c}' がありません．")

    df["frame"] = pd.to_numeric(df["frame"], errors="coerce").astype(int)
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype(int)

    for c in ["location_x", "location_y", "location_z"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df_norm = pd.DataFrame(
        {
            "frame": df["frame"],
            "actor_id": df["id"],
            "x": df["location_x"],
            "y": df["location_y"],
            "z": df["location_z"],
        }
    )
    df_norm = df_norm.dropna(subset=["frame", "actor_id", "x", "y", "z"]).copy()
    df_norm["frame"] = df_norm["frame"].astype(int)
    df_norm["actor_id"] = df_norm["actor_id"].astype(int)
    return df_norm


# ========= ユーティリティ =========

def max_consecutive_true(flags: List[bool]) -> int:
    max_run = 0
    cur = 0
    for v in flags:
        if v:
            cur += 1
            if cur > max_run:
                max_run = cur
        else:
            cur = 0
    return max_run


def load_switch_frame_from_meta(logs_dir: Path) -> int | None:
    meta_path = logs_dir / "meta.json"
    if not meta_path.exists():
        return None

    try:
        data = json.loads(meta_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None

    v = data.get("switch_payload_frame_observed")
    if v is None:
        return None

    try:
        return int(float(str(v).strip()))
    except Exception:
        return None


def first_segment_where_flags_run(
    frames: np.ndarray,
    flags: np.ndarray,
    min_consecutive: int,
) -> Tuple[int | None, int | None]:
    """
    frames: 昇順 frame
    flags: True の連続区間を探す
    min_consecutive: 必要連続長

    最初に条件を満たした区間の (segment_start, segment_end) を返す．
    segment_end は成立した瞬間のフレーム（連続の最後）．
    """
    if len(frames) == 0:
        return None, None

    run = 0
    for i, ok in enumerate(flags):
        if ok:
            run += 1
            if run >= min_consecutive:
                seg_end = int(frames[i])
                seg_start = int(frames[i - min_consecutive + 1])
                return seg_start, seg_end
        else:
            run = 0
    return None, None


def compute_speed_series_kmh_for_actor(
    df_pos_all: pd.DataFrame,
    actor_id: int,
    center_frame: int,
    dt: float,
    window_frames: int,
) -> pd.DataFrame:
    """
    actor_id の位置系列から速度[km/h]を算出して，
    center_frame±window_frames の範囲で返す．

    欠損フレームを考慮し，delta_t = dframe * dt を使用する．
    """
    fmin = int(center_frame - window_frames)
    fmax = int(center_frame + window_frames)

    df = df_pos_all[
        (df_pos_all["actor_id"] == int(actor_id))
        & (df_pos_all["frame"] >= fmin)
        & (df_pos_all["frame"] <= fmax)
    ].copy()

    if df.empty:
        return pd.DataFrame(columns=["frame", "rel_frame", "speed_kmh"])

    df = df.sort_values("frame").copy()

    df["dx"] = df["x"].diff()
    df["dy"] = df["y"].diff()
    df["dz"] = df["z"].diff()
    df["dframe"] = df["frame"].diff()
    df["delta_t"] = df["dframe"] * float(dt)

    df["dist_m"] = np.sqrt(df["dx"] ** 2 + df["dy"] ** 2 + df["dz"] ** 2)
    df["speed_mps"] = df["dist_m"] / df["delta_t"]
    df.loc[(df["delta_t"] <= 0) | (~np.isfinite(df["speed_mps"])), "speed_mps"] = np.nan

    df["speed_kmh"] = df["speed_mps"] * 3.6
    df["rel_frame"] = df["frame"] - int(center_frame)

    return df[["frame", "rel_frame", "speed_kmh"]].copy()


def plot_speed_all_actors_per_lead(
    series_rows: List[dict],
    out_pdf: Path,
    title: str,
):
    """
    series_rows: 同一 method+lead の集合
      row:
        method, lead_sec, rep, detect_frame, actor_id, speed_df
    全線を重ね描きする．
    """
    if not series_rows:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=300)

    for row in series_rows:
        sdf: pd.DataFrame = row["speed_df"]
        if sdf.empty:
            continue
        ax.plot(sdf["rel_frame"], sdf["speed_kmh"], linewidth=1.0, alpha=0.18)

    ax.axvline(0, linewidth=1.2, linestyle="--", alpha=0.85)
    ax.set_xlabel("検出フレームからの相対フレーム")
    ax.set_ylabel("速度 [km/h]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", direction="in")
    ax.set_axisbelow(True)
    ax.set_title(title)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_pdf}")


# ========= 危険領域への侵入判定 =========

def compute_predicted_risk_per_run(
    base_dir: Path,
    baseline: dict,
    frame_window: int,
    radius: float,
    min_consecutive_frames: int,
    min_actor_inside: int,
    frame_window_before: int = 0,
    use_switch_window: bool = False,
    window_from_switch_sec: float = 10.0,
    dt: float = 0.1,
) -> pd.DataFrame:
    """
    各 run (method, lead_sec, rep) について，
    事故危険領域への侵入フラグを計算する．

    追加出力:
      - segment_start_frame, segment_end_frame, detect_frame
        (risk_flag==1 のときだけ有効，無効なら -1)
      - entered_actor_ids_json
        リスク成立区間で危険領域内にいた actor_id の一覧（JSON文字列）
        (risk_flag==0 のときは "[]")
    """
    actor_files = find_actor_files(base_dir)

    base_frame = int(baseline["frame"])
    bx, by, bz = baseline["x"], baseline["y"], baseline["z"]

    if use_switch_window and window_from_switch_sec > 0.0:
        switch_window_frames = int(round(window_from_switch_sec / dt))
    else:
        switch_window_frames = None

    rows: List[dict] = []

    for (method, lead_sec, rep), path in sorted(actor_files.items()):
        df_pos = load_actor_csv_as_positions(path)

        logs_dir = path.parent
        if use_switch_window and switch_window_frames is not None:
            switch_frame = load_switch_frame_from_meta(logs_dir)
            if switch_frame is None:
                switch_frame = base_frame

            frame_min = max(0, switch_frame)
            frame_max = switch_frame + switch_window_frames
            window_mode = "switch"
        else:
            frame_min = max(0, base_frame - frame_window_before)
            frame_max = base_frame + frame_window
            window_mode = "baseline"

        df_win = df_pos[
            (df_pos["frame"] >= frame_min)
            & (df_pos["frame"] <= frame_max)
        ].copy()

        detect_frame = -1
        seg_start = -1
        seg_end = -1
        entered_actor_ids: List[int] = []

        if df_win.empty:
            risk_flag = 0
            n_points_window = 0
            n_points_inside = 0
            n_frames_window = 0
            n_frames_inside = 0
            max_consec = 0
            min_dist = np.nan
        else:
            diff_x = df_win["x"] - bx
            diff_y = df_win["y"] - by
            diff_z = df_win["z"] - bz
            dist = np.sqrt(diff_x * diff_x + diff_y * diff_y + diff_z * diff_z)
            df_win["dist_from_baseline"] = dist

            inside_point = df_win["dist_from_baseline"] <= radius

            n_points_window = int(len(df_win))
            n_points_inside = int(inside_point.sum())
            min_dist = float(df_win["dist_from_baseline"].min())

            df_frame = (
                df_win.assign(inside=inside_point)
                .groupby("frame", as_index=False)["inside"]
                .sum()
                .rename(columns={"inside": "n_inside"})
                .sort_values("frame")
            )

            flags_arr = (df_frame["n_inside"].to_numpy() >= int(min_actor_inside))
            frames_arr = df_frame["frame"].to_numpy(dtype=int)

            n_frames_window = int(len(df_frame))
            n_frames_inside = int(flags_arr.sum())
            max_consec = max_consecutive_true(flags_arr.tolist())

            risk_flag = int(max_consec >= int(min_consecutive_frames))

            if risk_flag == 1:
                s0, e0 = first_segment_where_flags_run(
                    frames=frames_arr,
                    flags=flags_arr,
                    min_consecutive=int(min_consecutive_frames),
                )
                if s0 is not None and e0 is not None:
                    seg_start = int(s0)
                    seg_end = int(e0)
                    detect_frame = int(e0)

                    df_seg = df_win[
                        (df_win["frame"] >= seg_start)
                        & (df_win["frame"] <= seg_end)
                    ].copy()

                    if not df_seg.empty:
                        inside_seg = (df_seg["dist_from_baseline"] <= radius)
                        entered_actor_ids = sorted(df_seg.loc[inside_seg, "actor_id"].astype(int).unique().tolist())

        rows.append(
            {
                "method": method,
                "lead_sec": lead_sec,
                "rep": rep,
                "baseline_frame": base_frame,
                "baseline_x": bx,
                "baseline_y": by,
                "baseline_z": bz,
                "window_mode": window_mode,
                "frame_min": frame_min,
                "frame_max": frame_max,
                "frame_window_before": frame_window_before,
                "frame_window": frame_window,
                "use_switch_window": use_switch_window,
                "window_from_switch_sec": window_from_switch_sec if use_switch_window else 0.0,
                "dt": dt,
                "radius": radius,
                "min_consecutive_frames": min_consecutive_frames,
                "min_actor_inside": min_actor_inside,
                "n_points_in_window": n_points_window,
                "n_points_inside_region": n_points_inside,
                "n_frames_in_window": n_frames_window,
                "n_frames_inside_region": n_frames_inside,
                "max_consecutive_inside_frames": max_consec,
                "risk_flag": risk_flag,
                "min_dist_from_baseline": min_dist,
                "segment_start_frame": int(seg_start),
                "segment_end_frame": int(seg_end),
                "detect_frame": int(detect_frame),
                "entered_actor_ids_json": json.dumps(entered_actor_ids, ensure_ascii=False),
                "n_entered_actors": int(len(entered_actor_ids)),
            }
        )

    return pd.DataFrame(rows)


def summarize_predicted_risk(per_run: pd.DataFrame) -> pd.DataFrame:
    grp = per_run.groupby(["method", "lead_sec"])

    summary = grp.agg(
        n_runs=("rep", "nunique"),
        risk_rate=("risk_flag", "mean"),
        avg_n_frames_in_window=("n_frames_in_window", "mean"),
        avg_n_frames_inside_region=("n_frames_inside_region", "mean"),
        avg_max_consecutive_inside_frames=("max_consecutive_inside_frames", "mean"),
        avg_min_dist_from_baseline=("min_dist_from_baseline", "mean"),
        avg_n_entered_actors=("n_entered_actors", "mean"),
    ).reset_index()

    return summary


# ========= CLI =========

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute risk-region invasion probability from actor.csv logs and optionally plot speeds."
    )
    p.add_argument(
        "--baseline-csv",
        type=Path,
        required=True,
        help="collisions_exp_accident.csv へのパス",
    )
    p.add_argument(
        "--base-frame",
        type=int,
        default=25411,
        help="基準とする事故フレーム番号",
    )
    p.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="results_grid_accident ディレクトリへのパス",
    )
    p.add_argument(
        "--frame-window",
        type=int,
        default=100,
        help="（baseline ベース）基準フレームから何フレーム先までを危険領域の時間範囲とするか",
    )
    p.add_argument(
        "--frame-window-before",
        type=int,
        default=0,
        help="（baseline ベース）基準フレームから何フレーム前までを危険領域の時間範囲とするか",
    )
    p.add_argument(
        "--radius",
        type=float,
        default=10.0,
        help="危険領域の半径 [m]",
    )
    p.add_argument(
        "--min-consecutive-frames",
        type=int,
        default=10,
        help="危険領域内に連続して存在するとみなす最小フレーム数",
    )
    p.add_argument(
        "--min-actor-inside",
        type=int,
        default=2,
        help="各フレームで危険領域内に存在するとみなす最小車両数（2台以上を想定）",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="1フレームあたりの秒数（例: 0.1）",
    )
    p.add_argument(
        "--use-switch-window",
        action="store_true",
        help=(
            "指定すると，baseline frame ではなく meta.json の "
            "switch_payload_frame_observed を基準にウィンドウを取る"
        ),
    )
    p.add_argument(
        "--window-from-switch-sec",
        type=float,
        default=10.0,
        help="--use-switch-window 有効時に，切り替えフレームから何秒先まで見るか",
    )
    p.add_argument(
        "--out-per-run",
        type=Path,
        default=Path("predicted_risk_per_run_3.csv"),
        help="run ごとの危険領域侵入フラグを出力するCSV",
    )
    p.add_argument(
        "--out-summary",
        type=Path,
        default=Path("predicted_risk_summary_3.csv"),
        help="method × lead_sec ごとの侵入率を出力するCSV",
    )

    # 速度プロット
    p.add_argument("--plot-speed", action="store_true", help="leadごとに速度[km/h]をプロットする")
    p.add_argument("--speed-window-frames", type=int, default=100, help="検出フレームの前後何フレームを描くか")
    p.add_argument("--speed-outdir", type=Path, default=Path("out_speed_by_lead_all"), help="速度図の出力ディレクトリ")
    p.add_argument(
        "--speed-save-csv",
        action="store_true",
        help="速度系列もCSVで保存する（leadごとにまとめて保存）",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()

    baseline = load_baseline_accident(
        path=args.baseline_csv,
        base_frame=args.base_frame,
        intensity_threshold=1000.0,
    )
    print(
        f"[INFO] Baseline accident: frame={baseline['frame']}, "
        f"x={baseline['x']:.2f}, y={baseline['y']:.2f}, z={baseline['z']:.2f}, "
        f"intensity={baseline['intensity']:.1f}"
    )

    per_run = compute_predicted_risk_per_run(
        base_dir=args.base_dir,
        baseline=baseline,
        frame_window=args.frame_window,
        radius=args.radius,
        min_consecutive_frames=args.min_consecutive_frames,
        min_actor_inside=args.min_actor_inside,
        frame_window_before=args.frame_window_before,
        use_switch_window=args.use_switch_window,
        window_from_switch_sec=args.window_from_switch_sec,
        dt=args.dt,
    )
    summary = summarize_predicted_risk(per_run)

    args.out_per_run.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    per_run.to_csv(args.out_per_run, index=False)
    summary.to_csv(args.out_summary, index=False)

    print(f"[OK] per-run predicted risk -> {args.out_per_run}")
    print(f"[OK] summary predicted risk -> {args.out_summary}")

    # ========= 速度プロット =========
    if args.plot_speed:
        args.speed_outdir.mkdir(parents=True, exist_ok=True)

        actor_files = find_actor_files(args.base_dir)

        speed_rows: List[dict] = []
        for _, r in per_run.iterrows():
            if int(r["risk_flag"]) != 1:
                continue

            method = str(r["method"])
            lead_sec = int(r["lead_sec"])
            rep = int(r["rep"])
            detect_frame = int(r.get("detect_frame", -1))
            if detect_frame < 0:
                continue

            ids_json = str(r.get("entered_actor_ids_json", "[]"))
            try:
                entered_ids = json.loads(ids_json)
                entered_ids = [int(x) for x in entered_ids]
            except Exception:
                entered_ids = []

            if not entered_ids:
                continue

            path = actor_files.get((method, lead_sec, rep))
            if path is None:
                continue

            df_pos_all = load_actor_csv_as_positions(path)

            for actor_id in entered_ids:
                sdf = compute_speed_series_kmh_for_actor(
                    df_pos_all=df_pos_all,
                    actor_id=int(actor_id),
                    center_frame=int(detect_frame),
                    dt=float(args.dt),
                    window_frames=int(args.speed_window_frames),
                )
                speed_rows.append(
                    {
                        "method": method,
                        "lead_sec": lead_sec,
                        "rep": rep,
                        "detect_frame": detect_frame,
                        "actor_id": int(actor_id),
                        "speed_df": sdf,
                    }
                )

        if not speed_rows:
            print("[WARN] plot-speed requested but no speed series were generated.")
            return

        methods = sorted(set([d["method"] for d in speed_rows]))
        for method in methods:
            leads = sorted(set([d["lead_sec"] for d in speed_rows if d["method"] == method]))
            for lead in leads:
                rows_ml = [d for d in speed_rows if d["method"] == method and d["lead_sec"] == lead]
                out_pdf = args.speed_outdir / f"speed_kmh_all_{method}_lead{lead:02d}.pdf"
                title = f"{method} lead={lead}s: speeds of all entered actors around detect_frame (±{args.speed_window_frames} frames)"
                plot_speed_all_actors_per_lead(rows_ml, out_pdf=out_pdf, title=title)

                if args.speed_save_csv:
                    # leadごとにまとめてCSV化
                    out_csv = args.speed_outdir / f"speed_kmh_all_{method}_lead{lead:02d}.csv"
                    flat_rows: List[dict] = []
                    for d in rows_ml:
                        sdf = d["speed_df"]
                        if sdf.empty:
                            continue
                        for _, rr in sdf.iterrows():
                            flat_rows.append(
                                {
                                    "method": d["method"],
                                    "lead_sec": d["lead_sec"],
                                    "rep": d["rep"],
                                    "detect_frame": d["detect_frame"],
                                    "actor_id": d["actor_id"],
                                    "frame": int(rr["frame"]),
                                    "rel_frame": int(rr["rel_frame"]),
                                    "speed_kmh": float(rr["speed_kmh"]) if np.isfinite(rr["speed_kmh"]) else np.nan,
                                }
                            )
                    pd.DataFrame(flat_rows).to_csv(out_csv, index=False)
                    print(f"[OK] {out_csv}")

        print(f"[OK] speed plots -> {args.speed_outdir}")


if __name__ == "__main__":
    main()
