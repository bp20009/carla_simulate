#!/usr/bin/env python3
"""
results_grid_accident 以下の logs/actor.csv を用いて，
予測された車両挙動が基準事故の危険領域に侵入した確率を method × lead_sec ごとに評価する．

追加:
  risk_flag==1 の run について，
  リスク成立の最初の連続区間 [segment_start, segment_end] を取り，
  その区間で危険領域内にいた actor_id を抽出する（entered_actor_ids）．

更に，
  detect_frame (=segment_end) を中心に，
  entered_actor_ids に含まれる「対象車両を全部」について，
    - 速度[km/h]の時系列（detect_frame±Nフレーム）
  を lead ごとに別PDFでプロットする．

更に，
  entered_actor_ids の全ペア（組合せ）について，
    - 車間距離[m]の時系列（detect_frame±Nフレーム）
  を lead ごとに別PDFでプロットする．

同一方向（同じベクトル）だけを考慮する拡張:
  --same-heading-only を指定すると，危険領域内の車両台数 n_inside を数える際に，
  速度ベクトルが代表ベクトルと同方向（cos類似度 >= 閾値）な車両のみをカウントする．

重要:
  actor.csv が疎で速度ベクトルが計算できないフレームが多いと，同一方向フィルタで
  n_inside が 0 になりやすい．このスクリプトではその対策として，
  代表ベクトルが作れない場合や通過台数が少なすぎる場合は，
  方向フィルタ無しの inside 台数へフォールバックする．
"""

from __future__ import annotations

import argparse
import itertools
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


# ========= 同一方向（ベクトル）フィルタのための速度推定 =========

def _add_velocity_columns(
    df_pos: pd.DataFrame,
    dt: float,
) -> pd.DataFrame:
    if df_pos.empty:
        out = df_pos.copy()
        out["vx"] = np.nan
        out["vy"] = np.nan
        out["vz"] = np.nan
        out["speed_mps"] = np.nan
        return out

    df = df_pos.sort_values(["actor_id", "frame"]).copy()
    g = df.groupby("actor_id", sort=False)

    dx_b = g["x"].diff()
    dy_b = g["y"].diff()
    dz_b = g["z"].diff()
    dt_b = g["frame"].diff() * float(dt)

    dx_f = g["x"].shift(-1) - df["x"]
    dy_f = g["y"].shift(-1) - df["y"]
    dz_f = g["z"].shift(-1) - df["z"]
    dt_f = (g["frame"].shift(-1) - df["frame"]) * float(dt)

    vx_b = dx_b / dt_b
    vy_b = dy_b / dt_b
    vz_b = dz_b / dt_b

    vx_f = dx_f / dt_f
    vy_f = dy_f / dt_f
    vz_f = dz_f / dt_f

    use_b = (dt_b > 0) & np.isfinite(vx_b) & np.isfinite(vy_b) & np.isfinite(vz_b)
    use_f = (dt_f > 0) & np.isfinite(vx_f) & np.isfinite(vy_f) & np.isfinite(vz_f)

    df["vx"] = np.where(use_b, vx_b, np.where(use_f, vx_f, np.nan))
    df["vy"] = np.where(use_b, vy_b, np.where(use_f, vy_f, np.nan))
    df["vz"] = np.where(use_b, vz_b, np.where(use_f, vz_f, np.nan))

    df["speed_mps"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2 + df["vz"] ** 2)
    df.loc[~np.isfinite(df["speed_mps"]), "speed_mps"] = np.nan
    return df


def _framewise_inside_counts_with_heading_filter(
    df_win: pd.DataFrame,
    inside_point: pd.Series,
    min_actor_inside: int,
    same_heading_only: bool,
    heading_cos_thresh: float,
    heading_min_speed_mps: float,
    heading_use_3d: bool,
) -> tuple[pd.DataFrame, dict[int, set[int]]]:
    if df_win.empty:
        return (
            pd.DataFrame(columns=["frame", "n_inside"]),
            {},
        )

    df_in = df_win.loc[inside_point].copy()
    if df_in.empty:
        frames = np.sort(df_win["frame"].unique())
        return (
            pd.DataFrame({"frame": frames.astype(int), "n_inside": np.zeros_like(frames, dtype=int)}),
            {},
        )

    pass_ids_by_frame: dict[int, set[int]] = {}

    if not same_heading_only:
        df_frame = (
            df_in.groupby("frame", as_index=False)["actor_id"]
            .nunique()
            .rename(columns={"actor_id": "n_inside"})
            .sort_values("frame")
        )
        df_frame["frame"] = df_frame["frame"].astype(int)
        df_frame["n_inside"] = df_frame["n_inside"].astype(int)
        return df_frame, pass_ids_by_frame

    frames_sorted = np.sort(df_win["frame"].unique()).astype(int)
    inside_by_frame = {int(fr): g.copy() for fr, g in df_in.groupby("frame", sort=False)}

    out_rows: List[dict] = []
    eps = 1e-9
    cos_th = float(heading_cos_thresh)
    min_v = float(heading_min_speed_mps)

    for fr in frames_sorted:
        gi = inside_by_frame.get(int(fr))
        if gi is None or gi.empty:
            out_rows.append({"frame": int(fr), "n_inside": 0})
            pass_ids_by_frame[int(fr)] = set()
            continue

        valid = gi.copy()
        valid = valid[np.isfinite(valid["speed_mps"]) & (valid["speed_mps"] >= min_v)].copy()

        if heading_use_3d:
            vx = valid["vx"].to_numpy()
            vy = valid["vy"].to_numpy()
            vz = valid["vz"].to_numpy()
        else:
            vx = valid["vx"].to_numpy()
            vy = valid["vy"].to_numpy()
            vz = np.zeros_like(vx)

        if valid.empty or len(vx) == 0:
            n_inside = int(gi["actor_id"].nunique())
            out_rows.append({"frame": int(fr), "n_inside": n_inside})
            pass_ids_by_frame[int(fr)] = set(int(x) for x in gi["actor_id"].unique().tolist())
            continue

        vnorm = np.sqrt(vx * vx + vy * vy + vz * vz)
        good = np.isfinite(vnorm) & (vnorm > eps)
        if not np.any(good):
            n_inside = int(gi["actor_id"].nunique())
            out_rows.append({"frame": int(fr), "n_inside": n_inside})
            pass_ids_by_frame[int(fr)] = set(int(x) for x in gi["actor_id"].unique().tolist())
            continue

        ux = vx[good] / vnorm[good]
        uy = vy[good] / vnorm[good]
        uz = vz[good] / vnorm[good]
        rep = np.array([np.mean(ux), np.mean(uy), np.mean(uz)], dtype=float)
        rep_norm = float(np.linalg.norm(rep))
        if not np.isfinite(rep_norm) or rep_norm <= eps:
            n_inside = int(gi["actor_id"].nunique())
            out_rows.append({"frame": int(fr), "n_inside": n_inside})
            pass_ids_by_frame[int(fr)] = set(int(x) for x in gi["actor_id"].unique().tolist())
            continue

        rep_u = rep / rep_norm

        if heading_use_3d:
            gvx = gi["vx"].to_numpy()
            gvy = gi["vy"].to_numpy()
            gvz = gi["vz"].to_numpy()
        else:
            gvx = gi["vx"].to_numpy()
            gvy = gi["vy"].to_numpy()
            gvz = np.zeros_like(gvx)

        gnorm = np.sqrt(gvx * gvx + gvy * gvy + gvz * gvz)
        gspeed = gi["speed_mps"].to_numpy()
        ggood = np.isfinite(gnorm) & (gnorm > eps) & np.isfinite(gspeed) & (gspeed >= min_v)

        cos = np.full(len(gi), np.nan, dtype=float)
        cos[ggood] = (gvx[ggood] / gnorm[ggood]) * rep_u[0] + (gvy[ggood] / gnorm[ggood]) * rep_u[1] + (gvz[ggood] / gnorm[ggood]) * rep_u[2]
        passed = np.isfinite(cos) & (cos >= cos_th)

        gi_pass = gi.loc[passed].copy()
        n_pass = int(gi_pass["actor_id"].nunique())

        if n_pass < int(min_actor_inside):
            n_inside = int(gi["actor_id"].nunique())
            pass_ids_by_frame[int(fr)] = set(int(x) for x in gi["actor_id"].unique().tolist())
        else:
            n_inside = n_pass
            pass_ids_by_frame[int(fr)] = set(int(x) for x in gi_pass["actor_id"].unique().tolist())

        out_rows.append({"frame": int(fr), "n_inside": int(n_inside)})

    df_frame = pd.DataFrame(out_rows).sort_values("frame").copy()
    df_frame["frame"] = df_frame["frame"].astype(int)
    df_frame["n_inside"] = df_frame["n_inside"].astype(int)
    return df_frame, pass_ids_by_frame


# ========= 速度と車間距離 =========

def compute_speed_series_kmh_for_actor(
    df_pos_all: pd.DataFrame,
    actor_id: int,
    center_frame: int,
    dt: float,
    window_frames: int,
) -> pd.DataFrame:
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


def compute_headway_distance_series(
    df_pos_all: pd.DataFrame,
    actor_id_1: int,
    actor_id_2: int,
    center_frame: int,
    window_frames: int,
    use_3d: bool = False,
) -> pd.DataFrame:
    fmin = int(center_frame - window_frames)
    fmax = int(center_frame + window_frames)

    a = df_pos_all[
        (df_pos_all["actor_id"] == int(actor_id_1))
        & (df_pos_all["frame"] >= fmin)
        & (df_pos_all["frame"] <= fmax)
    ][["frame", "x", "y", "z"]].copy()
    b = df_pos_all[
        (df_pos_all["actor_id"] == int(actor_id_2))
        & (df_pos_all["frame"] >= fmin)
        & (df_pos_all["frame"] <= fmax)
    ][["frame", "x", "y", "z"]].copy()

    if a.empty or b.empty:
        return pd.DataFrame(columns=["frame", "rel_frame", "headway_m"])

    a = a.rename(columns={"x": "x1", "y": "y1", "z": "z1"})
    b = b.rename(columns={"x": "x2", "y": "y2", "z": "z2"})

    m = pd.merge(a, b, on="frame", how="inner").sort_values("frame").copy()
    if m.empty:
        return pd.DataFrame(columns=["frame", "rel_frame", "headway_m"])

    dx = m["x1"] - m["x2"]
    dy = m["y1"] - m["y2"]
    if use_3d:
        dz = m["z1"] - m["z2"]
        m["headway_m"] = np.sqrt(dx * dx + dy * dy + dz * dz)
    else:
        m["headway_m"] = np.sqrt(dx * dx + dy * dy)

    m["rel_frame"] = m["frame"] - int(center_frame)
    return m[["frame", "rel_frame", "headway_m"]].copy()


# ========= プロット =========

def plot_speed_all_targets_per_lead(
    rows: List[dict],
    out_pdf: Path,
    title: str,
    show_legend: bool,
):
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=300)

    for row in rows:
        sdf: pd.DataFrame = row["speed_df"]
        if sdf.empty:
            continue
        lab = f"rep{row['rep']:02d} id={row['actor_id']}"
        ax.plot(sdf["rel_frame"], sdf["speed_kmh"], linewidth=1.4, alpha=0.75, label=lab)

    ax.axvline(0, linewidth=1.2, linestyle="--", alpha=0.85)
    ax.set_xlabel("検出フレームからの相対フレーム")
    ax.set_ylabel("速度 [km/h]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", direction="in")
    ax.set_axisbelow(True)
    ax.set_title(title)
    if show_legend:
        ax.legend(framealpha=1.0, facecolor="white", ncol=1)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_pdf}")


def plot_headway_all_pairs_per_lead(
    rows: List[dict],
    out_pdf: Path,
    title: str,
    show_legend: bool,
):
    if not rows:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=300)

    for row in rows:
        hdf: pd.DataFrame = row["headway_df"]
        if hdf.empty:
            continue
        lab = f"rep{row['rep']:02d} {row['actor_id_1']}-{row['actor_id_2']}"
        ax.plot(hdf["rel_frame"], hdf["headway_m"], linewidth=1.6, alpha=0.75, label=lab)

    ax.axvline(0, linewidth=1.2, linestyle="--", alpha=0.85)
    ax.set_xlabel("検出フレームからの相対フレーム")
    ax.set_ylabel("車間距離 [m]")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", direction="in")
    ax.set_axisbelow(True)
    ax.set_title(title)
    if show_legend:
        ax.legend(framealpha=1.0, facecolor="white", ncol=1)

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
    same_heading_only: bool = False,
    heading_cos_thresh: float = 0.8,
    heading_min_speed_mps: float = 2.0,
    heading_use_3d: bool = False,
) -> pd.DataFrame:
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

            inside_point = df_win["dist_from_baseline"] <= float(radius)

            n_points_window = int(len(df_win))
            n_points_inside = int(inside_point.sum())
            min_dist = float(df_win["dist_from_baseline"].min())

            if bool(same_heading_only):
                df_win = _add_velocity_columns(df_win, dt=float(dt))
            else:
                df_win["vx"] = np.nan
                df_win["vy"] = np.nan
                df_win["vz"] = np.nan
                df_win["speed_mps"] = np.nan

            df_frame, pass_ids_by_frame = _framewise_inside_counts_with_heading_filter(
                df_win=df_win,
                inside_point=inside_point,
                min_actor_inside=int(min_actor_inside),
                same_heading_only=bool(same_heading_only),
                heading_cos_thresh=float(heading_cos_thresh),
                heading_min_speed_mps=float(heading_min_speed_mps),
                heading_use_3d=bool(heading_use_3d),
            )

            flags_arr = (df_frame["n_inside"].to_numpy(dtype=int) >= int(min_actor_inside))
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

                    if bool(same_heading_only):
                        ids: set[int] = set()
                        for fr in range(int(seg_start), int(seg_end) + 1):
                            ids |= pass_ids_by_frame.get(int(fr), set())
                        entered_actor_ids = sorted(ids)
                    else:
                        df_seg = df_win[
                            (df_win["frame"] >= seg_start)
                            & (df_win["frame"] <= seg_end)
                        ].copy()
                        if not df_seg.empty:
                            inside_seg = (df_seg["dist_from_baseline"] <= float(radius))
                            entered_actor_ids = sorted(
                                df_seg.loc[inside_seg, "actor_id"].astype(int).unique().tolist()
                            )

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
                "same_heading_only": int(bool(same_heading_only)),
                "heading_cos_thresh": float(heading_cos_thresh),
                "heading_min_speed_mps": float(heading_min_speed_mps),
                "heading_use_3d": int(bool(heading_use_3d)),
                "n_points_in_window": int(n_points_window),
                "n_points_inside_region": int(n_points_inside),
                "n_frames_in_window": int(n_frames_window),
                "n_frames_inside_region": int(n_frames_inside),
                "max_consecutive_inside_frames": int(max_consec),
                "risk_flag": int(risk_flag),
                "min_dist_from_baseline": float(min_dist) if np.isfinite(min_dist) else np.nan,
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
        description="Compute risk-region invasion probability and plot speed/headway for ALL target actors."
    )
    p.add_argument("--baseline-csv", type=Path, required=True, help="collisions_exp_accident.csv へのパス")
    p.add_argument("--base-frame", type=int, default=25411, help="基準とする事故フレーム番号")
    p.add_argument("--base-dir", type=Path, required=True, help="results_grid_accident ディレクトリへのパス")

    p.add_argument("--frame-window", type=int, default=100, help="基準フレームから先のウィンドウ長[frames]")
    p.add_argument("--frame-window-before", type=int, default=0, help="基準フレームから前のウィンドウ長[frames]")
    p.add_argument("--radius", type=float, default=10.0, help="危険領域の半径 [m]")
    p.add_argument("--min-consecutive-frames", type=int, default=10, help="連続成立とみなす最小フレーム数")
    p.add_argument("--min-actor-inside", type=int, default=2, help="危険領域内とみなす最小車両数（2台以上想定）")
    p.add_argument("--dt", type=float, default=0.1, help="1フレームあたりの秒数（例: 0.1）")

    p.add_argument("--use-switch-window", action="store_true", help="meta.json の switch_payload_frame_observed を基準にする")
    p.add_argument("--window-from-switch-sec", type=float, default=10.0, help="--use-switch-window 時の先読み秒数")

    # same-heading options
    p.add_argument("--same-heading-only", action="store_true", help="危険領域内台数を同一方向（cos類似度）でフィルタして数える")
    p.add_argument("--heading-cos-thresh", type=float, default=0.8, help="同一方向とみなすcos類似度の下限（例: 0.2〜0.9）")
    p.add_argument("--heading-min-speed-mps", type=float, default=2.0, help="方向判定に使う最小速度[m/s]（停車ノイズ除外）")
    p.add_argument("--heading-use-3d", action="store_true", help="方向判定を3D（XYZ）で行う（デフォルトはXY）")

    p.add_argument("--out-per-run", type=Path, default=Path("predicted_risk_per_run_4.csv"))
    p.add_argument("--out-summary", type=Path, default=Path("predicted_risk_summary_4.csv"))

    # 対象車両（全員）と全ペアのプロット
    p.add_argument("--plot-targets", action="store_true", help="risk_flag==1 の entered_actor_ids を全員プロットする")
    p.add_argument("--pair-window-frames", type=int, default=100, help="detect_frame の前後何フレームを描くか")
    p.add_argument("--pair-outdir", type=Path, default=Path("out_targets_speed_headway"), help="出力ディレクトリ")
    p.add_argument("--headway-use-3d", action="store_true", help="車間距離を3D（XYZ）で算出する（デフォルト2D）")
    p.add_argument("--targets-max-actors", type=int, default=12, help="1 run あたりの対象車両数の上限（多すぎる場合は近い順で切る）")
    p.add_argument("--pairs-max", type=int, default=60, help="1 run あたりのペア数上限（多すぎる場合は先頭から切る）")
    p.add_argument("--show-legend", action="store_true", help="凡例を表示する（多いと潰れる）")
    p.add_argument("--targets-save-csv", action="store_true", help="対象速度/車間距離もCSVで保存する")

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
        same_heading_only=bool(args.same_heading_only),
        heading_cos_thresh=float(args.heading_cos_thresh),
        heading_min_speed_mps=float(args.heading_min_speed_mps),
        heading_use_3d=bool(args.heading_use_3d),
    )
    summary = summarize_predicted_risk(per_run)

    args.out_per_run.parent.mkdir(parents=True, exist_ok=True)
    args.out_summary.parent.mkdir(parents=True, exist_ok=True)

    per_run.to_csv(args.out_per_run, index=False)
    summary.to_csv(args.out_summary, index=False)

    print(f"[OK] per-run predicted risk -> {args.out_per_run}")
    print(f"[OK] summary predicted risk -> {args.out_summary}")

    # ========= 対象車両（全員）と全ペアの速度/車間距離 =========
    if args.plot_targets:
        args.pair_outdir.mkdir(parents=True, exist_ok=True)

        actor_files = find_actor_files(args.base_dir)
        baseline_xyz = (baseline["x"], baseline["y"], baseline["z"])
        bx, by, bz = baseline_xyz

        speed_rows: List[dict] = []
        headway_rows: List[dict] = []

        for _, r in per_run.iterrows():
            if int(r.get("risk_flag", 0)) != 1:
                continue

            method = str(r["method"])
            lead_sec = int(r["lead_sec"])
            rep = int(r["rep"])
            detect_frame = int(r.get("detect_frame", -1))
            if detect_frame < 0:
                continue

            path = actor_files.get((method, lead_sec, rep))
            if path is None:
                continue

            df_pos_all = load_actor_csv_as_positions(path)

            # entered_actor_ids をそのまま対象にする
            ids_json = str(r.get("entered_actor_ids_json", "[]"))
            try:
                target_ids = [int(x) for x in json.loads(ids_json)]
            except Exception:
                target_ids = []

            if len(target_ids) < 2:
                # 2台未満なら，detect_frame 時点で半径内にいる車両を拾う（保険）
                df_fr = df_pos_all[df_pos_all["frame"] == detect_frame].copy()
                if not df_fr.empty:
                    dx = df_fr["x"] - bx
                    dy = df_fr["y"] - by
                    dz = df_fr["z"] - bz
                    df_fr["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
                    df_in = df_fr[df_fr["dist"] <= float(args.radius)].sort_values("dist")
                    target_ids = [int(x) for x in df_in["actor_id"].astype(int).unique().tolist()]

            if len(target_ids) == 0:
                continue

            # 多すぎる場合は「detect_frame の基準点に近い順」で上位のみ採用
            if len(target_ids) > int(args.targets_max_actors):
                df_fr = df_pos_all[df_pos_all["frame"] == detect_frame].copy()
                df_fr = df_fr[df_fr["actor_id"].astype(int).isin(set(target_ids))].copy()
                if not df_fr.empty:
                    dx = df_fr["x"] - bx
                    dy = df_fr["y"] - by
                    dz = df_fr["z"] - bz
                    df_fr["dist"] = np.sqrt(dx * dx + dy * dy + dz * dz)
                    df_fr = df_fr.sort_values("dist")
                    target_ids = [int(x) for x in df_fr["actor_id"].astype(int).tolist()[: int(args.targets_max_actors)]]
                else:
                    target_ids = target_ids[: int(args.targets_max_actors)]

            # ---- 速度: 対象車両“全員” ----
            for aid in target_ids:
                s = compute_speed_series_kmh_for_actor(
                    df_pos_all=df_pos_all,
                    actor_id=int(aid),
                    center_frame=int(detect_frame),
                    dt=float(args.dt),
                    window_frames=int(args.pair_window_frames),
                )
                speed_rows.append(
                    {
                        "method": method,
                        "lead_sec": lead_sec,
                        "rep": rep,
                        "detect_frame": detect_frame,
                        "actor_id": int(aid),
                        "speed_df": s,
                    }
                )

            # ---- 車間距離: 対象車両の全ペア ----
            pairs = list(itertools.combinations(sorted(set(target_ids)), 2))
            if len(pairs) > int(args.pairs_max):
                pairs = pairs[: int(args.pairs_max)]

            for a1, a2 in pairs:
                h = compute_headway_distance_series(
                    df_pos_all=df_pos_all,
                    actor_id_1=int(a1),
                    actor_id_2=int(a2),
                    center_frame=int(detect_frame),
                    window_frames=int(args.pair_window_frames),
                    use_3d=bool(args.headway_use_3d),
                )
                headway_rows.append(
                    {
                        "method": method,
                        "lead_sec": lead_sec,
                        "rep": rep,
                        "detect_frame": detect_frame,
                        "actor_id_1": int(a1),
                        "actor_id_2": int(a2),
                        "headway_df": h,
                    }
                )

        if not speed_rows and not headway_rows:
            print("[WARN] plot-targets requested but no valid series were generated.")
            return

        methods = sorted(set([d["method"] for d in (speed_rows if speed_rows else headway_rows)]))
        for method in methods:
            leads = sorted(set([d["lead_sec"] for d in (speed_rows if speed_rows else headway_rows) if d["method"] == method]))
            for lead in leads:
                # 速度（対象全員）
                rows_s = [d for d in speed_rows if d["method"] == method and d["lead_sec"] == lead]
                if rows_s:
                    out_pdf_s = args.pair_outdir / f"targets_speed_kmh_{method}_lead{lead:02d}.pdf"
                    title_s = f"{method} lead={lead}s: ALL target actors speed around detect_frame (±{args.pair_window_frames} frames)"
                    plot_speed_all_targets_per_lead(rows_s, out_pdf=out_pdf_s, title=title_s, show_legend=bool(args.show_legend))

                # 車間距離（対象全ペア）
                rows_h = [d for d in headway_rows if d["method"] == method and d["lead_sec"] == lead]
                if rows_h:
                    out_pdf_h = args.pair_outdir / f"targets_headway_m_{method}_lead{lead:02d}.pdf"
                    tag3d = "3D" if args.headway_use_3d else "2D"
                    title_h = f"{method} lead={lead}s: ALL target pairs headway ({tag3d}) around detect_frame (±{args.pair_window_frames} frames)"
                    plot_headway_all_pairs_per_lead(rows_h, out_pdf=out_pdf_h, title=title_h, show_legend=bool(args.show_legend))

                if args.targets_save_csv:
                    if rows_s:
                        out_csv_s = args.pair_outdir / f"targets_speed_kmh_{method}_lead{lead:02d}.csv"
                        flat_s: List[dict] = []
                        for d in rows_s:
                            sdf = d["speed_df"]
                            if sdf.empty:
                                continue
                            for _, rr in sdf.iterrows():
                                flat_s.append(
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
                        pd.DataFrame(flat_s).to_csv(out_csv_s, index=False)
                        print(f"[OK] {out_csv_s}")

                    if rows_h:
                        out_csv_h = args.pair_outdir / f"targets_headway_m_{method}_lead{lead:02d}.csv"
                        flat_h: List[dict] = []
                        for d in rows_h:
                            hdf = d["headway_df"]
                            if hdf.empty:
                                continue
                            for _, rr in hdf.iterrows():
                                flat_h.append(
                                    {
                                        "method": d["method"],
                                        "lead_sec": d["lead_sec"],
                                        "rep": d["rep"],
                                        "detect_frame": d["detect_frame"],
                                        "actor_id_1": d["actor_id_1"],
                                        "actor_id_2": d["actor_id_2"],
                                        "frame": int(rr["frame"]),
                                        "rel_frame": int(rr["rel_frame"]),
                                        "headway_m": float(rr["headway_m"]) if np.isfinite(rr["headway_m"]) else np.nan,
                                    }
                                )
                        pd.DataFrame(flat_h).to_csv(out_csv_h, index=False)
                        print(f"[OK] {out_csv_h}")

        print(f"[OK] targets plots -> {args.pair_outdir}")


if __name__ == "__main__":
    main()
