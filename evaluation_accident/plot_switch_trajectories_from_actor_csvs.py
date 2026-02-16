#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D


# -------------------------------------------------------
# フォント設定
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
# ディレクトリ探索
# results_grid_accident/{method}/lead_*/rep_*/logs/actor.csv
# -------------------------------------------------------
LEAD_RE = re.compile(r"lead_(\d+)$")
REP_RE = re.compile(r"rep_(\d+)$")


@dataclass
class RunRef:
    method: str
    lead: int
    rep: int
    actor_csv: Path


def discover_actor_csvs(root: Path, methods: Optional[List[str]] = None) -> List[RunRef]:
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

                apath = rep_dir / "logs" / "actor.csv"
                if apath.exists():
                    runs.append(RunRef(method=method, lead=lead, rep=rep, actor_csv=apath))
    return runs


# -------------------------------------------------------
# 列名の自動推定（pred側用）
# -------------------------------------------------------
CANDIDATES: Dict[str, List[str]] = {
    "x": ["x", "location_x", "pos_x", "world_x"],
    "y": ["y", "location_y", "pos_y", "world_y"],
    "z": ["z", "location_z", "pos_z", "world_z"],
    "payload_frame": ["payload_frame", "payloadframe", "frame_payload"],
    "carla_frame": ["frame", "tick", "world_frame"],
    "time_sec": ["time_sec", "timestamp", "t_sec", "time"],
    "actor_id": ["actor_id", "id", "carla_actor_id"],
    "actor_type": ["actor_type", "type", "blueprint", "bp", "class"],
}


def pick_col(df: pd.DataFrame, key: str) -> Optional[str]:
    for c in CANDIDATES.get(key, []):
        if c in df.columns:
            return c
    return None


def require_cols(df: pd.DataFrame, keys: List[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    for k in keys:
        col = pick_col(df, k)
        if col is None:
            missing.append(k)
        else:
            mapping[k] = col
    if missing:
        raise SystemExit(
            f"Missing required columns for: {missing}\n"
            f"Available columns:\n  {', '.join(df.columns.astype(str))}\n"
            f"Edit CANDIDATES in this script to match your CSV."
        )
    return mapping


# -------------------------------------------------------
# 前処理
# -------------------------------------------------------
@dataclass
class Prepared:
    df: pd.DataFrame
    cols: Dict[str, str]
    time_col: str
    time_kind: str
    switch_pf: int
    csv_path: Path
    kind: str  # "pred" or "raw"


def choose_time_col_pred(df: pd.DataFrame, csv_path: Path) -> Tuple[str, str]:
    pf_col = pick_col(df, "payload_frame")
    if pf_col is not None:
        return pf_col, "payload_frame"
    cf_col = pick_col(df, "carla_frame")
    if cf_col is None:
        raise SystemExit(f"{csv_path} に payload_frame/carla_frame がありません")
    print(f"[WARN] {csv_path} payload_frameが無いので carla_frame を使用します（predは時間窓は切りません）")
    return cf_col, "carla_frame"


def choose_time_col_raw(df: pd.DataFrame, csv_path: Path) -> Tuple[str, str]:
    # rawは frame を最優先（事故フレーム基準）
    if "frame" in df.columns:
        return "frame", "frame"
    # フォールバック
    pf_col = pick_col(df, "payload_frame")
    if pf_col is not None:
        return pf_col, "payload_frame"
    cf_col = pick_col(df, "carla_frame")
    if cf_col is not None:
        return cf_col, "carla_frame"
    raise SystemExit(f"{csv_path} に frame/payload_frame/carla_frame がありません")


def prepare_df_pred(
    csv_path: Path,
    base_payload_frame: int,
    lead: int,
    dt: float,
    actor_type_prefix: str,
) -> Prepared:
    """
    pred側：時間窓で切らない（全件出力）
    """
    df = pd.read_csv(csv_path)
    cols = require_cols(df, ["x", "y", "actor_id"])
    time_col, time_kind = choose_time_col_pred(df, csv_path)

    at_col = pick_col(df, "actor_type")
    if at_col is not None and actor_type_prefix:
        df[at_col] = df[at_col].astype(str)
        df = df[df[at_col].str.lower().str.startswith(actor_type_prefix.lower())]

    df[cols["x"]] = pd.to_numeric(df[cols["x"]], errors="coerce")
    df[cols["y"]] = pd.to_numeric(df[cols["y"]], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[cols["actor_id"]] = pd.to_numeric(df[cols["actor_id"]], errors="coerce")
    df = df.dropna(subset=[cols["x"], cols["y"], time_col, cols["actor_id"]]).copy()

    frames_per_sec = int(round(1.0 / dt))
    switch_pf = base_payload_frame - lead * frames_per_sec

    return Prepared(df=df, cols=cols, time_col=time_col, time_kind=time_kind, switch_pf=switch_pf, csv_path=csv_path, kind="pred")


def prepare_df_raw(
    csv_path: Path,
    base_payload_frame: int,
    lead: int,
    dt: float,
    pre_frames: int,
    post_frames: int,
) -> Prepared:
    """
    raw側：
      - x,y,z を優先
      - time_col は frame を最優先
      - 抽出範囲は switch_pf を中心に [switch_pf-pre_frames, switch_pf+post_frames-1]
        ただし pre 側は switch_pf-1 まで，post 側は switch_pf から
    """
    df = pd.read_csv(csv_path)

    cols: Dict[str, str] = {}
    for k in ["x", "y", "z"]:
        if k in df.columns:
            cols[k] = k
    if "x" not in cols or "y" not in cols:
        tmp = require_cols(df, ["x", "y"])
        cols["x"] = tmp["x"]
        cols["y"] = tmp["y"]
    if "z" not in cols:
        zc = pick_col(df, "z")
        if zc is not None:
            cols["z"] = zc

    aid = pick_col(df, "actor_id")
    if aid is None:
        raise SystemExit(f"{csv_path} に actor_id/id/carla_actor_id がありません（CANDIDATES['actor_id'] を調整して）")
    cols["actor_id"] = aid

    time_col, time_kind = choose_time_col_raw(df, csv_path)

    df[cols["x"]] = pd.to_numeric(df[cols["x"]], errors="coerce")
    df[cols["y"]] = pd.to_numeric(df[cols["y"]], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df[cols["actor_id"]] = pd.to_numeric(df[cols["actor_id"]], errors="coerce")
    df = df.dropna(subset=[cols["x"], cols["y"], time_col, cols["actor_id"]]).copy()

    frames_per_sec = int(round(1.0 / dt))
    switch_pf = base_payload_frame - lead * frames_per_sec

    # rawはフレーム数で切る（要求）
    if not df.empty:
        start_f = switch_pf - int(pre_frames)
        end_f = switch_pf + int(post_frames) - 1  # inclusive
        df = df[(df[time_col] >= start_f) & (df[time_col] <= end_f)]

    return Prepared(df=df, cols=cols, time_col=time_col, time_kind=time_kind, switch_pf=switch_pf, csv_path=csv_path, kind="raw")


# -------------------------------------------------------
# 描画
# -------------------------------------------------------
def common_axes_style(ax):
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.tick_params(axis="both", direction="in")
    ax.set_axisbelow(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")


def add_inside_legend_dots(
    ax,
    label_pre: str,
    label_post: str,
    loc: str,
    label_accident: str = "事故地点",
    show_accident: bool = False,
    ):
    handles = [
        Line2D([0], [0], marker="o", linestyle="None", color="blue", label=label_pre),
        Line2D([0], [0], marker="o", linestyle="None", color="red", label=label_post),
    ]
    if show_accident:
        handles.append(Line2D([0], [0], marker="x", linestyle="None", color="black", label=label_accident))
    ax.legend(handles=handles, loc=loc, bbox_to_anchor=(0.6,0.9),frameon=True)


def choose_keep_ids(pred: Prepared, raw: Prepared, max_actors: int, mode: str) -> List[int]:
    """
    mode:
      - pred: pred側の出現点数上位（predは全区間）
      - raw:  raw側の出現点数上位（rawは切り出し後）
      - intersection: 共通IDのみで，pred側の出現点数上位
    """
    pid = pred.cols["actor_id"]
    rid = raw.cols["actor_id"]

    pred_ids = set(pred.df[pid].dropna().astype(int).unique().tolist())
    raw_ids = set(raw.df[rid].dropna().astype(int).unique().tolist())

    if mode == "pred":
        base_df = pred.df
        base_col = pid
        allowed = None
    elif mode == "raw":
        base_df = raw.df
        base_col = rid
        allowed = None
    elif mode == "intersection":
        allowed = pred_ids & raw_ids
        base_df = pred.df[pred.df[pid].astype(int).isin(allowed)].copy()
        base_col = pid
    else:
        raise SystemExit(f"Unknown keep-id-mode: {mode}")

    if base_df.empty:
        return []

    counts = base_df.groupby(base_col).size().sort_values(ascending=False)
    keep = [int(x) for x in counts.head(max_actors).index.to_list()]

    if allowed is not None:
        keep = [x for x in keep if x in allowed]

    return keep

def compute_xy_limits_from_raw(
    raw: Prepared,
    keep_ids: List[int],
    pad_ratio: float = 0.02,
    accident_xy: Optional[Tuple[float, float]] = None,
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    表示範囲は raw（切り出し後）から決める．
    """
    if raw.df.empty:
        raise SystemExit("raw is empty for compute_xy_limits_from_raw")

    id_col = raw.cols["actor_id"]
    df = raw.df[raw.df[id_col].astype(int).isin(keep_ids)]
    if df.empty:
        raise SystemExit("raw kept is empty for compute_xy_limits_from_raw")

    x = df[raw.cols["x"]]
    y = df[raw.cols["y"]]

    xmin = float(x.min())
    xmax = float(x.max())
    ymin = float(y.min())
    ymax = float(y.max())

    # 事故地点も表示範囲に含める
    if accident_xy is not None:
        ax0, ay0 = accident_xy
        xmin = min(xmin, float(ax0))
        xmax = max(xmax, float(ax0))
        ymin = min(ymin, float(ay0))
        ymax = max(ymax, float(ay0))
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)
    padx = dx * pad_ratio
    pady = dy * pad_ratio

    return (xmin - padx, xmax + padx), (ymin - pady, ymax + pady)


def plot_split(
    prepared: Prepared,
    out_pdf: Path,
    keep_ids: List[int],
    linewidth: float,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    legend_inside: bool,
    legend_loc: str,
    label_pre: str,
    label_post: str,
    accident_xy: Optional[Tuple[float, float]] = None,
    label_accident: str = "事故地点",
):
    df = prepared.df
    if df.empty:
        print(f"[WARN] 空データ: {prepared.csv_path}")
        return

    id_col = prepared.cols["actor_id"]
    df = df[df[id_col].astype(int).isin(keep_ids)].copy()
    if df.empty:
        print(f"[WARN] keep_ids適用後に空: {out_pdf.name}")
        return

    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    for aid, sub in df.groupby(id_col):
        sub = sub.sort_values(prepared.time_col)

        pre = sub[sub[prepared.time_col] < prepared.switch_pf]
        post = sub[sub[prepared.time_col] >= prepared.switch_pf]

        if len(pre) >= 2:
            ax.plot(pre[prepared.cols["x"]], pre[prepared.cols["y"]], linewidth=linewidth, color="blue")
        if len(post) >= 2:
            ax.plot(post[prepared.cols["x"]], post[prepared.cols["y"]], linewidth=linewidth, color="red")

    common_axes_style(ax)
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    if legend_inside:
         add_inside_legend_dots(
            ax,
            label_pre="同期区間",
            label_post="予測区間",
            loc=legend_loc,
            label_accident=label_accident,
            show_accident=(accident_xy is not None),
        )

    # 事故地点の点を打つ
    if accident_xy is not None:
        ax.scatter([accident_xy[0]], [accident_xy[1]], marker="x", s=90, color="black", linewidths=2.0, zorder=10)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] {out_pdf}")


# -------------------------------------------------------
# main
# -------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--root", type=str, default="results_grid_accident", help="results_grid_accident")
    ap.add_argument("--raw-csv", type=str, required=True, help="元データCSV（exp_accident.csv）を指定")
    ap.add_argument("--outdir", type=str, default="", help="出力先ディレクトリ（未指定なら自動生成）")

    ap.add_argument("--methods", type=str, nargs="*", default=None, help="autopilot lstm など．未指定は両方")
    ap.add_argument("--lead", type=int, default=-1, help="leadを絞る（-1で全て）")
    ap.add_argument("--rep", type=int, default=-1, help="repを絞る（-1で全て）")

    ap.add_argument("--base-payload-frame", type=int, default=25411, help="基準事故フレーム番号（rawのframeに入っている想定）")
    ap.add_argument("--dt", type=float, default=0.1, help="frame間隔[s]（switch_pf計算に使う．raw抽出はフレーム数指定）")

    # raw抽出フレーム数（要求どおり）
    ap.add_argument("--raw-pre-frames", type=int, default=600, help="raw: switch前に何フレーム出すか")
    ap.add_argument("--raw-post-frames", type=int, default=300, help="raw: switch後に何フレーム出すか")

    ap.add_argument("--actor-type-prefix", type=str, default="vehicle", help="pred側のactor_typeでフィルタ（rawには適用しない）")
    ap.add_argument("--max-actors", type=int, default=300, help="描画する最大アクタ数")
    ap.add_argument("--linewidth", type=float, default=1.2, help="線幅")

    ap.add_argument(
        "--keep-id-mode",
        type=str,
        default="pred",
        choices=["pred", "raw", "intersection"],
        help="描画対象actor_idの選び方",
    )

    ap.add_argument("--legend", action="store_true", help="凡例を図内に表示（ドットのみ）")
    ap.add_argument("--legend-loc", type=str, default="upper left", help="凡例の位置（matplotlib loc）")
    ap.add_argument("--label-pre", type=str, default="pre", help="凡例ラベル（切替前）")
    ap.add_argument("--label-post", type=str, default="post", help="凡例ラベル（切替後）")

    # 事故地点（X,Y）
    ap.add_argument("--accident-x", type=float, default=106.95402527, help="事故地点X")
    ap.add_argument("--accident-y", type=float, default=-39.21111488, help="事故地点Y")
    ap.add_argument("--label-accident", type=str, default="事故地点", help="事故地点の凡例ラベル")

    args = ap.parse_args()
    accident_xy = (float(args.accident_x), float(args.accident_y))
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise SystemExit(f"root not found: {root}")

    raw_csv = Path(args.raw_csv).expanduser().resolve()
    if not raw_csv.exists():
        raise SystemExit(f"raw csv not found: {raw_csv}")

    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
        outdir.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = root / f"out_switch_compare_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)

    runs = discover_actor_csvs(root, methods=args.methods)
    if args.lead >= 0:
        runs = [r for r in runs if r.lead == args.lead]
    if args.rep >= 0:
        runs = [r for r in runs if r.rep == args.rep]

    if not runs:
        raise SystemExit("No actor.csv found after filtering.")

    for run in runs:
        pred = prepare_df_pred(
            csv_path=run.actor_csv,
            base_payload_frame=args.base_payload_frame,
            lead=run.lead,
            dt=args.dt,
            actor_type_prefix=args.actor_type_prefix,
        )
        if pred.df.empty:
            print(f"[WARN] empty pred: {run.actor_csv}")
            continue

        raw = prepare_df_raw(
            csv_path=raw_csv,
            base_payload_frame=args.base_payload_frame,
            lead=run.lead,
            dt=args.dt,
            pre_frames=args.raw_pre_frames,
            post_frames=args.raw_post_frames,
        )
        if raw.df.empty:
            print(f"[WARN] empty raw after frame-window filtering for lead={run.lead}: {raw_csv}")
            continue

        keep_ids = choose_keep_ids(pred=pred, raw=raw, max_actors=args.max_actors, mode=args.keep_id_mode)
        if not keep_ids:
            print(f"[WARN] keep_ids empty (mode={args.keep_id_mode}) for lead={run.lead} rep={run.rep}")
            continue

        # 表示範囲は raw（切り出し後）で決める
        xlim, ylim = compute_xy_limits_from_raw(raw, keep_ids=keep_ids, pad_ratio=0.02)

        out_pred = outdir / f"traj_switch_{run.method}_lead{run.lead:02d}_rep{run.rep:02d}.pdf"
        out_raw = outdir / f"traj_switch_raw_lead{run.lead:02d}_rep{run.rep:02d}.pdf"

        plot_split(
            prepared=pred,
            out_pdf=out_pred,
            keep_ids=keep_ids,
            linewidth=args.linewidth,
            xlim=xlim,
            ylim=ylim,
            legend_inside=bool(args.legend),
            legend_loc=args.legend_loc,
            label_pre=args.label_pre,
            label_post=args.label_post,
            accident_xy=accident_xy,
            label_accident=args.label_accident,
        )
        plot_split(
            prepared=raw,
            out_pdf=out_raw,
            keep_ids=keep_ids,
            linewidth=args.linewidth,
            xlim=xlim,
            ylim=ylim,
            legend_inside=bool(args.legend),
            legend_loc=args.legend_loc,
            label_pre=args.label_pre,
            label_post=args.label_post,
            accident_xy=accident_xy,
            label_accident=args.label_accident,
        )

    print(f"[OK] 出力ディレクトリ: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
