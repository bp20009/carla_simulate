from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run autopilot vs LSTM grid sweeps with trimmed CSV sends."
    )
    parser.add_argument("--csv", type=Path, default=Path("send_data/exp_accident.csv"))
    parser.add_argument(
        "--replay-script",
        type=Path,
        required=True,
        help="Path to replay script (e.g., scripts/udp_replay/replay_from_udp_carla_pred.py)",
    )
    parser.add_argument(
        "--sender-script",
        type=Path,
        default=Path("send_data/send_udp_frames_from_csv.py"),
    )
    parser.add_argument("--outdir", type=Path, default=Path("results_future_grid"))

    parser.add_argument("--target-frame", type=int, default=25411)
    parser.add_argument("--fixed-delta", type=float, default=0.1)
    parser.add_argument("--post-sec", type=float, default=10.0)

    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--pre-sec-list", type=str, default="0.5,1,2,3,5,10")

    parser.add_argument("--carla-host", default="127.0.0.1")
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=5005)

    parser.add_argument("--tm-seed", type=int, default=20009)
    parser.add_argument("--sender-interval", type=float, default=None)
    parser.add_argument("--startup-delay", type=float, default=1.0)

    parser.add_argument("--lstm-model", type=Path, default=Path("traj_lstm.pt"))
    parser.add_argument("--lstm-device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser.parse_args()


def make_trimmed_csv(src: Path, dst: Path, *, end_frame_inclusive: int) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        with dst.open("w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
            writer.writeheader()
            for row in reader:
                frame_val = row.get("frame")
                if frame_val is None:
                    continue
                try:
                    frame_int = int(float(frame_val))
                except ValueError:
                    continue
                if frame_int <= end_frame_inclusive:
                    writer.writerow(row)


def has_accident_in_window(collision_csv: Path, start_frame: int, end_frame: int) -> bool:
    if not collision_csv.exists():
        return False
    with collision_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("is_accident") != "1":
                continue
            try:
                payload_frame = int(float(row["payload_frame"]))
            except Exception:
                continue
            if start_frame <= payload_frame <= end_frame:
                return True
    return False


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    pre_list = [float(x.strip()) for x in args.pre_sec_list.split(",") if x.strip()]
    sender_interval = (
        args.sender_interval if args.sender_interval is not None else args.fixed_delta
    )
    post_frames = int(round(args.post_sec / args.fixed_delta))
    eval_end_frame = args.target_frame + post_frames

    summary_rows: List[Tuple[float, str, int, int]] = []

    for pre_sec in pre_list:
        switch_frame = args.target_frame - int(round(pre_sec / args.fixed_delta))
        if switch_frame < 0:
            switch_frame = 0

        trimmed_csv = args.outdir / "trimmed" / f"exp_until_{switch_frame}.csv"
        if not trimmed_csv.exists():
            make_trimmed_csv(args.csv, trimmed_csv, end_frame_inclusive=switch_frame)

        for mode in ("autopilot", "lstm"):
            hits = 0
            for run_idx in range(1, args.runs + 1):
                run_id = f"pre_{pre_sec:g}s_{mode}_run_{run_idx:04d}"
                run_dir = args.outdir / run_id
                logs_dir = run_dir / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)

                collision_log = logs_dir / "pred_collisions.csv"
                actor_log = logs_dir / "actor.csv"
                meta_json = logs_dir / "meta.json"
                id_map = logs_dir / "id_map.csv"

                replay_cmd = [
                    sys.executable,
                    str(args.replay_script),
                    "--carla-host",
                    args.carla_host,
                    "--carla-port",
                    str(args.carla_port),
                    "--listen-host",
                    args.listen_host,
                    "--listen-port",
                    str(args.listen_port),
                    "--fixed-delta",
                    str(args.fixed_delta),
                    "--poll-interval",
                    str(args.fixed_delta),
                    "--switch-payload-frame",
                    str(switch_frame),
                    "--end-payload-frame",
                    str(eval_end_frame),
                    "--future-mode",
                    mode,
                    "--tm-seed",
                    str(args.tm_seed + run_idx),
                    "--collision-log",
                    str(collision_log),
                    "--actor-log",
                    str(actor_log),
                    "--metadata-output",
                    str(meta_json),
                    "--id-map-file",
                    str(id_map),
                    "--log-level",
                    args.log_level,
                ]
                if mode == "lstm":
                    replay_cmd += [
                        "--lstm-model",
                        str(args.lstm_model),
                        "--lstm-device",
                        args.lstm_device,
                    ]

                sender_cmd = [
                    sys.executable,
                    str(args.sender_script),
                    str(trimmed_csv),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(args.listen_port),
                    "--interval",
                    str(sender_interval),
                    "--frame-stride",
                    "1",
                    "--log-level",
                    args.log_level,
                ]

                replay_proc = subprocess.Popen(replay_cmd)
                sender_proc = None
                try:
                    time.sleep(max(args.startup_delay, 0.0))
                    sender_proc = subprocess.Popen(sender_cmd)

                    sender_proc.wait(timeout=120)
                    replay_proc.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    if sender_proc is not None and sender_proc.poll() is None:
                        sender_proc.terminate()
                        try:
                            sender_proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            sender_proc.kill()
                    if replay_proc.poll() is None:
                        replay_proc.terminate()
                        try:
                            replay_proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            replay_proc.kill()

                hit = has_accident_in_window(
                    collision_log, args.target_frame, eval_end_frame
                )
                hits += int(hit)
                print(f"[{run_id}] hit={hit}")

            summary_rows.append((pre_sec, mode, hits, args.runs))
            print(f"[SUMMARY] pre={pre_sec:g}s mode={mode} hits={hits}/{args.runs}")

    summary_path = args.outdir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pre_sec", "mode", "hits", "runs", "target_frame", "post_sec"])
        for pre_sec, mode, hits, runs in summary_rows:
            writer.writerow([pre_sec, mode, hits, runs, args.target_frame, args.post_sec])

    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
