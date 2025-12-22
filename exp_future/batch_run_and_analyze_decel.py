"""Batch-run UDP replay experiments and analyze deceleration after switch."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

LOGGER = logging.getLogger(__name__)


@dataclass
class ActorDecelResult:
    object_id: str
    carla_actor_id: str
    carla_type: str
    min_accel: Optional[float]
    min_accel_frame: Optional[int]


@dataclass
class RunSummary:
    run_id: str
    switch_frame: Optional[int]
    window_sec: float
    fixed_delta_seconds: Optional[float]
    min_accel: Optional[float]
    min_accel_actor: Optional[str]
    min_accel_switch: Optional[float]
    min_accel_switch_actor: Optional[str]
    hard_brake_switch_count: int
    switch_eval_ticks: int
    hard_brake_threshold: float


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of runs to execute (default: 100)",
    )
    parser.add_argument(
        "--window-sec",
        type=float,
        default=5.0,
        help="Seconds after switch to compute min deceleration (default: 5.0)",
    )
    parser.add_argument(
        "--switch-eval-ticks",
        type=int,
        default=2,
        help="Evaluate deceleration using the first N ticks after switch (default: 2)",
    )
    parser.add_argument(
        "--hard-brake-threshold",
        type=float,
        default=-5.0,
        help="Threshold (m/s^2) to count hard braking right after switch (default: -5.0)",
    )
    parser.add_argument(
        "--tracking-sec",
        type=float,
        default=30.0,
        help="Tracking phase duration before switch (default: 30.0)",
    )
    parser.add_argument(
        "--future-sec",
        type=float,
        default=10.0,
        help="Future/autopilot duration after switch (default: 10.0)",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=0.05,
        help="Fixed delta seconds for replay (default: 0.05)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Seconds to wait for UDP data before ticking CARLA (default: 0.05)",
    )
    parser.add_argument(
        "--sender-interval",
        type=float,
        default=None,
        help="Override sender interval in seconds (default: fixed-delta)",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Send every Nth frame via UDP (default: 1)",
    )
    parser.add_argument(
        "--tm-seed",
        type=int,
        default=None,
        help="Optional Traffic Manager random seed for replay",
    )
    parser.add_argument(
        "--replay-script",
        type=Path,
        default=Path("scripts/udp_replay/replay_from_udp_carla_pred.py"),
        help="Path to replay script (default: scripts/udp_replay/replay_from_udp_carla_pred.py)",
    )
    parser.add_argument(
        "--sender-script",
        type=Path,
        default=Path("send_data/send_udp_frames_from_csv.py"),
        help="Path to UDP sender script (default: send_data/send_udp_frames_from_csv.py)",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        required=True,
        help="Path to the reduced CSV used by the sender",
    )
    parser.add_argument(
        "--carla-host",
        default="127.0.0.1",
        help="CARLA host for replay (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--carla-port",
        type=int,
        default=2000,
        help="CARLA port for replay (default: 2000)",
    )
    parser.add_argument(
        "--listen-host",
        default="0.0.0.0",
        help="Host for replay UDP listener (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--listen-port",
        type=int,
        default=5005,
        help="Port for replay UDP listener (default: 5005)",
    )
    parser.add_argument(
        "--sender-host",
        default="127.0.0.1",
        help="Host for UDP sender destination (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--sender-port",
        type=int,
        default=5005,
        help="Port for UDP sender destination (default: 5005)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity for this script (default: INFO)",
    )
    parser.add_argument(
        "--replay-log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity for replay script (default: INFO)",
    )
    parser.add_argument(
        "--sender-log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity for sender script (default: INFO)",
    )
    parser.add_argument(
        "--startup-delay",
        type=float,
        default=1.0,
        help="Seconds to wait before starting sender (default: 1.0)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results"),
        help="Output directory for run logs and summaries (default: results)",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def _run_command(command: Sequence[str]) -> None:
    LOGGER.debug("Running command: %s", " ".join(command))
    subprocess.run(command, check=True)


def _start_process(command: Sequence[str]) -> subprocess.Popen:
    LOGGER.debug("Starting process: %s", " ".join(command))
    return subprocess.Popen(command, text=True)


def _compute_switch_payload_frame(tracking_sec: float, fixed_delta: float) -> Optional[int]:
    if tracking_sec <= 0 or fixed_delta <= 0:
        return None
    return int(round(tracking_sec / fixed_delta))


def _load_metadata(metadata_path: Path) -> Dict[str, object]:
    with metadata_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_actor_log(actor_log_path: Path) -> Dict[str, List[Tuple[int, float, float, float, str, str]]]:
    records: Dict[str, List[Tuple[int, float, float, float, str, str]]] = {}
    with actor_log_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            carla_frame = row.get("carla_frame")
            object_id = row.get("id")
            if not carla_frame or not object_id:
                continue
            try:
                frame_id = int(carla_frame)
                x = float(row.get("location_x", ""))
                y = float(row.get("location_y", ""))
                z = float(row.get("location_z", ""))
            except (TypeError, ValueError):
                continue
            carla_actor_id = row.get("carla_actor_id") or ""
            carla_type = row.get("type") or ""
            records.setdefault(object_id, []).append(
                (frame_id, x, y, z, carla_actor_id, carla_type)
            )
    return records


def _analyze_decel_at_switch(
    points: List[Tuple[int, float, float, float, str, str]],
    *,
    switch_frame: int,
    fixed_delta: float,
    eval_ticks: int,
) -> Optional[float]:
    """Return acceleration (m/s^2) right after switch using eval_ticks."""
    if eval_ticks <= 0:
        return None

    points.sort(key=lambda item: item[0])

    idx0 = None
    for i, (frame_id, *_rest) in enumerate(points):
        if frame_id >= switch_frame:
            idx0 = i
            break
    if idx0 is None:
        return None

    need_points = eval_ticks + 1
    if idx0 + need_points > len(points):
        return None

    seg_speeds: List[Tuple[float, float]] = []
    for k in range(eval_ticks):
        f1, x1, y1, z1, *_ = points[idx0 + k]
        f2, x2, y2, z2, *_ = points[idx0 + k + 1]
        frame_delta = f2 - f1
        if frame_delta <= 0:
            return None
        dt = frame_delta * fixed_delta
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        speed = math.sqrt(dx * dx + dy * dy + dz * dz) / dt
        seg_speeds.append((speed, dt))

    if len(seg_speeds) < 2:
        return None

    v1, _dt1 = seg_speeds[0]
    v2, dt2 = seg_speeds[1]
    if dt2 <= 0:
        return None
    return (v2 - v1) / dt2


def _analyze_deceleration(
    actor_log_path: Path,
    metadata_path: Path,
    window_sec: float,
    *,
    switch_eval_ticks: int,
    hard_brake_threshold: float,
) -> Tuple[List[ActorDecelResult], RunSummary]:
    metadata = _load_metadata(metadata_path)
    fixed_delta_seconds = metadata.get("fixed_delta_seconds")
    switch_frame = metadata.get("switch_frame")

    fixed_delta = float(fixed_delta_seconds) if fixed_delta_seconds else None
    switch_frame_int = int(switch_frame) if switch_frame is not None else None

    results: List[ActorDecelResult] = []
    min_accel = None
    min_accel_actor = None
    min_accel_switch = None
    min_accel_switch_actor = None
    hard_brake_switch_count = 0

    if fixed_delta is None or fixed_delta <= 0 or switch_frame_int is None:
        summary = RunSummary(
            run_id=metadata_path.parent.parent.name,
            switch_frame=switch_frame_int,
            window_sec=window_sec,
            fixed_delta_seconds=fixed_delta,
            min_accel=None,
            min_accel_actor=None,
            min_accel_switch=None,
            min_accel_switch_actor=None,
            hard_brake_switch_count=0,
            switch_eval_ticks=switch_eval_ticks,
            hard_brake_threshold=hard_brake_threshold,
        )
        return results, summary

    window_frames = int(round(window_sec / fixed_delta))
    window_end_frame = switch_frame_int + max(window_frames, 0)

    records = _parse_actor_log(actor_log_path)
    for object_id, points in records.items():
        points.sort(key=lambda item: item[0])
        prev_frame: Optional[int] = None
        prev_point: Optional[Tuple[float, float, float]] = None
        prev_speed: Optional[float] = None
        actor_min_accel: Optional[float] = None
        actor_min_frame: Optional[int] = None
        carla_actor_id = points[0][4]
        carla_type = points[0][5]

        for frame_id, x, y, z, _, _ in points:
            if prev_frame is None or prev_point is None:
                prev_frame = frame_id
                prev_point = (x, y, z)
                continue

            frame_delta = frame_id - prev_frame
            if frame_delta <= 0:
                prev_frame = frame_id
                prev_point = (x, y, z)
                continue

            dt = frame_delta * fixed_delta
            dx = x - prev_point[0]
            dy = y - prev_point[1]
            dz = z - prev_point[2]
            speed = math.sqrt(dx * dx + dy * dy + dz * dz) / dt

            if prev_speed is not None:
                accel = (speed - prev_speed) / dt
                if switch_frame_int < frame_id <= window_end_frame:
                    if actor_min_accel is None or accel < actor_min_accel:
                        actor_min_accel = accel
                        actor_min_frame = frame_id
                        if min_accel is None or accel < min_accel:
                            min_accel = accel
                            min_accel_actor = object_id

            prev_speed = speed
            prev_frame = frame_id
            prev_point = (x, y, z)

        results.append(
            ActorDecelResult(
                object_id=object_id,
                carla_actor_id=carla_actor_id,
                carla_type=carla_type,
                min_accel=actor_min_accel,
                min_accel_frame=actor_min_frame,
            )
        )
        a_switch = _analyze_decel_at_switch(
            points,
            switch_frame=switch_frame_int,
            fixed_delta=fixed_delta,
            eval_ticks=switch_eval_ticks,
        )
        if a_switch is not None:
            if min_accel_switch is None or a_switch < min_accel_switch:
                min_accel_switch = a_switch
                min_accel_switch_actor = object_id
            if a_switch <= hard_brake_threshold:
                hard_brake_switch_count += 1

    summary = RunSummary(
        run_id=metadata_path.parent.parent.name,
        switch_frame=switch_frame_int,
        window_sec=window_sec,
        fixed_delta_seconds=fixed_delta,
        min_accel=min_accel,
        min_accel_actor=min_accel_actor,
        min_accel_switch=min_accel_switch,
        min_accel_switch_actor=min_accel_switch_actor,
        hard_brake_switch_count=hard_brake_switch_count,
        switch_eval_ticks=switch_eval_ticks,
        hard_brake_threshold=hard_brake_threshold,
    )
    return results, summary


def _write_actor_results(
    output_path: Path,
    run_id: str,
    switch_frame: Optional[int],
    window_frames: Optional[int],
    results: Sequence[ActorDecelResult],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "run_id",
                "object_id",
                "carla_actor_id",
                "carla_type",
                "switch_frame",
                "window_start_frame",
                "window_end_frame",
                "min_accel",
                "min_accel_frame",
            ]
        )
        window_start = switch_frame + 1 if switch_frame is not None else None
        window_end = (
            switch_frame + window_frames
            if switch_frame is not None and window_frames is not None
            else None
        )
        for result in results:
            writer.writerow(
                [
                    run_id,
                    result.object_id,
                    result.carla_actor_id,
                    result.carla_type,
                    switch_frame,
                    window_start,
                    window_end,
                    result.min_accel,
                    result.min_accel_frame,
                ]
            )


def _write_summary(output_path: Path, summaries: Sequence[RunSummary]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "run_id",
                "switch_frame",
                "window_sec",
                "fixed_delta_seconds",
                "min_accel",
                "min_accel_actor",
                "switch_eval_ticks",
                "hard_brake_threshold",
                "min_accel_switch",
                "min_accel_switch_actor",
                "hard_brake_switch_count",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.run_id,
                    summary.switch_frame,
                    summary.window_sec,
                    summary.fixed_delta_seconds,
                    summary.min_accel,
                    summary.min_accel_actor,
                    summary.switch_eval_ticks,
                    summary.hard_brake_threshold,
                    summary.min_accel_switch,
                    summary.min_accel_switch_actor,
                    summary.hard_brake_switch_count,
                ]
            )


def run_experiment(args: argparse.Namespace) -> None:
    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    sender_interval = args.sender_interval
    if sender_interval is None:
        sender_interval = args.fixed_delta

    switch_payload_frame = _compute_switch_payload_frame(
        args.tracking_sec, args.fixed_delta
    )
    max_runtime = max(args.tracking_sec + args.future_sec, 0.0)

    results_dir = args.outdir
    summaries: List[RunSummary] = []

    for run_index in range(1, args.runs + 1):
        run_id = f"run_{run_index:04d}"
        run_dir = results_dir / run_id
        logs_dir = run_dir / "logs"
        actor_log_path = logs_dir / "actor.csv"
        metadata_path = logs_dir / "meta.json"
        id_map_path = logs_dir / "id_map.csv"

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
            "--poll-interval",
            str(args.poll_interval),
            "--fixed-delta",
            str(args.fixed_delta),
            "--metadata-output",
            str(metadata_path),
            "--actor-log",
            str(actor_log_path),
            "--id-map-file",
            str(id_map_path),
            "--log-level",
            args.replay_log_level,
        ]

        if switch_payload_frame is not None:
            replay_cmd.extend([
                "--switch-payload-frame",
                str(switch_payload_frame),
            ])

        if args.tm_seed is not None:
            replay_cmd.extend(["--tm-seed", str(args.tm_seed)])

        if max_runtime > 0:
            replay_cmd.extend(["--max-runtime", str(max_runtime)])

        sender_cmd = [
            sys.executable,
            str(args.sender_script),
            str(args.csv_path),
            "--host",
            args.sender_host,
            "--port",
            str(args.sender_port),
            "--interval",
            str(sender_interval),
            "--frame-stride",
            str(args.frame_stride),
            "--log-level",
            args.sender_log_level,
        ]

        logs_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info("Starting run %s", run_id)
        replay_proc = _start_process(replay_cmd)
        try:
            time.sleep(max(args.startup_delay, 0.0))
            _run_command(sender_cmd)
            replay_proc.wait(timeout=max_runtime + 30.0 if max_runtime > 0 else None)
        finally:
            if replay_proc.poll() is None:
                replay_proc.terminate()
                replay_proc.wait(timeout=10)

        actor_results, summary = _analyze_deceleration(
            actor_log_path,
            metadata_path,
            args.window_sec,
            switch_eval_ticks=args.switch_eval_ticks,
            hard_brake_threshold=args.hard_brake_threshold,
        )
        fixed_delta = summary.fixed_delta_seconds
        window_frames = (
            int(round(args.window_sec / fixed_delta))
            if fixed_delta is not None and fixed_delta > 0
            else None
        )
        _write_actor_results(
            run_dir / "decel_after_switch_actors.csv",
            run_id,
            summary.switch_frame,
            window_frames,
            actor_results,
        )
        summaries.append(summary)

    _write_summary(results_dir / "summary_runs.csv", summaries)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))
    run_experiment(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
