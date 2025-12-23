#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Row:
    method: str
    lead_sec: int
    rep: int
    seed: int
    switch_payload_frame: int
    ran_ok: int
    accident_after_switch: int
    first_accident_payload_frame: Optional[int]
    status: str


def run_cmd(cmd: Sequence[str]) -> None:
    subprocess.run(cmd, check=True)


def start_proc(cmd: Sequence[str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, text=True)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def first_accident_payload_frame(meta: Dict[str, Any]) -> Optional[int]:
    acc = meta.get("accidents")
    if not isinstance(acc, list) or not acc:
        return None
    pf = acc[0].get("payload_frame")
    return int(pf) if pf is not None else None


def any_accident_after(meta: Dict[str, Any], switch_pf: int) -> bool:
    acc = meta.get("accidents")
    if not isinstance(acc, list):
        return False
    for event in acc:
        pf = event.get("payload_frame")
        if pf is None:
            continue
        if int(pf) >= switch_pf:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay-script", type=Path, required=True)
    ap.add_argument("--sender-script", type=Path, required=True)
    ap.add_argument("--csv-path", type=Path, required=True)

    ap.add_argument("--fixed-delta", type=float, default=0.1)
    ap.add_argument("--poll-interval", type=float, default=0.1)
    ap.add_argument("--tracking-sec", type=float, default=30.0)
    ap.add_argument("--future-sec", type=float, default=10.0)

    ap.add_argument("--lead-min", type=int, default=1)
    ap.add_argument("--lead-max", type=int, default=10)
    ap.add_argument("--reps", type=int, default=5)

    ap.add_argument("--base-seed", type=int, default=20009)
    ap.add_argument("--startup-delay", type=float, default=1.0)

    ap.add_argument("--carla-host", default="127.0.0.1")
    ap.add_argument("--carla-port", type=int, default=2000)
    ap.add_argument("--listen-host", default="0.0.0.0")
    ap.add_argument("--listen-port", type=int, default=5005)
    ap.add_argument("--sender-host", default="127.0.0.1")
    ap.add_argument("--sender-port", type=int, default=5005)

    ap.add_argument("--outdir", type=Path, default=Path("results_grid"))
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    sender_cmd = [
        sys.executable,
        str(args.sender_script),
        str(args.csv_path),
        "--host",
        args.sender_host,
        "--port",
        str(args.sender_port),
        "--interval",
        str(args.fixed_delta),
    ]

    calib_dir = args.outdir / "calibration"
    logs = calib_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    meta_path = logs / "meta.json"
    collisions_path = logs / "collisions.csv"

    replay_calib = [
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
        "--max-runtime",
        str(args.tracking_sec + args.future_sec),
        "--tm-seed",
        str(args.base_seed),
        "--future-mode",
        "none",
        "--metadata-output",
        str(meta_path),
        "--collision-log",
        str(collisions_path),
    ]

    replay_proc = start_proc(replay_calib)
    time.sleep(max(args.startup_delay, 0.0))
    run_cmd(sender_cmd)
    replay_proc.wait(timeout=args.tracking_sec + args.future_sec + 30.0)

    meta = load_json(meta_path)
    accident_pf = first_accident_payload_frame(meta)
    if accident_pf is None:
        raise RuntimeError(
            "Calibration failed: meta.json に事故が記録されていません（accidents が空）"
        )

    rows: List[Row] = []
    summary_path = args.outdir / "summary_grid.csv"

    for method in ("autopilot", "lstm"):
        for lead_sec in range(args.lead_min, args.lead_max + 1):
            lead_frames = int(round(lead_sec / args.fixed_delta))
            switch_pf = max(accident_pf - lead_frames, 0)

            for rep in range(1, args.reps + 1):
                seed = args.base_seed + rep

                run_dir = args.outdir / method / f"lead_{lead_sec:02d}" / f"rep_{rep:02d}"
                logs = run_dir / "logs"
                logs.mkdir(parents=True, exist_ok=True)
                meta_path = logs / "meta.json"
                collisions_path = logs / "collisions.csv"
                actor_log = logs / "actor.csv"
                id_map = logs / "id_map.csv"

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
                    "--max-runtime",
                    str(args.tracking_sec + args.future_sec),
                    "--tm-seed",
                    str(seed),
                    "--future-mode",
                    method,
                    "--switch-payload-frame",
                    str(switch_pf),
                    "--metadata-output",
                    str(meta_path),
                    "--collision-log",
                    str(collisions_path),
                    "--actor-log",
                    str(actor_log),
                    "--id-map-file",
                    str(id_map),
                ]

                ran_ok = 1
                status = "ok"
                replay_proc: Optional[subprocess.Popen] = None
                try:
                    replay_proc = start_proc(replay_cmd)
                    time.sleep(max(args.startup_delay, 0.0))
                    run_cmd(sender_cmd)
                    replay_proc.wait(timeout=args.tracking_sec + args.future_sec + 30.0)
                except Exception as exc:
                    ran_ok = 0
                    status = f"exception:{type(exc).__name__}"
                finally:
                    if replay_proc is not None and replay_proc.poll() is None:
                        replay_proc.terminate()
                        try:
                            replay_proc.wait(timeout=10)
                        except Exception:
                            pass

                meta = load_json(meta_path)
                first_pf = first_accident_payload_frame(meta)
                after = 1 if any_accident_after(meta, switch_pf) else 0

                rows.append(
                    Row(
                        method=method,
                        lead_sec=lead_sec,
                        rep=rep,
                        seed=seed,
                        switch_payload_frame=switch_pf,
                        ran_ok=ran_ok,
                        accident_after_switch=after,
                        first_accident_payload_frame=first_pf,
                        status=status,
                    )
                )

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "method",
                "lead_sec",
                "rep",
                "seed",
                "switch_payload_frame",
                "ran_ok",
                "accident_after_switch",
                "first_accident_payload_frame",
                "status",
                "accident_payload_frame_ref",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.method,
                    row.lead_sec,
                    row.rep,
                    row.seed,
                    row.switch_payload_frame,
                    row.ran_ok,
                    row.accident_after_switch,
                    row.first_accident_payload_frame,
                    row.status,
                    accident_pf,
                ]
            )

    print(f"Wrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
