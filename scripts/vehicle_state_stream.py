"""Stream CARLA vehicle and pedestrian states with stable identifiers."""

from __future__ import annotations

import argparse
import csv
import json
import itertools
import sys
import time
from typing import Dict, Iterable, TextIO

import carla


def parse_arguments(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA server port")
    parser.add_argument(
        "--timeout", default=10.0, type=float, help="Client connection timeout (s)"
    )
    parser.add_argument(
        "--interval",
        default=0.0,
        type=float,
        help="Optional sleep interval between snapshots in seconds",
    )
    parser.add_argument(
        "--mode",
        choices=("wait", "on-tick"),
        default="wait",
        help=(
            "Synchronization strategy: block on wait_for_tick (default) or register an "
            "on_tick callback"
        ),
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Destination CSV file (use '-' for stdout)",
    )
    parser.add_argument(
        "--wall-clock",
        action="store_true",
        help=(
            "Include wall-clock timestamps alongside CARLA frame numbers in the CSV "
            "output"
        ),
    )
    parser.add_argument(
        "--frame-elapsed",
        action="store_true",
        help=(
            "Include the frame-to-frame delta time reported by CARLA in the CSV output"
        ),
    )
    parser.add_argument(
        "--include-velocity",
        action="store_true",
        help="Include actor velocity vectors in the CSV output",
    )
    parser.add_argument(
        "--control-state-file",
        default=None,
        help=(
            "Optional JSON file containing autopilot/control mode overrides keyed by "
            "CARLA actor ID"
        ),
    )
    parser.add_argument(
        "--role-prefix",
        default="",
        help="Only log actors whose role_name starts with this prefix",
    )
    parser.add_argument(
        "--include-monotonic",
        action="store_true",
        help="Include time.monotonic() in the CSV output",
    )
    parser.add_argument(
        "--include-tick-wall-dt",
        action="store_true",
        help="Include wall-clock delta between snapshots in the CSV output",
    )
    parser.add_argument(
        "--timing-output",
        default=None,
        help="Optional CSV file to write per-frame timing measurements",
    )
    parser.add_argument(
        "--timing-flush-every",
        default=1,
        type=int,
        help="Flush timing output every N frames (default: 1)",
    )
    return parser.parse_args(list(argv))


def _get_autopilot_state(actor: carla.Actor) -> bool:
    """Return True if the actor reports being under autopilot control.

    CARLA only exposes autopilot on vehicle types; pedestrians and other actors
    will always report False. Some CARLA versions expose it as a property and
    others as a method, so both are handled defensively.
    """

    if not actor.type_id.startswith("vehicle."):
        return False

    autopilot_attr = getattr(actor, "is_autopilot_enabled", None)
    try:
        if callable(autopilot_attr):
            return bool(autopilot_attr())
        if autopilot_attr is not None:
            return bool(autopilot_attr)
    except RuntimeError:
        # Actor may have been destroyed between snapshot and query.
        return False
    return False


def stream_vehicle_states(
    host: str,
    port: int,
    timeout: float,
    interval: float,
    output: TextIO,
    include_wall_clock: bool,
    include_frame_elapsed: bool,
    include_velocity: bool,
    mode: str,
    control_state_file: str | None,
    role_prefix: str,
    include_monotonic: bool,
    include_tick_wall_dt: bool,
    timing_output: TextIO | None,
    timing_flush_every: int,
) -> None:
    """Continuously write vehicle and pedestrian transforms with stable IDs to CSV."""
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()

    actor_to_custom_id: Dict[int, int] = {}
    id_sequence = itertools.count(1)

    writer = csv.writer(output)
    header_prefix = []
    if include_tick_wall_dt:
        header_prefix.append("tick_wall_dt")
    if include_monotonic:
        header_prefix.append("monotonic_time")
    if include_frame_elapsed:
        header_prefix.append("frame_elapsed")
    if include_wall_clock:
        header_prefix.append("wall_time")
    header = header_prefix + [
        "frame",
        "external_id",
        "role_name",
        "id",
        "carla_actor_id",
        "type",
        "location_x",
        "location_y",
        "location_z",
        "rotation_roll",
        "rotation_pitch",
        "rotation_yaw",
    ]
    if include_velocity:
        header.extend(["velocity_x", "velocity_y", "velocity_z"])
    header.extend(["autopilot_enabled", "control_mode"])
    writer.writerow(header)
    output.flush()

    throttle_with_interval = interval > 0.0 and mode == "on-tick"
    last_emit_monotonic: float | None = None
    last_snapshot_monotonic: float | None = None
    timing_writer = None
    timing_frame_count = 0
    if timing_output is not None:
        timing_writer = csv.writer(timing_output)
        timing_header = ["t_monotonic", "carla_frame", "frame_processing_ms"]
        if include_tick_wall_dt:
            timing_header.append("tick_wall_dt_ms")
        timing_writer.writerow(timing_header)
        timing_output.flush()

    def load_control_overrides() -> Dict[int, Dict[str, object]]:
        if not control_state_file:
            return {}

        try:
            with open(control_state_file, "r", encoding="utf-8") as f:
                data = f.read()
        except FileNotFoundError:
            return {}
        except OSError:
            return {}

        if not data.strip():
            return {}

        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return {}

        overrides: Dict[int, Dict[str, object]] = {}
        if isinstance(parsed, dict):
            for key, value in parsed.items():
                try:
                    actor_id = int(key)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, dict):
                    overrides[actor_id] = value
        return overrides

    def handle_snapshot(world_snapshot: carla.WorldSnapshot) -> None:
        nonlocal last_emit_monotonic, last_snapshot_monotonic
        nonlocal timing_frame_count

        now_monotonic = time.monotonic()
        tick_wall_dt = None
        if last_snapshot_monotonic is not None:
            tick_wall_dt = now_monotonic - last_snapshot_monotonic
        last_snapshot_monotonic = now_monotonic
        processing_start_ns = time.perf_counter_ns()

        control_overrides = load_control_overrides()

        if throttle_with_interval:
            if (
                last_emit_monotonic is not None
                and now_monotonic - last_emit_monotonic < interval
            ):
                return
            last_emit_monotonic = now_monotonic

        wall_time = time.time() if include_wall_clock else None
        frame_elapsed = (
            world_snapshot.timestamp.delta_seconds if include_frame_elapsed else None
        )
        actors = world.get_actors()
        tracked_actors = (
            actor for actor in actors if actor.type_id.startswith(("vehicle.", "walker."))
        )

        wrote_frame = False
        for actor in tracked_actors:
            actor_id = actor.id
            if actor_id not in actor_to_custom_id:
                actor_to_custom_id[actor_id] = next(id_sequence)

            role_name = actor.attributes.get("role_name", "")
            if role_prefix and not role_name.startswith(role_prefix):
                continue

            external_id = ""
            if role_prefix and role_name.startswith(role_prefix):
                external_id = role_name[len(role_prefix) :]
            elif role_name.startswith("udp_replay:"):
                external_id = role_name.split(":", 1)[1]

            transform = actor.get_transform()
            try:
                velocity = actor.get_velocity() if include_velocity else None
            except RuntimeError:
                velocity = None
            autopilot_enabled = _get_autopilot_state(actor)
            override = control_overrides.get(actor_id)
            if isinstance(override, dict):
                if "autopilot_enabled" in override:
                    autopilot_enabled = bool(override["autopilot_enabled"])
                control_mode = str(override.get("control_mode", "")) or (
                    "autopilot" if autopilot_enabled else "direct"
                )
            else:
                control_mode = "autopilot" if autopilot_enabled else "direct"
            row_prefix = []
            if include_tick_wall_dt:
                row_prefix.append(tick_wall_dt)
            if include_monotonic:
                row_prefix.append(now_monotonic)
            if include_frame_elapsed:
                row_prefix.append(frame_elapsed)
            if include_wall_clock:
                row_prefix.append(wall_time)
            row = row_prefix + [
                world_snapshot.frame,
                external_id,
                role_name,
                actor_to_custom_id[actor_id],
                actor_id,
                actor.type_id,
                transform.location.x,
                transform.location.y,
                transform.location.z,
                transform.rotation.roll,
                transform.rotation.pitch,
                transform.rotation.yaw,
            ]
            if include_velocity:
                row.extend(
                    [
                        velocity.x if velocity is not None else None,
                        velocity.y if velocity is not None else None,
                        velocity.z if velocity is not None else None,
                    ]
                )
            row.extend([autopilot_enabled, control_mode])
            writer.writerow(row)
            wrote_frame = True

        if wrote_frame:
            output.flush()
        processing_end_ns = time.perf_counter_ns()
        if timing_writer is not None:
            frame_processing_ms = (processing_end_ns - processing_start_ns) / 1_000_000.0
            timing_row = [
                now_monotonic,
                world_snapshot.frame,
                frame_processing_ms,
            ]
            if include_tick_wall_dt:
                timing_row.append(tick_wall_dt * 1000.0 if tick_wall_dt is not None else None)
            timing_writer.writerow(timing_row)
            timing_frame_count += 1
            if timing_output is not None and timing_flush_every > 0:
                if timing_frame_count % timing_flush_every == 0:
                    timing_output.flush()

    try:
        if mode == "on-tick":
            callback_id = world.on_tick(handle_snapshot)
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                world.remove_on_tick(callback_id)
        else:
            while True:
                world_snapshot = world.wait_for_tick()
                handle_snapshot(world_snapshot)
                if interval > 0.0:
                    time.sleep(interval)
    except KeyboardInterrupt:
        pass


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_arguments(argv or sys.argv[1:])
    if args.output == "-":
        output_stream = sys.stdout
        close_output = False
    else:
        output_stream = open(args.output, "w", newline="")
        close_output = True
    if args.timing_output:
        timing_output = open(args.timing_output, "w", newline="")
    else:
        timing_output = None

    try:
        stream_vehicle_states(
            args.host,
            args.port,
            args.timeout,
            args.interval,
            output_stream,
            args.wall_clock,
            args.frame_elapsed,
            args.include_velocity,
            args.mode,
            args.control_state_file,
            args.role_prefix,
            args.include_monotonic,
            args.include_tick_wall_dt,
            timing_output,
            args.timing_flush_every,
        )
    finally:
        if close_output:
            output_stream.close()
        if timing_output is not None and timing_output is not output_stream:
            timing_output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
