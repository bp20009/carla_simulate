"""Stream CARLA vehicle and pedestrian states with stable identifiers."""

from __future__ import annotations

import argparse
import csv
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
    return parser.parse_args(list(argv))


def stream_vehicle_states(
    host: str,
    port: int,
    timeout: float,
    interval: float,
    output: TextIO,
    include_wall_clock: bool,
    include_frame_elapsed: bool,
    mode: str,
) -> None:
    """Continuously write vehicle and pedestrian transforms with stable IDs to CSV."""
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()

    actor_to_custom_id: Dict[int, int] = {}
    id_sequence = itertools.count(1)

    writer = csv.writer(output)
    header_prefix = []
    if include_frame_elapsed:
        header_prefix.append("frame_elapsed")
    if include_wall_clock:
        header_prefix.append("wall_time")
    header = header_prefix + [
        "frame",
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
    writer.writerow(header)
    output.flush()

    throttle_with_interval = interval > 0.0 and mode == "on-tick"
    last_emit_monotonic: float | None = None

    def handle_snapshot(world_snapshot: carla.WorldSnapshot) -> None:
        nonlocal last_emit_monotonic

        if throttle_with_interval:
            now = time.monotonic()
            if last_emit_monotonic is not None and now - last_emit_monotonic < interval:
                return
            last_emit_monotonic = now

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

            transform = actor.get_transform()
            row_prefix = []
            if include_frame_elapsed:
                row_prefix.append(frame_elapsed)
            if include_wall_clock:
                row_prefix.append(wall_time)
            row = row_prefix + [
                world_snapshot.frame,
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
            writer.writerow(row)
            wrote_frame = True

        if wrote_frame:
            output.flush()

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

    try:
        stream_vehicle_states(
            args.host,
            args.port,
            args.timeout,
            args.interval,
            output_stream,
            args.wall_clock,
            args.frame_elapsed,
            args.mode,
        )
    finally:
        if close_output:
            output_stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
