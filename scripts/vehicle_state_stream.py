"""Stream CARLA vehicle states with stable identifiers."""

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
        "--output",
        default="-",
        help="Destination CSV file (use '-' for stdout)",
    )
    return parser.parse_args(list(argv))


def stream_vehicle_states(
    host: str, port: int, timeout: float, interval: float, output: TextIO
) -> None:
    """Continuously write vehicle transforms with stable IDs to CSV."""
    client = carla.Client(host, port)
    client.set_timeout(timeout)
    world = client.get_world()

    actor_to_custom_id: Dict[int, int] = {}
    id_sequence = itertools.count(1)

    writer = csv.writer(output)
    writer.writerow(
        [
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
    )
    output.flush()

    try:
        while True:
            world_snapshot = world.wait_for_tick()
            vehicles = world.get_actors().filter("vehicle.*")

            wrote_frame = False
            for vehicle in vehicles:
                actor_id = vehicle.id
                if actor_id not in actor_to_custom_id:
                    actor_to_custom_id[actor_id] = next(id_sequence)

                transform = vehicle.get_transform()
                writer.writerow(
                    [
                        world_snapshot.frame,
                        actor_to_custom_id[actor_id],
                        actor_id,
                        vehicle.type_id,
                        transform.location.x,
                        transform.location.y,
                        transform.location.z,
                        transform.rotation.roll,
                        transform.rotation.pitch,
                        transform.rotation.yaw,
                    ]
                )
                wrote_frame = True

            if wrote_frame:
                output.flush()

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
            args.host, args.port, args.timeout, args.interval, output_stream
        )
    finally:
        if close_output:
            output_stream.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
