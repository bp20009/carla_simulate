"""Convert CARLA vehicle state CSVs to a reduced format."""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Iterable, TextIO


def parse_arguments(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source",
        help="Source CSV file produced by vehicle_state_stream.py (use '-' for stdin)",
    )
    parser.add_argument(
        "destination",
        help="Destination CSV file (use '-' for stdout)",
    )
    return parser.parse_args(list(argv))


def map_actor_type(type_id: str) -> str:
    if type_id.startswith("vehicle."):
        return "vehicle"
    if type_id.startswith("walker."):
        return "walker"
    return type_id


def convert_rows(source: TextIO, destination: TextIO) -> None:
    reader = csv.DictReader(source)
    fieldnames = ["frame", "id", "type", "x", "y", "z"]
    writer = csv.DictWriter(destination, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        writer.writerow(
            {
                "frame": row.get("frame", ""),
                "id": row.get("id", ""),
                "type": map_actor_type(row.get("type", "")),
                "x": row.get("location_x", ""),
                "y": row.get("location_y", ""),
                "z": row.get("location_z", ""),
            }
        )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_arguments(argv or sys.argv[1:])

    if args.source == "-":
        source_stream = sys.stdin
        close_source = False
    else:
        source_stream = open(args.source, "r", newline="")
        close_source = True

    if args.destination == "-":
        destination_stream = sys.stdout
        close_destination = False
    else:
        destination_stream = open(args.destination, "w", newline="")
        close_destination = True

    try:
        convert_rows(source_stream, destination_stream)
    finally:
        if close_source:
            source_stream.close()
        if close_destination:
            destination_stream.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
