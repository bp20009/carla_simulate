#!/usr/bin/env python3
"""Send grouped CARLA frame snapshots from a reduced CSV over UDP."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

from send_data.send_udp_from_csv import (
    Row,
    configure_common_sender_arguments,
    iter_rows,
)

LOGGER = logging.getLogger(__name__)

FrameRows = Sequence[Row]
Payload = Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send UDP packets where each payload represents all actors in a frame "
            "from a reduced vehicle state CSV."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help=(
            "Path to the reduced CSV produced by scripts/convert_vehicle_state_csv.py"
        ),
    )
    configure_common_sender_arguments(parser)
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Field delimiter used in the CSV file (default: ',')",
    )
    parser.add_argument(
        "--quotechar",
        default='"',
        help="Quote character used in the CSV file (default: '\"')",
    )
    return parser.parse_args()


def iter_frames(rows: Iterable[Row]) -> Iterator[Tuple[str, List[Row]]]:
    current_frame: str | None = None
    frame_rows: List[Row] = []

    for row in rows:
        frame_value = row.get("frame")
        if frame_value is None:
            raise KeyError("CSV row missing required 'frame' column")

        if current_frame is None:
            current_frame = frame_value
        if frame_value != current_frame:
            LOGGER.debug("Grouped %d actors for frame %s", len(frame_rows), current_frame)
            yield current_frame, frame_rows
            frame_rows = []
            current_frame = frame_value

        frame_rows.append(row)

    if current_frame is not None:
        LOGGER.debug("Grouped %d actors for frame %s", len(frame_rows), current_frame)
        yield current_frame, frame_rows


def _coerce_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        LOGGER.debug("Unable to convert value '%s' to float", value)
        return None


def build_frame_payload(frame_id: str, rows: FrameRows) -> Payload:
    actors = []
    for row in rows:
        actors.append(
            {
                "id": row.get("id"),
                "type": row.get("type"),
                "x": _coerce_float(row.get("x")),
                "y": _coerce_float(row.get("y")),
                "z": _coerce_float(row.get("z")),
            }
        )
    return {"frame": frame_id, "actors": actors}


def send_frames(
    frames: Iterable[Tuple[str, FrameRows]],
    *,
    destination: Tuple[str, int],
    interval: float,
    delay_column: str | None,
    encoding: str,
) -> None:
    host, port = destination
    sent_frames = 0
    missing_delay_logged = False

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for frame_id, frame_rows in frames:
            if not frame_rows:
                continue

            payload_dict = build_frame_payload(frame_id, frame_rows)
            payload_text = json.dumps(payload_dict, ensure_ascii=False)
            LOGGER.debug(
                "Sending frame %s payload to %s:%d: %s", frame_id, host, port, payload_text
            )
            sock.sendto(payload_text.encode(encoding), destination)
            sent_frames += 1

            delay = interval
            if delay_column:
                delay_value = frame_rows[0].get(delay_column)
                if delay_value is None:
                    if not missing_delay_logged:
                        LOGGER.warning(
                            "Frame %s missing delay column '%s'; falling back to interval",
                            frame_id,
                            delay_column,
                        )
                        missing_delay_logged = True
                else:
                    try:
                        delay = float(delay_value)
                    except ValueError:
                        LOGGER.warning(
                            "Invalid delay value '%s' for frame %s; falling back to interval",
                            delay_value,
                            frame_id,
                        )
            if delay > 0:
                LOGGER.debug("Sleeping for %s seconds", delay)
                time.sleep(delay)

    LOGGER.info("Sent %d frames via UDP to %s:%d", sent_frames, host, port)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    rows = iter_rows(
        args.csv_path,
        delimiter=args.delimiter,
        quotechar=args.quotechar,
        encoding=args.encoding,
    )
    frames = iter_frames(rows)
    send_frames(
        frames,
        destination=(args.host, args.port),
        interval=args.interval,
        delay_column=args.delay_column,
        encoding=args.encoding,
    )


if __name__ == "__main__":
    main()
