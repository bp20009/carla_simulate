#!/usr/bin/env python3
"""Send UDP packets based on the contents of a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import socket
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

LOGGER = logging.getLogger(__name__)


Row = Dict[str, str]


def configure_common_sender_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach CLI options shared by UDP sender utilities."""

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Destination host for UDP packets (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5005,
        help="Destination port for UDP packets (default: 5005)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Fixed delay in seconds between packets (default: 0)",
    )
    parser.add_argument(
        "--delay-column",
        help=(
            "Optional CSV column specifying the delay in seconds after "
            "sending each message"
        ),
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for reading the CSV file and sending payloads",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: INFO)",
    )
    return parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send UDP packets where each payload is derived from a row in the "
            "provided CSV file."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file containing data to send",
    )
    configure_common_sender_arguments(parser)
    parser.add_argument(
        "--message-column",
        help=(
            "Optional CSV column whose value will be used as the packet payload. "
            "If omitted, the entire row is sent as JSON."
        ),
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Field delimiter used in the CSV file (default: ',')",
    )
    parser.add_argument(
        "--quotechar",
        default='"',
        help='Quote character used in the CSV file (default: \'"\')',
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional limit on the number of rows to send",
    )
    return parser.parse_args()


def iter_rows(csv_path: Path, *, delimiter: str, quotechar: str, encoding: str) -> Iterator[Row]:
    with csv_path.open("r", encoding=encoding, newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter, quotechar=quotechar)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row")
        for index, row in enumerate(reader, start=1):
            LOGGER.debug("Loaded row %d: %s", index, row)
            yield row


def send_rows(
    rows: Iterable[Row],
    *,
    destination: Tuple[str, int],
    interval: float,
    delay_column: str | None,
    message_column: str | None,
    encoding: str,
    max_rows: int | None,
) -> None:
    sent_rows = 0
    host, port = destination
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        for row in rows:
            if max_rows is not None and sent_rows >= max_rows:
                LOGGER.info("Reached maximum row limit (%d)", max_rows)
                break

            payload_text = build_payload(row, message_column=message_column)
            payload = payload_text.encode(encoding)
            LOGGER.debug("Sending payload to %s:%d: %s", host, port, payload_text)
            sock.sendto(payload, destination)
            sent_rows += 1

            delay = interval
            if delay_column:
                delay_value = row.get(delay_column)
                if delay_value is None:
                    LOGGER.warning(
                        "Row missing delay column '%s'; falling back to interval",
                        delay_column,
                    )
                else:
                    try:
                        delay = float(delay_value)
                    except ValueError:
                        LOGGER.warning(
                            "Invalid delay value '%s'; falling back to interval",
                            delay_value,
                        )
            if delay > 0:
                LOGGER.debug("Sleeping for %s seconds", delay)
                time.sleep(delay)

    LOGGER.info("Sent %d rows via UDP to %s:%d", sent_rows, host, port)


def build_payload(row: Row, *, message_column: str | None) -> str:
    if message_column is None:
        return json.dumps(row, ensure_ascii=False)

    if message_column not in row:
        raise KeyError(
            f"CSV does not contain a column named '{message_column}'"
        )

    return row[message_column]


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
    send_rows(
        rows,
        destination=(args.host, args.port),
        interval=args.interval,
        delay_column=args.delay_column,
        message_column=args.message_column,
        encoding=args.encoding,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
