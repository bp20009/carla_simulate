#!/usr/bin/env python3
"""Send MQTT messages based on rows in a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator

import paho.mqtt.client as mqtt

LOGGER = logging.getLogger(__name__)

Row = Dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send MQTT messages where each payload is derived from a row in the "
            "provided CSV file."
        )
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to the CSV file containing data to send",
    )
    parser.add_argument(
        "topic",
        help="MQTT topic to publish messages to",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Hostname or IP address of the MQTT broker (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1883,
        help="TCP port of the MQTT broker (default: 1883)",
    )
    parser.add_argument(
        "--client-id",
        help="Optional MQTT client identifier (default: random)",
    )
    parser.add_argument(
        "--username",
        help="Username for authenticated MQTT brokers",
    )
    parser.add_argument(
        "--password",
        help="Password for authenticated MQTT brokers",
    )
    parser.add_argument(
        "--keepalive",
        type=int,
        default=60,
        help="Keepalive interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--qos",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help="MQTT QoS level for published messages (default: 0)",
    )
    parser.add_argument(
        "--retain",
        action="store_true",
        help="Set the retain flag on published messages",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Fixed delay in seconds between messages (default: 0)",
    )
    parser.add_argument(
        "--delay-column",
        help=(
            "Optional CSV column specifying the delay in seconds after "
            "publishing that row"
        ),
    )
    parser.add_argument(
        "--message-column",
        help=(
            "Optional CSV column whose value will be used as the message payload. "
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
        help='Quote character used in the CSV file (default: \"\")',
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for reading the CSV file and sending payloads",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional limit on the number of rows to send",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging verbosity (default: INFO)",
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
    client: mqtt.Client,
    topic: str,
    interval: float,
    delay_column: str | None,
    message_column: str | None,
    encoding: str,
    max_rows: int | None,
    qos: int,
    retain: bool,
) -> None:
    sent_rows = 0
    for row in rows:
        if max_rows is not None and sent_rows >= max_rows:
            LOGGER.info("Reached maximum row limit (%d)", max_rows)
            break

        payload_text = build_payload(row, message_column=message_column)
        payload = payload_text.encode(encoding)
        LOGGER.debug("Publishing payload to %s: %s", topic, payload_text)
        result = client.publish(topic, payload, qos=qos, retain=retain)
        result.wait_for_publish()
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            raise RuntimeError(f"Failed to publish message: {mqtt.error_string(result.rc)}")
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

    LOGGER.info("Sent %d rows via MQTT to topic '%s'", sent_rows, topic)


def build_payload(row: Row, *, message_column: str | None) -> str:
    if message_column is None:
        return json.dumps(row, ensure_ascii=False)

    if message_column not in row:
        raise KeyError(f"CSV does not contain a column named '{message_column}'")

    return row[message_column]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    if not args.csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")

    client = mqtt.Client(client_id=args.client_id)
    if args.username or args.password:
        client.username_pw_set(args.username, password=args.password)

    LOGGER.info(
        "Connecting to MQTT broker %s:%d (keepalive=%d)",
        args.host,
        args.port,
        args.keepalive,
    )
    client.connect(args.host, args.port, keepalive=args.keepalive)
    client.loop_start()
    try:
        rows = iter_rows(
            args.csv_path,
            delimiter=args.delimiter,
            quotechar=args.quotechar,
            encoding=args.encoding,
        )
        send_rows(
            rows,
            client=client,
            topic=args.topic,
            interval=args.interval,
            delay_column=args.delay_column,
            message_column=args.message_column,
            encoding=args.encoding,
            max_rows=args.max_rows,
            qos=args.qos,
            retain=args.retain,
        )
    finally:
        LOGGER.info("Disconnecting from MQTT broker")
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    main()
