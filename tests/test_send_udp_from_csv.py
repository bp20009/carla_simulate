import json
import socket
from pathlib import Path

import pytest

from send_data.send_udp_from_csv import build_payload, iter_rows, send_rows


def test_build_payload_uses_json_when_no_message_column() -> None:
    row = {"a": "1", "b": "text"}
    payload = build_payload(row, message_column=None)
    assert json.loads(payload) == row


def test_build_payload_uses_message_column() -> None:
    row = {"message": "hello", "x": "1"}
    assert build_payload(row, message_column="message") == "hello"


def test_build_payload_raises_for_missing_message_column() -> None:
    with pytest.raises(KeyError):
        build_payload({"x": "1"}, message_column="message")


def test_iter_rows_reads_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    rows = list(iter_rows(csv_path, delimiter=",", quotechar='"', encoding="utf-8"))
    assert rows == [{"a": "1", "b": "2"}, {"a": "3", "b": "4"}]


def test_send_rows_sends_udp_packets_and_honors_max_rows() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(0.5)
    host, port = receiver.getsockname()

    rows = [
        {"message": "one"},
        {"message": "two"},
        {"message": "three"},
    ]
    send_rows(
        rows,
        destination=(host, port),
        interval=0.0,
        delay_column=None,
        message_column="message",
        encoding="utf-8",
        max_rows=2,
    )

    first, _ = receiver.recvfrom(4096)
    second, _ = receiver.recvfrom(4096)
    assert first.decode("utf-8") == "one"
    assert second.decode("utf-8") == "two"
    with pytest.raises(socket.timeout):
        receiver.recvfrom(4096)
    receiver.close()
