import json
import socket

import pytest

from send_data.send_udp_frames_from_csv import build_frame_payload, iter_frames, send_frames


def test_iter_frames_groups_by_frame() -> None:
    rows = [
        {"frame": "10", "id": "a"},
        {"frame": "10", "id": "b"},
        {"frame": "11", "id": "c"},
    ]
    grouped = list(iter_frames(rows))
    assert grouped == [
        ("10", [{"frame": "10", "id": "a"}, {"frame": "10", "id": "b"}]),
        ("11", [{"frame": "11", "id": "c"}]),
    ]


def test_iter_frames_raises_when_frame_missing() -> None:
    with pytest.raises(KeyError):
        list(iter_frames([{"id": "1"}]))


def test_build_frame_payload_with_max_actors_and_float_coercion() -> None:
    payload = build_frame_payload(
        "12",
        [
            {"id": "1", "type": "vehicle", "x": "1.5", "y": "2", "z": ""},
            {"id": "2", "type": "walker", "x": "bad", "y": "3", "z": "4"},
        ],
        max_actors=1,
    )
    assert payload == {
        "frame": "12",
        "actors": [{"id": "1", "type": "vehicle", "x": 1.5, "y": 2.0, "z": None}],
    }


def test_send_frames_filters_by_range_and_stride() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(0.5)
    host, port = receiver.getsockname()

    frames = [
        ("10", [{"frame": "10", "id": "a", "x": "1", "y": "1", "z": "0"}]),
        ("11", [{"frame": "11", "id": "b", "x": "2", "y": "2", "z": "0"}]),
        ("12", [{"frame": "12", "id": "c", "x": "3", "y": "3", "z": "0"}]),
        ("13", [{"frame": "13", "id": "d", "x": "4", "y": "4", "z": "0"}]),
    ]
    send_frames(
        frames,
        destination=(host, port),
        interval=0.0,
        delay_column=None,
        encoding="utf-8",
        frame_stride=2,
        start_frame=11,
        end_frame=12,
        max_actors=None,
    )

    packet, _ = receiver.recvfrom(4096)
    payload = json.loads(packet.decode("utf-8"))
    assert payload["frame"] == "12"
    with pytest.raises(socket.timeout):
        receiver.recvfrom(4096)
    receiver.close()
