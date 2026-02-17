from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path


class UdpMockServer:
    def __init__(self, expected_packets: int, timeout_sec: float = 3.0) -> None:
        self.expected_packets = expected_packets
        self.timeout_sec = timeout_sec
        self.received: list[str] = []
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.settimeout(0.2)
        self.host, self.port = self.sock.getsockname()

    def start(self) -> None:
        def _run() -> None:
            deadline = time.time() + self.timeout_sec
            try:
                while len(self.received) < self.expected_packets and time.time() < deadline:
                    try:
                        payload, _ = self.sock.recvfrom(65535)
                    except socket.timeout:
                        continue
                    self.received.append(payload.decode("utf-8"))
            except BaseException as exc:  # pragma: no cover
                self._error = exc
            finally:
                self.sock.close()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def join(self) -> None:
        if self._thread is None:
            return
        self._thread.join(timeout=self.timeout_sec + 0.5)
        if self._error is not None:
            raise self._error


def _run_sender(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"Sender failed rc={proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def test_integration_send_udp_from_csv_with_mock_server(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "input.csv"
    csv_path.write_text("message,id\nhello,1\nworld,2\n", encoding="utf-8")

    server = UdpMockServer(expected_packets=2)
    server.start()
    _run_sender(
        [
            sys.executable,
            "send_data/send_udp_from_csv.py",
            str(csv_path),
            "--host",
            server.host,
            "--port",
            str(server.port),
            "--message-column",
            "message",
            "--interval",
            "0",
            "--max-rows",
            "2",
        ],
        cwd=repo_root,
    )
    server.join()

    assert server.received == ["hello", "world"]


def test_integration_send_udp_frames_with_mock_server(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    csv_path = tmp_path / "reduced.csv"
    csv_path.write_text(
        "frame,id,type,x,y,z\n"
        "100,1,vehicle,1.0,2.0,0\n"
        "100,2,walker,3.0,4.0,0\n"
        "101,3,vehicle,5.0,6.0,0\n",
        encoding="utf-8",
    )

    server = UdpMockServer(expected_packets=2)
    server.start()
    _run_sender(
        [
            sys.executable,
            "send_data/send_udp_frames_from_csv.py",
            str(csv_path),
            "--host",
            server.host,
            "--port",
            str(server.port),
            "--interval",
            "0",
            "--frame-stride",
            "1",
        ],
        cwd=repo_root,
    )
    server.join()

    assert len(server.received) == 2
    first = json.loads(server.received[0])
    second = json.loads(server.received[1])
    assert first["frame"] == "100"
    assert second["frame"] == "101"
    assert len(first["actors"]) == 2
