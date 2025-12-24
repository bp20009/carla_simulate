"""Replay external tracking data inside a CARLA simulation via UDP."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    Deque,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    TextIO,
    Tuple,
)

import carla
from collections import deque

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
LOGGER = logging.getLogger(__name__)

# 秒数だけ外部トラッキングを行い，その後は CARLA の autopilot に制御を渡す
TRACKING_PHASE_DURATION = 30.0  # 好きな秒数に変えて下さい


def parse_arguments(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--carla-host", default="127.0.0.1", help="Hostname of the CARLA server")
    parser.add_argument(
        "--carla-port", type=int, default=2000, help="TCP port of the CARLA server"
    )
    parser.add_argument(
        "--timeout", type=float, default=10.0, help="Client connection timeout in seconds"
    )
    parser.add_argument(
        "--listen-host", default="0.0.0.0", help="Address that receives the UDP packets"
    )
    parser.add_argument(
        "--listen-port", type=int, default=5005, help="UDP port that receives external data"
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.05,
        help="Seconds to wait for UDP data before advancing the simulation tick",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        default=0.05,
        help="Fixed delta time used in synchronous mode (seconds per frame)",
    )
    parser.add_argument(
        "--stale-timeout",
        type=float,
        default=2.0,
        help="Destroy actors whose data has not been updated for this many seconds",
    )
    parser.add_argument(
        "--max-runtime",
        type=float,
        default=0.0,
        help="Optional maximum runtime in seconds (0 runs indefinitely)",
    )
    parser.add_argument(
        "--future-duration-ticks",
        type=int,
        default=None,
        help=(
            "Optional maximum number of simulation ticks to run after entering "
            "future mode; unlimited when unset"
        ),
    )
    parser.add_argument(
        "--future-duration-sec",
        type=float,
        default=None,
        help=(
            "Optional maximum simulated seconds to run after entering future mode; "
            "ignored when --future-duration-ticks is provided; unlimited when unset"
        ),
    )
    parser.add_argument(
        "--switch-payload-frame",
        type=int,
        default=None,
        help=(
            "Payload frame at which to hand control over to autopilot; defaults to "
            "time-based switch when unset"
        ),
    )
    parser.add_argument(
        "--end-payload-frame",
        type=int,
        default=None,
        help="Payload frame at which to end the replay; runs indefinitely when unset",
    )
    parser.add_argument(
        "--lead-time-sec",
        type=float,
        default=None,
        help=(
            "Optional lead time (seconds) used to compute the autopilot switch frame "
            "when only an end payload frame is provided"
        ),
    )
    parser.add_argument(
        "--future-mode",
        default="autopilot",
        choices=("autopilot", "lstm", "none"),
        help="Future simulation method after switch (default: autopilot)",
    )
    parser.add_argument(
        "--collision-log",
        default=None,
        help="CSV path to write collision/accident events",
    )
    parser.add_argument("--lstm-model", default=None, help="Path to traj_lstm.pt")
    parser.add_argument(
        "--lstm-device",
        default="cpu",
        choices=("cpu", "cuda"),
        help="Device for LSTM",
    )
    parser.add_argument(
        "--lstm-sample-interval",
        type=float,
        default=None,
        help=(
            "Expected sampling interval (seconds) for the LSTM model. When unset, "
            "defaults to the fixed delta time."
        ),
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity",
    )
    parser.add_argument(
        "--measure-update-times",
        action="store_true",
        help="Log frame and per-actor update timings during replay",
    )
    parser.add_argument(
        "--enable-completion",
        action="store_true",
        help=(
            "Fill in missing yaw/heading values from movement direction when incoming data lacks yaw"
        ),
    )
    parser.add_argument(
        "--timing-output",
        default="update_timings.csv",
        help="CSV file that stores frame and actor update timings",
    )
    parser.add_argument(
        "--control-state-file",
        default=None,
        help=(
            "Optional JSON file to mirror current control modes per CARLA actor for "
            "downstream consumers"
        ),
    )
    parser.add_argument(
        "--tm-seed",
        type=int,
        default=None,
        help="Optional random seed for the Traffic Manager",
    )
    parser.add_argument(
        "--metadata-output",
        default=None,
        help=(
            "Optional JSON file that captures run metadata for experiment "
            "traceability"
        ),
    )
    parser.add_argument(
        "--actor-log",
        default=None,
        help="CSV file path for actor pose states per frame (omit to disable)",
    )
    parser.add_argument(
        "--id-map-file",
        default=None,
        help="CSV file that maps external object IDs to CARLA actor IDs/types",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


TYPE_FILTERS: Mapping[str, List[str]] = {
    "vehicle": ["vehicle.ue4.mercedes.ccc", "vehicle.lincoln.mkz_2017", "vehicle.*"],
    "pedestrian": ["walker.pedestrian.0010", "walker.pedestrian.*"],
    "bicycle": ["vehicle.diamondback.century", "vehicle.bh.crossbike", "vehicle.*bike*"],
}

TYPE_ALIASES: Mapping[str, str] = {
    "car": "vehicle",
    "truck": "vehicle",
    "motorcycle": "vehicle",
    "walker": "pedestrian",
    "person": "pedestrian",
    "ped": "pedestrian",
    "cyclist": "bicycle",
    "bike": "bicycle",
}


@dataclass
class IncomingState:
    object_id: str
    object_type: str
    x: float
    y: float
    z: float
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    yaw_provided: bool = False
    payload_frame: Optional[int] = None


class MissingTargetDataError(ValueError):
    """Raised when a message does not contain mandatory target coordinates."""


@dataclass
class LSTMHistoryStep:
    dx: float
    dy: float
    dt: float


@dataclass
class EntityRecord:
    actor: carla.Actor
    object_type: str
    last_update: float
    last_observed_location: Optional[carla.Location] = None
    last_observed_time: Optional[float] = None
    target: Optional[carla.Location] = None
    predicted_target: Optional[carla.Location] = None
    previous_location: Optional[carla.Location] = None
    throttle_pid: Optional[PIDController] = None
    steering_pid: Optional[PIDController] = None
    max_speed: float = 10.0
    autopilot_enabled: bool = False
    control_mode: str = "direct"
    collision_sensor: Optional[carla.Sensor] = None
    last_payload_frame: Optional[int] = None
    lstm_history: Optional[Deque[LSTMHistoryStep]] = None
    lstm_plan: Optional[List[Tuple[float, float]]] = None
    lstm_step: int = 0


class ControlStateBroadcaster:
    """Persist current control modes so other tools can read them."""

    def __init__(self, path: Optional[str]) -> None:
        self._path = Path(path) if path else None
        self._last_payload: Optional[str] = None

    def publish(self, control_state: Mapping[int, Mapping[str, object]]) -> None:
        if self._path is None:
            return

        payload = json.dumps(control_state, ensure_ascii=False, sort_keys=True)
        if payload == self._last_payload:
            return

        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(payload, encoding="utf-8")
            self._last_payload = payload
        except OSError:
            LOGGER.warning("Failed to write control state file '%s'", self._path)


class ActorCSVLogger:
    """Persist per-frame actor transforms and id mappings."""

    def __init__(self, actor_log_path: Optional[str], id_map_path: Optional[str]) -> None:
        self._actor_path = Path(actor_log_path).expanduser() if actor_log_path else None
        self._id_map_path = Path(id_map_path).expanduser() if id_map_path else None
        self._actor_file: Optional[TextIO] = None
        self._actor_writer: Optional[csv.writer] = None
        self._id_map: Dict[str, Tuple[int, str]] = {}

        if self._actor_path is not None:
            try:
                self._actor_path.parent.mkdir(parents=True, exist_ok=True)
                self._actor_file = self._actor_path.open("w", newline="")
                self._actor_writer = csv.writer(self._actor_file)
                self._actor_writer.writerow(
                    [
                        "frame",
                        "frame_source",
                        "carla_frame",
                        "id",
                        "carla_actor_id",
                        "type",
                        "location_x",
                        "location_y",
                        "location_z",
                        "rotation_roll",
                        "rotation_pitch",
                        "rotation_yaw",
                        "autopilot_enabled",
                        "control_mode",
                    ]
                )
                self._actor_file.flush()
            except OSError:
                LOGGER.warning("Failed to open actor log file '%s'", self._actor_path)
                self._actor_file = None
                self._actor_writer = None

        if self._id_map_path is not None:
            try:
                self._id_map_path.parent.mkdir(parents=True, exist_ok=True)
            except OSError:
                LOGGER.warning("Failed to create id map directory '%s'", self._id_map_path)
                self._id_map_path = None

    def register_actor(self, object_id: str, actor: carla.Actor) -> None:
        if self._id_map_path is None:
            return
        self._id_map[object_id] = (actor.id, actor.type_id)
        self._write_id_map()

    def _write_id_map(self) -> None:
        if self._id_map_path is None:
            return
        try:
            with self._id_map_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["object_id", "carla_actor_id", "carla_type"])
                for object_id, (actor_id, actor_type) in sorted(self._id_map.items()):
                    writer.writerow([object_id, actor_id, actor_type])
        except OSError:
            LOGGER.warning("Failed to write id map file '%s'", self._id_map_path)

    def log_frame(
        self,
        payload_frame: Optional[int],
        carla_frame: Optional[int],
        entities: Mapping[str, EntityRecord],
    ) -> None:
        if self._actor_writer is None:
            return

        wrote_row = False
        frame_source = "payload" if payload_frame is not None else "carla"
        frame_value = payload_frame if payload_frame is not None else carla_frame
        if frame_value is None:
            return

        for object_id, record in entities.items():
            actor = record.actor
            if not actor.is_alive:
                continue
            try:
                transform = actor.get_transform()
            except RuntimeError:
                continue

            self._actor_writer.writerow(
                [
                    frame_value,
                    frame_source,
                    carla_frame,
                    object_id,
                    actor.id,
                    actor.type_id,
                    transform.location.x,
                    transform.location.y,
                    transform.location.z,
                    transform.rotation.roll,
                    transform.rotation.pitch,
                    transform.rotation.yaw,
                    record.autopilot_enabled,
                    record.control_mode,
                ]
            )
            wrote_row = True

        if wrote_row and self._actor_file is not None:
            self._actor_file.flush()

    def close(self) -> None:
        if self._actor_file is not None:
            self._actor_file.flush()
            self._actor_file.close()
            self._actor_file = None
            self._actor_writer = None


class PIDController:
    """Simple PID controller for throttle and steering outputs."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        *,
        integral_limit: Optional[float] = None,
        output_limits: Optional[Tuple[float, float]] = None,
    ) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.output_limits = output_limits
        self._integral = 0.0
        self._previous_error: Optional[float] = None

    def reset(self) -> None:
        self._integral = 0.0
        self._previous_error = None

    def step(self, error: float, dt: float) -> float:
        if dt <= 0.0:
            dt = 1e-6

        self._integral += error * dt
        if self.integral_limit is not None:
            limit = abs(self.integral_limit)
            self._integral = max(-limit, min(self._integral, limit))

        derivative = 0.0
        if self._previous_error is not None:
            derivative = (error - self._previous_error) / dt
        self._previous_error = error

        output = self.kp * error + self.ki * self._integral + self.kd * derivative

        if self.output_limits is not None:
            low, high = self.output_limits
            output = max(low, min(output, high))

        return output


def _require_torch() -> Tuple["torch", "nn"]:
    try:
        import torch  # type: ignore
        import torch.nn as nn  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "LSTM mode requires PyTorch; install torch to enable LSTM predictions or "
            "use --future-mode autopilot."
        ) from exc
    return torch, nn


def _load_lstm_model(
    lstm_model_path: str, lstm_device: str
) -> Tuple[object, int, int, "torch"]:
    torch, nn = _require_torch()

    class TrajLSTM(nn.Module):
        def __init__(
            self,
            feature_dim: int = 2,
            hidden_dim: int = 64,
            num_layers: int = 1,
            horizon_steps: int = 50,
        ) -> None:
            super().__init__()
            self.horizon_steps = horizon_steps
            self.lstm = nn.LSTM(
                feature_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
            )
            self.decoder = nn.Linear(hidden_dim, feature_dim)

        def forward(self, history: "torch.Tensor") -> "torch.Tensor":
            _, hidden = self.lstm(history)
            step_input = history[:, -1:, :]
            preds = []
            for _ in range(self.horizon_steps):
                step_out, hidden = self.lstm(step_input, hidden)
                dxy = self.decoder(step_out[:, -1, :])
                preds.append(dxy)
                step_input = dxy.unsqueeze(1)
            return torch.stack(preds, dim=1)

    ckpt = torch.load(lstm_model_path, map_location=lstm_device)
    history_steps = int(ckpt.get("history_steps", 10))
    horizon_steps = int(ckpt.get("horizon_steps", 50))
    model = TrajLSTM(
        feature_dim=2,
        hidden_dim=64,
        num_layers=1,
        horizon_steps=horizon_steps,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    lstm_model = model.to(lstm_device)
    return lstm_model, history_steps, horizon_steps, torch


def _iter_message_objects(obj: object) -> Iterator[Mapping[str, object]]:
    if isinstance(obj, Mapping):
        actors = obj.get("actors")
        if isinstance(actors, list):
            shared_data = {k: v for k, v in obj.items() if k != "actors"}
            for actor in actors:
                if not isinstance(actor, Mapping):
                    LOGGER.debug("Unsupported actor entry in frame payload: %r", actor)
                    continue
                merged: Dict[str, object] = {**shared_data, **actor}
                yield merged
        else:
            yield obj
    elif isinstance(obj, list):
        for item in obj:
            yield from _iter_message_objects(item)
    else:
        LOGGER.debug("Unsupported JSON payload element: %r", obj)


def decode_messages(payload: bytes) -> Iterator[Mapping[str, object]]:
    text = payload.decode("utf-8").strip()
    if not text:
        return

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield from _iter_message_objects(json.loads(stripped))
            except json.JSONDecodeError:
                LOGGER.debug("Discarding invalid JSON fragment: %s", stripped)
        return

    yield from _iter_message_objects(data)


def extract_payload_frame(message: Mapping[str, object]) -> Optional[int]:
    frame = (
        message.get("payload_frame")
        or message.get("frame")
        or message.get("frame_id")
    )
    if frame is None:
        return None

    try:
        return int(frame)
    except (TypeError, ValueError):
        LOGGER.debug("Invalid payload frame value: %r", frame)
        return None


def normalise_message(
    message: Mapping[str, object], *, payload_frame: Optional[int] = None
) -> IncomingState:
    object_id = message.get("id") or message.get("object_id")
    if object_id is None:
        raise ValueError("Missing object identifier in message")

    object_type = message.get("type") or message.get("category")
    if not object_type:
        raise ValueError("Missing object type in message")
    object_type = TYPE_ALIASES.get(str(object_type).lower(), str(object_type).lower())

    def _extract_coordinate(coordinate: str) -> float:
        candidates = (
            coordinate,
            f"target_{coordinate}",
        )
        containers = (
            message,
            message.get("location"),
            message.get("target_location"),
            message.get("target_data"),
        )

        for container in containers:
            if not isinstance(container, Mapping):
                continue
            for key in candidates:
                value = container.get(key)
                if value is not None:
                    try:
                        return float(value)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Invalid numeric value '{value}' for coordinate '{key}'"
                        ) from None

        raise MissingTargetDataError(
            f"Missing coordinate value for '{coordinate}' in message {message}"
        )

    x = _extract_coordinate("x")
    y = _extract_coordinate("y")
    z = _extract_coordinate("z")

    def _extract_rotation(key: str) -> Tuple[float, bool]:
        candidates = (
            key,
            f"target_{key}",
        )
        containers = (
            message,
            message.get("rotation"),
            message.get("target_rotation"),
        )

        for container in containers:
            if not isinstance(container, Mapping):
                continue
            for candidate in candidates:
                value = container.get(candidate)
                if value is not None:
                    try:
                        return float(value), True
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Invalid numeric value '{value}' for rotation '{candidate}'"
                        ) from None
        return 0.0, False

    roll, _ = _extract_rotation("roll")
    pitch, _ = _extract_rotation("pitch")
    yaw, yaw_provided = _extract_rotation("yaw")

    return IncomingState(
        object_id=str(object_id),
        object_type=object_type,
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
        yaw_provided=yaw_provided,
        payload_frame=payload_frame,
    )


class CollisionLogger:
    """Attach collision sensors and persist filtered collision events."""

    ACCIDENT_THRESHOLD = 150.0
    COOLDOWN_SEC = 0.5
    VEHICLE_ONLY = True

    def __init__(
        self,
        world: carla.World,
        *,
        payload_frame_getter: Optional[Callable[[], Optional[int]]] = None,
        log_path: Optional[str] = None,
    ) -> None:
        self._world = world
        self._blueprint = world.get_blueprint_library().find("sensor.other.collision")
        self._log_path = Path(log_path) if log_path else Path("pred_collisions.csv")
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_file = self._log_path.open("w", newline="")
        self._writer = csv.writer(self._log_file)
        self._writer.writerow(
            [
                "time_sec",
                "payload_frame",
                "payload_frame_source",
                "carla_frame",
                "actor_id",
                "actor_type",
                "other_id",
                "other_type",
                "other_class",
                "x",
                "y",
                "z",
                "intensity",
                "is_accident",
            ]
        )

        self._pending_by_frame: Dict[Tuple[int, int], Tuple[List[object], float]] = {}
        self._last_flushed_frame = -1
        self._last_accident_time: Dict[int, float] = {}
        self._accident_summaries: List[Dict[str, object]] = []
        self._payload_frame_getter = payload_frame_getter
        self._ignore_before_time: Optional[float] = None
        self._min_payload_frame: Optional[int] = None

    def set_min_timestamp(self, timestamp: Optional[float]) -> None:
        self._ignore_before_time = timestamp

    def set_min_payload_frame(self, payload_frame: Optional[int]) -> None:
        self._min_payload_frame = int(payload_frame) if payload_frame is not None else None

    @staticmethod
    def _other_class(other_type: str) -> str:
        s = (other_type or "").lower()
        if s.startswith("vehicle."):
            return "vehicle"
        if s.startswith("walker."):
            return "walker"
        if s.startswith("static."):
            return "static"
        if s.startswith("traffic."):
            return "traffic"
        if s.strip() == "":
            return "unknown"
        return "other"

    def _flush_until(self, frame_now: int) -> None:
        if frame_now < 0:
            return

        to_delete: List[Tuple[int, int]] = []
        for (actor_id, frame), (row, _best_intensity) in self._pending_by_frame.items():
            if frame <= frame_now - 1:
                self._writer.writerow(row)
                to_delete.append((actor_id, frame))

        for key in to_delete:
            self._pending_by_frame.pop(key, None)

        if to_delete:
            self._log_file.flush()

        self._last_flushed_frame = max(self._last_flushed_frame, frame_now - 1)

    def flush_all(self) -> None:
        if self._pending_by_frame:
            for row, _ in self._pending_by_frame.values():
                self._writer.writerow(row)
            self._pending_by_frame.clear()
            self._log_file.flush()

    def _make_callback(self, record: EntityRecord):
        actor = record.actor

        def _on_collision(event: carla.CollisionEvent) -> None:
            carla_frame = int(event.frame)
            timestamp = float(getattr(event, "timestamp", time.monotonic()))

            if (
                self._ignore_before_time is not None
                and timestamp < self._ignore_before_time
            ):
                return

            other = event.other_actor
            other_id = other.id if other is not None else None
            other_type = other.type_id if other is not None else ""
            other_cls = self._other_class(other_type)

            impulse = event.normal_impulse
            intensity = (impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2) ** 0.5

            contact_point = getattr(event, "contact_point", None)
            if contact_point is None and hasattr(event, "transform"):
                try:
                    contact_point = event.transform.location
                except Exception:
                    contact_point = None
            if contact_point is None:
                try:
                    contact_point = actor.get_transform().location
                except RuntimeError:
                    contact_point = carla.Location(0.0, 0.0, 0.0)

            is_vehicle_vehicle = other_cls == "vehicle"
            is_accident = (intensity >= self.ACCIDENT_THRESHOLD) and (
                is_vehicle_vehicle if self.VEHICLE_ONLY else True
            )

            if is_accident:
                last_t = self._last_accident_time.get(actor.id, -1e9)
                if (timestamp - last_t) < self.COOLDOWN_SEC:
                    return
                self._last_accident_time[actor.id] = timestamp

            payload_frame = None
            payload_frame_source = "getter"

            if self._payload_frame_getter is not None:
                try:
                    payload_frame = self._payload_frame_getter()
                except Exception:
                    payload_frame = None

            if payload_frame is None:
                payload_frame = record.last_payload_frame
                payload_frame_source = "record"

            if payload_frame is None:
                payload_frame = carla_frame
                payload_frame_source = "carla_fallback"

            # future 開始 payload_frame より前の衝突は記録しない
            if self._min_payload_frame is not None and int(payload_frame) < self._min_payload_frame:
                return

            self._flush_until(int(payload_frame))

            row = [
                float(timestamp),
                payload_frame,
                payload_frame_source,
                carla_frame,
                actor.id,
                record.object_type,
                other_id,
                other_type,
                other_cls,
                float(contact_point.x),
                float(contact_point.y),
                float(contact_point.z),
                float(intensity),
                int(is_accident),
            ]
            key = (actor.id, int(payload_frame))
            prev = self._pending_by_frame.get(key)
            if prev is None or intensity > prev[1]:
                self._pending_by_frame[key] = (row, intensity)

            if is_accident:
                LOGGER.info(
                    "[ACCIDENT] time=%.3fs frame=%d actor_id=%d other_id=%s "
                    "loc=(%.2f,%.2f,%.2f) intensity=%.2f",
                    timestamp,
                    payload_frame,
                    actor.id,
                    str(other_id),
                    contact_point.x,
                    contact_point.y,
                    contact_point.z,
                    intensity,
                )
                self._accident_summaries.append(
                    {
                        "time_sec": float(timestamp),
                        "payload_frame": int(payload_frame),
                        "carla_frame": int(carla_frame),
                        "actor_id": int(actor.id),
                        "other_id": int(other_id) if other_id is not None else None,
                        "intensity": float(intensity),
                    }
                )

        return _on_collision

    def _cleanup_sensor(self, record: EntityRecord) -> None:
        sensor = record.collision_sensor
        if sensor is not None and sensor.is_alive:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                LOGGER.debug(
                    "Failed to destroy collision sensor for actor '%s'", record.actor.id
                )
        record.collision_sensor = None

    def ensure_sensor(self, record: EntityRecord) -> None:
        if record.object_type not in {"vehicle", "bicycle"}:
            return

        if not record.actor.is_alive:
            self._cleanup_sensor(record)
            return

        sensor = record.collision_sensor
        if sensor is not None and sensor.is_alive:
            return
        if sensor is not None:
            self._cleanup_sensor(record)

        try:
            sensor_actor = self._world.spawn_actor(
                self._blueprint, carla.Transform(), attach_to=record.actor
            )
        except RuntimeError:
            LOGGER.debug("Unable to attach collision sensor to actor '%s'", record.actor.id)
            return

        record.collision_sensor = sensor_actor
        sensor_actor.listen(self._make_callback(record))

    def ensure_sensors_for_entities(self, records: Iterable[EntityRecord]) -> None:
        for record in records:
            self.ensure_sensor(record)

    def shutdown(self) -> None:
        self.flush_all()
        self._log_file.close()

    def accident_summaries(self) -> List[Dict[str, object]]:
        return list(self._accident_summaries)


class EntityManager:
    def __init__(
        self,
        world: carla.World,
        blueprint_library: carla.BlueprintLibrary,
        *,
        enable_timing: bool = False,
        timing_output: Optional[TextIO] = None,
        enable_completion: bool = False,
        use_lstm_target: bool = False,
        actor_logger: Optional[ActorCSVLogger] = None,
        lstm_model_path: Optional[str] = None,
        lstm_device: str = "cpu",
        lstm_sample_interval: Optional[float] = None,
        simulation_step_interval: Optional[float] = None,
    ) -> None:
        self._world = world
        self._map = world.get_map()
        self._blueprint_library = blueprint_library
        self._entities: Dict[str, EntityRecord] = {}
        self._timing_enabled = enable_timing
        self._enable_completion = enable_completion
        self._use_lstm_target = use_lstm_target
        self._frame_start_ns: Optional[int] = None
        self._actor_timings: List[Tuple[str, int]] = []
        self._timing_output = timing_output
        self._timing_writer = csv.writer(timing_output) if timing_output else None
        self._timing_header_written = False
        self._next_frame_sequence = 1
        self._active_frame_sequence: Optional[int] = None
        self._current_payload_frame: Optional[int] = None
        self._active_payload_frame: Optional[int] = None
        self._actor_logger = actor_logger

        # autopilot 有効化済みかどうか
        self._autopilot_enabled = False
        self._tm_port: Optional[int] = None
        self._lstm: Optional[object] = None
        self._lstm_history_steps: Optional[int] = None
        self._lstm_horizon_steps: Optional[int] = None
        self._lstm_device = lstm_device
        self._torch: Optional["torch"] = None
        self._lstm_reference_dt = lstm_sample_interval
        self._lstm_step_dt = simulation_step_interval or lstm_sample_interval
        if self._lstm_reference_dt is None:
            self._lstm_reference_dt = self._lstm_step_dt
        if self._lstm_step_dt is None:
            self._lstm_step_dt = self._lstm_reference_dt

        if lstm_model_path:
            lstm_model_path = str(lstm_model_path)
            lstm_device = str(lstm_device)
            model, history_steps, horizon_steps, torch_module = _load_lstm_model(
                lstm_model_path, lstm_device
            )
            self._lstm = model
            self._torch = torch_module
            self._lstm_history_steps = history_steps
            self._lstm_horizon_steps = horizon_steps
            LOGGER.info(
                "Loaded LSTM model: %s (history=%d, horizon=%d, device=%s)",
                lstm_model_path,
                history_steps,
                horizon_steps,
                lstm_device,
            )
            if self._lstm_reference_dt is not None and self._lstm_step_dt is not None:
                LOGGER.info(
                    "Normalizing LSTM history to %.3f s steps and scaling predictions to %.3f s steps",
                    self._lstm_reference_dt,
                    self._lstm_step_dt,
                )

    @property
    def entities(self) -> Mapping[str, EntityRecord]:
        return self._entities

    @property
    def timing_enabled(self) -> bool:
        return self._timing_enabled

    def _record_lstm_delta(
        self, record: EntityRecord, dx: float, dy: float, dt_obs: Optional[float]
    ) -> None:
        if self._lstm_history_steps is None:
            return
        if record.lstm_history is None:
            record.lstm_history = deque(maxlen=self._lstm_history_steps)
        sample_dt = float(dt_obs) if dt_obs is not None else 0.0
        norm_dx, norm_dy = dx, dy
        if self._lstm_reference_dt is not None and sample_dt > 0.0:
            scale = self._lstm_reference_dt / sample_dt
            norm_dx *= scale
            norm_dy *= scale
        record.lstm_history.append(
            LSTMHistoryStep(dx=norm_dx, dy=norm_dy, dt=sample_dt)
        )

    def enable_autopilot(self, traffic_manager: carla.TrafficManager) -> None:
        """現在の vehicle / bicycle アクターを CARLA の autopilot 配下に移行する．"""
        if self._autopilot_enabled:
            return
        self._autopilot_enabled = True

        tm_port = traffic_manager.get_port()
        self._tm_port = tm_port
        for object_id, record in self._entities.items():
            actor = record.actor
            if not actor.is_alive:
                continue

            if record.object_type in {"vehicle", "bicycle"}:
                # Vehicle のみ set_autopilot が有効なので，安全に呼ぶ
                try:
                    actor.set_autopilot(True, tm_port)
                    LOGGER.info("Enabled autopilot for '%s' (%s)", object_id, record.object_type)
                except RuntimeError:
                    LOGGER.warning("Failed to enable autopilot for '%s'", object_id)

                # 以後は自前 PID での制御をやめる
                record.throttle_pid = None
                record.steering_pid = None
                record.autopilot_enabled = True
                record.control_mode = "autopilot"

            # 歩行者は WalkerAIController を使っていないので現状はそのまま

    def control_state_snapshot(self) -> Dict[int, Dict[str, object]]:
        """Return a mapping of CARLA actor id to control mode info."""

        snapshot: Dict[int, Dict[str, object]] = {}
        for record in self._entities.values():
            actor = record.actor
            if not actor.is_alive:
                continue
            snapshot[actor.id] = {
                "autopilot_enabled": bool(record.autopilot_enabled),
                "control_mode": record.control_mode,
            }
        return snapshot

    def begin_frame(self) -> None:
        if not self._timing_enabled:
            return
        self._frame_start_ns = time.perf_counter_ns()
        self._actor_timings.clear()
        self._active_frame_sequence = self._next_frame_sequence
        self._next_frame_sequence += 1
        self._active_payload_frame = self._current_payload_frame

    def end_frame(self, *, completed: bool, frame_id: Optional[int]) -> None:
        if not self._timing_enabled or self._frame_start_ns is None:
            return

        frame_duration_ns = time.perf_counter_ns() - self._frame_start_ns
        actor_count = len(self._actor_timings)
        frame_sequence = self._active_frame_sequence or ""
        payload_frame = self._active_payload_frame
        if payload_frame is None:
            payload_frame = self._current_payload_frame
        if payload_frame is None:
            payload_frame = ""
        if completed:
            LOGGER.info(
                "Frame processed in %.3f ms (payload frame %s, %d actor updates)",
                frame_duration_ns / 1e6,
                payload_frame,
                actor_count,
            )
        else:
            LOGGER.info(
                "Aborted frame after %.3f ms (payload frame %s, %d actor updates)",
                frame_duration_ns / 1e6,
                payload_frame,
                actor_count,
            )

        if LOGGER.isEnabledFor(logging.DEBUG):
            for object_id, elapsed_ns in self._actor_timings:
                LOGGER.debug("  %s updated in %.3f ms", object_id, elapsed_ns / 1e6)

        if self._timing_writer is not None:
            if not self._timing_header_written:
                self._timing_writer.writerow(
                    [
                        "frame_sequence",
                        "payload_frame",
                        "frame_completed",
                        "frame_duration_ms",
                        "actor_count",
                        "actor_id",
                        "actor_duration_ms",
                    ]
                )
                self._timing_header_written = True
            frame_duration_ms = frame_duration_ns / 1e6
            self._timing_writer.writerow(
                [
                    frame_sequence,
                    payload_frame,
                    "yes" if completed else "no",
                    f"{frame_duration_ms:.6f}",
                    actor_count,
                    "",
                    "",
                ]
            )
            for object_id, elapsed_ns in self._actor_timings:
                self._timing_writer.writerow(
                    [
                        frame_sequence,
                        payload_frame,
                        "",
                        "",
                        "",
                        object_id,
                        f"{elapsed_ns / 1e6:.6f}",
                    ]
                )
            self._timing_output.flush()

        self._frame_start_ns = None
        self._actor_timings.clear()
        self._active_frame_sequence = None
        self._active_payload_frame = None

    def update_payload_frame(self, payload_frame: Optional[int]) -> None:
        if payload_frame is None:
            return
        self._current_payload_frame = payload_frame
        self._active_payload_frame = payload_frame

    @property
    def current_payload_frame(self) -> Optional[int]:
        return self._current_payload_frame

    def _select_blueprint(self, object_type: str) -> Optional[carla.ActorBlueprint]:
        filters = TYPE_FILTERS.get(object_type, [])
        for pattern in filters:
            matches = self._blueprint_library.filter(pattern)
            if matches:
                blueprint = matches[0]
                if blueprint.has_attribute("role_name"):
                    blueprint.set_attribute("role_name", "udp_replay")
                return blueprint
        LOGGER.warning("No blueprint found for type '%s'", object_type)
        return None

    def _spawn_actor(self, state: IncomingState) -> Optional[EntityRecord]:
        blueprint = self._select_blueprint(state.object_type)
        if blueprint is None:
            return None

        transform = carla.Transform(
            carla.Location(x=state.x, y=state.y, z=state.z),
            carla.Rotation(roll=state.roll, pitch=state.pitch, yaw=state.yaw),
        )

        actor = self._world.try_spawn_actor(blueprint, transform)
        if actor is None:
            LOGGER.debug("Failed to spawn actor for %s", state.object_id)
            return None

        LOGGER.info("Spawned %s '%s'", state.object_type, state.object_id)

        object_type = state.object_type
        initial_observation = carla.Location(x=state.x, y=state.y, z=state.z)
        now = time.monotonic()
        record = EntityRecord(
            actor=actor,
            object_type=object_type,
            last_update=now,
            last_observed_location=initial_observation,
            last_observed_time=now,
            last_payload_frame=state.payload_frame,
        )

        if object_type in {"vehicle", "bicycle"}:
            try:
                actor.set_autopilot(False)
                actor.set_simulate_physics(True)
            except RuntimeError:
                LOGGER.debug("Unable to configure physics for actor '%s'", state.object_id)

            throttle_pid = PIDController(1.0, 0.0, 0.05, integral_limit=10.0, output_limits=(0.0, 1.0))
            steering_pid = PIDController(2.0, 0.0, 0.2, integral_limit=5.0, output_limits=(-1.0, 1.0))
            record.throttle_pid = throttle_pid
            record.steering_pid = steering_pid
            record.max_speed = 20.0 if object_type == "vehicle" else 10.0
        elif object_type == "pedestrian":
            try:
                actor.set_simulate_physics(True)
            except RuntimeError:
                LOGGER.debug("Unable to enable physics for pedestrian '%s'", state.object_id)
            record.max_speed = 3.0

        if self._autopilot_enabled and object_type in {"vehicle", "bicycle"}:
            try:
                if self._tm_port is not None:
                    actor.set_autopilot(True, self._tm_port)
                else:
                    actor.set_autopilot(True)
                record.autopilot_enabled = True
                record.control_mode = "autopilot"
            except RuntimeError:
                LOGGER.warning("Failed to enable autopilot for late join '%s'", state.object_id)

        if self._actor_logger is not None:
            self._actor_logger.register_actor(state.object_id, actor)

        return record

    def apply_state(self, state: IncomingState, timestamp: float) -> bool:
        record = self._entities.get(state.object_id)
        new_location = carla.Location(x=state.x, y=state.y, z=state.z)
        timing_start_ns = time.perf_counter_ns() if self._frame_start_ns is not None else None

        newly_spawned = False
        if record is None or not record.actor.is_alive:
            record = self._spawn_actor(state)
            if record is None:
                return False
            self._entities[state.object_id] = record
            newly_spawned = True

        actor = record.actor

        actor_transform = actor.get_transform()
        previous_target = (
            record.last_observed_location
            or record.target
            or record.previous_location
            or actor_transform.location
        )
        dx = new_location.x - previous_target.x
        dy = new_location.y - previous_target.y
        distance_sq = dx * dx + dy * dy
        yaw = actor_transform.rotation.yaw
        if state.yaw_provided:
            yaw = state.yaw
        elif self._enable_completion and distance_sq > 1e-8:
            yaw = math.degrees(math.atan2(dy, dx))

        transform = carla.Transform(
            new_location,
            carla.Rotation(roll=state.roll, pitch=state.pitch, yaw=yaw),
        )
        actor.set_transform(transform)

        dt_obs: Optional[float] = None
        observed_velocity = carla.Vector3D(0.0, 0.0, 0.0)
        if record.last_observed_location is not None and record.last_observed_time is not None:
            dt_obs = timestamp - record.last_observed_time
            if dt_obs > 0.0:
                observed_velocity = carla.Vector3D(
                    (new_location.x - record.last_observed_location.x) / dt_obs,
                    (new_location.y - record.last_observed_location.y) / dt_obs,
                    (new_location.z - record.last_observed_location.z) / dt_obs,
                )

        try:
            actor.set_target_velocity(observed_velocity)
            actor.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
        except RuntimeError:
            LOGGER.debug("Unable to set target velocity for actor '%s'", state.object_id)

        if record.throttle_pid is not None:
            record.throttle_pid.reset()
        if record.steering_pid is not None:
            record.steering_pid.reset()

        record.last_update = timestamp
        record.last_observed_location = new_location
        record.last_observed_time = timestamp
        record.previous_location = previous_target
        record.target = new_location
        if state.payload_frame is not None:
            record.last_payload_frame = state.payload_frame
        if not record.autopilot_enabled:
            record.control_mode = "tracking"
        if (
            record.object_type in {"vehicle", "bicycle"}
            and self._lstm is not None
            and self._lstm_history_steps is not None
        ):
            dx = float(new_location.x - previous_target.x)
            dy = float(new_location.y - previous_target.y)
            self._record_lstm_delta(record, dx, dy, dt_obs)

        if timing_start_ns is not None and self._frame_start_ns is not None:
            self._actor_timings.append(
                (state.object_id, time.perf_counter_ns() - timing_start_ns)
            )
        return True

    def destroy_stale(self, now: float, timeout: float) -> None:
        if timeout <= 0:
            return

        stale_ids = [
            object_id
            for object_id, record in self._entities.items()
            if now - record.last_update > timeout
        ]
        for object_id in stale_ids:
            record = self._entities.pop(object_id)
            actor = record.actor
            if actor.is_alive:
                LOGGER.info("Destroying stale actor '%s'", object_id)
                actor.destroy()
            if record.collision_sensor is not None and record.collision_sensor.is_alive:
                record.collision_sensor.stop()
                record.collision_sensor.destroy()
            
    def destroy_all(self) -> None:
        for object_id, record in list(self._entities.items()):
            actor = record.actor
            if actor.is_alive:
                LOGGER.info("Destroying actor '%s'", object_id)
                actor.destroy()
            if record.collision_sensor is not None and record.collision_sensor.is_alive:
                record.collision_sensor.stop()
                record.collision_sensor.destroy()
            self._entities.pop(object_id, None)

    def log_actor_states(
        self,
        logger: Optional[ActorCSVLogger],
        payload_frame: Optional[int],
        carla_frame: Optional[int],
    ) -> None:
        if logger is None:
            return
        logger.log_frame(payload_frame, carla_frame, self._entities)

    def prepare_lstm_plans(self) -> None:
        if (
            self._lstm is None
            or self._lstm_history_steps is None
            or self._lstm_horizon_steps is None
        ):
            LOGGER.warning("LSTM is not available; cannot prepare plans.")
            return
        if self._torch is None:
            LOGGER.error("PyTorch is not available; cannot prepare LSTM plans.")
            return

        torch = self._torch

        for record in self._entities.values():
            if record.object_type not in {"vehicle", "bicycle"}:
                continue
            if record.lstm_history is None or len(record.lstm_history) < self._lstm_history_steps:
                LOGGER.warning(
                    "Insufficient LSTM history for actor %d; len=%s",
                    record.actor.id,
                    0 if record.lstm_history is None else len(record.lstm_history),
                )
                record.lstm_plan = None
                record.lstm_step = 0
                continue

            history_steps = list(record.lstm_history)[-self._lstm_history_steps :]
            history = [(step.dx, step.dy) for step in history_steps]
            x = torch.tensor(
                history, dtype=torch.float32, device=self._lstm_device
            ).unsqueeze(0)
            with torch.no_grad():
                pred = self._lstm(x)
            plan = pred.squeeze(0).detach().cpu().tolist()
            plan_scale = 1.0
            if (
                self._lstm_reference_dt
                and self._lstm_step_dt
                and self._lstm_reference_dt > 0.0
            ):
                plan_scale = self._lstm_step_dt / self._lstm_reference_dt
            record.lstm_plan = [
                (float(d[0] * plan_scale), float(d[1] * plan_scale)) for d in plan
            ]
            record.lstm_step = 0
            LOGGER.info(
                "Prepared LSTM plan for actor %d: steps=%d",
                record.actor.id,
                len(record.lstm_plan),
            )

    def update_lstm_targets(self) -> None:
        if self._lstm is None:
            return
        for record in self._entities.values():
            if record.object_type not in {"vehicle", "bicycle"}:
                continue
            if not record.actor.is_alive:
                continue
            if not record.lstm_plan or record.lstm_step >= len(record.lstm_plan):
                continue

            dx, dy = record.lstm_plan[record.lstm_step]
            record.lstm_step += 1
            try:
                current_location = record.actor.get_transform().location
            except RuntimeError:
                continue

            record.predicted_target = carla.Location(
                x=float(current_location.x + dx),
                y=float(current_location.y + dy),
                z=float(current_location.z),
            )

    def step_all(self, dt: float) -> None:
        if not self._entities:
            return

        for record in self._entities.values():
            actor = record.actor
            if not actor.is_alive:
                continue

            target = (
                record.predicted_target
                if (self._use_lstm_target and record.predicted_target is not None)
                else record.target
            )
            if target is None:
                continue

            try:
                current_transform = actor.get_transform()
            except RuntimeError:
                continue

            current_location = current_transform.location
            direction_vector = target - current_location
            distance = math.sqrt(
                direction_vector.x ** 2 + direction_vector.y ** 2 + direction_vector.z ** 2
            )

            if record.object_type in {"vehicle", "bicycle"}:
                control = self._compute_vehicle_control(record, current_transform, direction_vector, distance, dt)
                actor.apply_control(control)
            elif record.object_type == "pedestrian":
                control = self._compute_pedestrian_control(record, direction_vector, distance, dt)
                actor.apply_control(control)

    def _compute_vehicle_control(
        self,
        record: EntityRecord,
        transform: carla.Transform,
        direction_vector: carla.Vector3D,
        distance: float,
        dt: float,
    ) -> carla.VehicleControl:
        velocity = record.actor.get_velocity()
        speed = math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

        desired_speed = min(distance / max(dt, 1e-3), record.max_speed)
        throttle_error = desired_speed - speed
        throttle = 0.0
        if record.throttle_pid is not None:
            throttle = record.throttle_pid.step(throttle_error, dt)

        yaw_rad = math.radians(transform.rotation.yaw)
        target_yaw = math.atan2(direction_vector.y, direction_vector.x)
        yaw_error = math.atan2(math.sin(target_yaw - yaw_rad), math.cos(target_yaw - yaw_rad))
        steer = 0.0
        if record.steering_pid is not None:
            steer = record.steering_pid.step(yaw_error, dt)

        brake = 0.0
        hand_brake = False
        reverse = False
        if distance < 0.5 and speed < 0.2:
            throttle = 0.0
            brake = 0.5

        return carla.VehicleControl(
            throttle=float(max(0.0, min(throttle, 1.0))),
            steer=float(max(-1.0, min(steer, 1.0))),
            brake=float(max(0.0, min(brake, 1.0))),
            hand_brake=hand_brake,
            reverse=reverse,
        )

    def _compute_pedestrian_control(
        self,
        record: EntityRecord,
        direction_vector: carla.Vector3D,
        distance: float,
        dt: float,
    ) -> carla.WalkerControl:
        control = carla.WalkerControl()
        if distance < 0.2:
            control.speed = 0.0
            control.direction = carla.Vector3D(0.0, 0.0, 0.0)
            return control

        norm = math.sqrt(direction_vector.x ** 2 + direction_vector.y ** 2 + direction_vector.z ** 2)
        if norm > 0:
            control.direction = carla.Vector3D(
                direction_vector.x / norm,
                direction_vector.y / norm,
                direction_vector.z / norm,
            )
        desired_speed = min(distance / max(dt, 1e-3), record.max_speed)
        control.speed = float(desired_speed)
        return control



@contextmanager
def synchronous_mode(world: carla.World, fixed_delta_seconds: float) -> Iterator[None]:
    original_settings = world.get_settings()
    new_settings = carla.WorldSettings()
    new_settings.no_rendering_mode = original_settings.no_rendering_mode
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = fixed_delta_seconds
    new_settings.substepping = original_settings.substepping
    new_settings.max_substeps = original_settings.max_substeps
    new_settings.max_substep_delta_time = original_settings.max_substep_delta_time

    world.apply_settings(new_settings)
    try:
        yield
    finally:
        world.apply_settings(original_settings)


def run(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_arguments(argv)
    logging.basicConfig(level=getattr(logging, args.log_level))

    lstm_sample_interval = args.lstm_sample_interval or args.fixed_delta
    if args.future_mode == "lstm":
        if args.lstm_model is None:
            LOGGER.error("LSTM future mode requested but --lstm-model was not provided.")
            return 1
        if lstm_sample_interval is not None and lstm_sample_interval <= 0:
            LOGGER.error(
                "LSTM sample interval must be positive; got %.3f", lstm_sample_interval
            )
            return 1

    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    # Traffic Manager を取得しておく（autopilot 用）
    traffic_manager = client.get_trafficmanager()
    # synchronous_mode のときは Traffic Manager も同期モードに
    traffic_manager.set_synchronous_mode(True)
    if args.tm_seed is not None:
        traffic_manager.set_random_device_seed(args.tm_seed)

    timing_file: Optional[TextIO] = None
    if args.measure_update_times:
        output_path = Path(args.timing_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        timing_file = output_path.open("w", newline="")

    metadata_path = Path(args.metadata_output).expanduser() if args.metadata_output else None
    metadata: Dict[str, object] | None = None
    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "input_identifiers": {
                "carla": f"{args.carla_host}:{args.carla_port}",
                "udp_listener": f"{args.listen_host}:{args.listen_port}",
            },
            "fixed_delta_seconds": args.fixed_delta,
            "traffic_manager_seed": args.tm_seed,
            "tracking_phase_duration_seconds": TRACKING_PHASE_DURATION,
            "future_duration_ticks": args.future_duration_ticks,
            "future_duration_seconds": (
                args.future_duration_ticks * args.fixed_delta
                if args.future_duration_ticks is not None and args.fixed_delta > 0.0
                else args.future_duration_sec
            ),
            "future_start_time_seconds": None,
            "first_frame": None,
            "switch_frame": None,
            "end_frame": None,
            "lead_time_seconds": None,
            "lstm_sample_interval_seconds": lstm_sample_interval
            if args.lstm_model is not None
            else None,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    actor_logger = ActorCSVLogger(args.actor_log, args.id_map_file)

    try:
        manager = EntityManager(
            world,
            blueprint_library,
            enable_timing=args.measure_update_times,
            timing_output=timing_file,
            enable_completion=args.enable_completion,
            actor_logger=actor_logger,
            lstm_model_path=args.lstm_model if args.future_mode == "lstm" else None,
            lstm_device=args.lstm_device,
            lstm_sample_interval=lstm_sample_interval,
            simulation_step_interval=args.fixed_delta,
        )
    except RuntimeError as exc:
        LOGGER.error("%s", exc)
        return 1

    control_broadcaster = ControlStateBroadcaster(args.control_state_file)
    collision_logger = CollisionLogger(
        world,
        payload_frame_getter=lambda: manager.current_payload_frame,
        log_path=args.collision_log,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.listen_host, args.listen_port))
    sock.settimeout(args.poll_interval)
    LOGGER.info(
        "Listening for UDP packets on %s:%d", args.listen_host, args.listen_port
    )

    start_time = time.monotonic()
    next_cleanup = start_time
    last_step_time = start_time
    switch_wall_time: Optional[float] = None

    switch_payload_frame = args.switch_payload_frame
    end_payload_frame = args.end_payload_frame
    computed_switch_payload_frame: Optional[int] = None
    future_mode_kind = args.future_mode
    if future_mode_kind == "none":
        switch_payload_frame = None
    if (
        switch_payload_frame is None
        and future_mode_kind != "none"
        and end_payload_frame is not None
        and args.lead_time_sec is not None
        and args.fixed_delta > 0.0
    ):
        frame_offset = int(round(args.lead_time_sec / args.fixed_delta))
        computed_switch_payload_frame = max(end_payload_frame - frame_offset, 0)
        switch_payload_frame = computed_switch_payload_frame

    if computed_switch_payload_frame is not None:
        LOGGER.info(
            "Computed switch payload frame %d using lead time %.3f s",
            computed_switch_payload_frame,
            args.lead_time_sec,
        )

    # 追加: トラッキング開始時刻と future モードフラグ
    has_received_first_data = False
    tracking_start_time: Optional[float] = None
    future_mode = False
    first_frame: Optional[int] = None
    switch_frame: Optional[int] = None
    end_frame: Optional[int] = None
    first_payload_frame: Optional[int] = None
    last_payload_frame: Optional[int] = None
    switch_payload_frame_observed: Optional[int] = None
    switch_reason: Optional[str] = None
    future_tick_budget_remaining: Optional[int] = None
    future_time_budget_remaining: Optional[float] = None
    future_start_sim_time: Optional[float] = None
    snapshot = world.get_snapshot()
    timestamp = getattr(snapshot, "timestamp", None)
    latest_sim_time: float = float(getattr(timestamp, "elapsed_seconds", 0.0)) if timestamp else 0.0
    latest_sim_delta: float = float(getattr(timestamp, "delta_seconds", args.fixed_delta)) if timestamp else args.fixed_delta

    def enter_future_mode(reason: str, observed_payload_frame: Optional[int]) -> None:
        nonlocal future_mode, switch_wall_time, switch_reason, switch_payload_frame_observed
        nonlocal future_tick_budget_remaining, future_time_budget_remaining, future_start_sim_time
        switch_wall_time = time.monotonic()
        future_mode = True
        switch_reason = reason
        switch_payload_frame_observed = observed_payload_frame
        future_start_sim_time = latest_sim_time
        if args.future_duration_ticks is not None:
            future_tick_budget_remaining = args.future_duration_ticks
        elif args.future_duration_sec is not None:
            future_time_budget_remaining = args.future_duration_sec
        collision_logger.set_min_payload_frame(observed_payload_frame)
        if future_mode_kind == "autopilot":
            manager.enable_autopilot(traffic_manager)
        elif future_mode_kind == "lstm":
            manager._use_lstm_target = True
            manager.prepare_lstm_plans()

    try:
        with synchronous_mode(world, args.fixed_delta):
            while True:
                frame_completed = False
                carla_frame_id: Optional[int] = None
                if manager.timing_enabled:
                    manager.begin_frame()
                try:
                    now = time.monotonic()
                    if args.max_runtime and now - start_time >= args.max_runtime:
                        LOGGER.info(
                            "Reached maximum runtime (%.1f s)", args.max_runtime
                        )
                        break

                    # --- トラッキングフェーズか future フェーズかで処理を切り替える ---

                    if not future_mode:
                        # ===== トラッキングフェーズ =====
                        applied_any = False
                        while True:
                            try:
                                payload, _ = sock.recvfrom(65535)
                            except socket.timeout:
                                break
                            except OSError as exc:
                                LOGGER.error("Socket error: %s", exc)
                                return 1
                            else:
                                for raw_message in decode_messages(payload):
                                    payload_frame = extract_payload_frame(raw_message)
                                    manager.update_payload_frame(payload_frame)
                                    if payload_frame is not None:
                                        if first_payload_frame is None:
                                            first_payload_frame = payload_frame
                                        last_payload_frame = payload_frame
                                    try:
                                        state = normalise_message(
                                            raw_message, payload_frame=payload_frame
                                        )
                                    except MissingTargetDataError as exc:
                                        LOGGER.warning("Incomplete target data: %s", exc)
                                        continue
                                    except (TypeError, ValueError) as exc:
                                        LOGGER.debug("Ignoring invalid message: %s", exc)
                                        continue

                                    applied = manager.apply_state(
                                        state, time.monotonic()
                                    )
                                    applied_any = applied_any or applied
                        if applied_any and not has_received_first_data:
                            LOGGER.info("Received first complete tracking update")
                            has_received_first_data = True
                            tracking_start_time = time.monotonic()

                        # 最新 payload_frame を取得した上で判定する
                        current_payload_frame = manager.current_payload_frame

                        # トラッキング開始から一定時間経過したら future モードへ
                        if (
                            switch_payload_frame is None
                            and not future_mode
                            and has_received_first_data
                            and tracking_start_time is not None
                            and (now - tracking_start_time) >= TRACKING_PHASE_DURATION
                            and future_mode_kind != "none"
                        ):
                            LOGGER.info(
                                "Switching to future simulation mode (autopilot) "
                                "after %.1f seconds of tracking",
                                TRACKING_PHASE_DURATION,
                            )
                            enter_future_mode("time_based", current_payload_frame)
                        elif (
                            switch_payload_frame is not None
                            and not future_mode
                            and current_payload_frame is not None
                            and current_payload_frame >= switch_payload_frame
                            and future_mode_kind != "none"
                        ):
                            LOGGER.info(
                                "Switching to future simulation mode (autopilot) "
                                "at payload frame %d",
                                current_payload_frame,
                            )
                            enter_future_mode("payload_frame", current_payload_frame)

                    current_payload_frame = manager.current_payload_frame

                    if (
                        end_payload_frame is not None
                        and current_payload_frame is not None
                        and current_payload_frame >= end_payload_frame
                    ):
                        LOGGER.info(
                            "Reached end payload frame %d; stopping replay",
                            current_payload_frame,
                        )
                        break

                    collision_logger.ensure_sensors_for_entities(
                        manager.entities.values()
                    )

                    current_time = time.monotonic()
                    elapsed = current_time - last_step_time
                    if args.fixed_delta > 0.0:
                        if elapsed < args.fixed_delta:
                            time.sleep(args.fixed_delta - elapsed)
                        current_time = time.monotonic()
                        elapsed = args.fixed_delta

                    if future_mode and future_mode_kind == "lstm":
                        manager.update_lstm_targets()

                    # autopilot では future_mode 中は step_all しない．
                    # lstm では future_mode 中も predicted_target へ追従する．
                    if (not future_mode) or (future_mode and future_mode_kind == "lstm"):
                        manager.step_all(elapsed)
                    # future_mode のときは Traffic Manager / autopilot が制御するので
                    # ここでは何もしない（world.tick() だけ進める）

                    # future_mode 中は UDP から payload_frame が増えないので，疑似的に進める
                    if future_mode:
                        pf = manager.current_payload_frame
                        if pf is None:
                            pf = first_payload_frame
                        if pf is not None:
                            pf_next = int(pf) + 1
                            manager.update_payload_frame(pf_next)
                            for rec in manager.entities.values():
                                rec.last_payload_frame = pf_next

                    carla_frame_id = world.tick()
                    snapshot = world.get_snapshot()
                    timestamp = getattr(snapshot, "timestamp", None)
                    if timestamp is not None:
                        latest_sim_time = float(
                            getattr(timestamp, "elapsed_seconds", latest_sim_time)
                        )
                        latest_sim_delta = float(
                            getattr(timestamp, "delta_seconds", latest_sim_delta)
                        )
                    if first_frame is None:
                        first_frame = carla_frame_id
                    if future_mode and switch_frame is None:
                        switch_frame = carla_frame_id
                        if metadata is not None and tracking_start_time is not None:
                            metadata["lead_time_seconds"] = (
                                switch_wall_time - tracking_start_time
                                if switch_wall_time
                                else None
                            )
                    end_frame = carla_frame_id
                    frame_completed = True
                    last_step_time = time.monotonic()

                    budget_exhausted = False
                    if future_mode:
                        if future_tick_budget_remaining is not None:
                            future_tick_budget_remaining -= 1
                            budget_exhausted = future_tick_budget_remaining <= 0
                        elif future_time_budget_remaining is not None:
                            future_time_budget_remaining -= latest_sim_delta
                            budget_exhausted = future_time_budget_remaining <= 0

                    now = last_step_time
                    # stale アクタの自動 destroy は future_mode では無効化しておく
                    if not future_mode and now >= next_cleanup:
                        manager.destroy_stale(now, args.stale_timeout)
                        next_cleanup = now + max(args.stale_timeout * 0.5, 0.5)

                    control_broadcaster.publish(manager.control_state_snapshot())
                    manager.log_actor_states(
                        actor_logger, manager.current_payload_frame, carla_frame_id
                    )
                    if budget_exhausted:
                        if args.future_duration_ticks is not None:
                            LOGGER.info(
                                "Future duration exhausted after %d ticks; stopping replay",
                                args.future_duration_ticks,
                            )
                        else:
                            LOGGER.info(
                                "Future duration exhausted after %.3f seconds; stopping replay",
                                args.future_duration_sec,
                            )
                        break
                finally:
                    if manager.timing_enabled:
                        manager.end_frame(
                            completed=frame_completed, frame_id=carla_frame_id
                        )

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        manager.destroy_all()
        collision_logger.shutdown()
        sock.close()
        if timing_file is not None:
            timing_file.close()
        actor_logger.close()

    if metadata is not None and metadata_path is not None:
        metadata["first_frame"] = first_frame
        metadata["switch_frame"] = switch_frame
        metadata["end_frame"] = end_frame
        metadata["future_mode"] = future_mode_kind
        metadata["first_payload_frame"] = first_payload_frame
        metadata["last_payload_frame"] = last_payload_frame
        metadata["switch_payload_frame_raw"] = args.switch_payload_frame
        metadata["switch_payload_frame_used"] = switch_payload_frame
        metadata["switch_payload_frame_observed"] = switch_payload_frame_observed
        metadata["switch_reason"] = switch_reason
        metadata["end_payload_frame"] = end_payload_frame
        metadata["computed_switch_payload_frame"] = computed_switch_payload_frame
        metadata["accidents"] = collision_logger.accident_summaries()
        metadata["future_start_time_seconds"] = future_start_sim_time
        metadata["future_duration_ticks"] = args.future_duration_ticks
        metadata["future_duration_seconds"] = (
            args.future_duration_ticks * args.fixed_delta
            if args.future_duration_ticks is not None and args.fixed_delta > 0.0
            else args.future_duration_sec
        )
        metadata["lead_time_seconds"] = metadata.get("lead_time_seconds") or (
            switch_wall_time - tracking_start_time
            if switch_wall_time is not None and tracking_start_time is not None
            else None
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(run())
