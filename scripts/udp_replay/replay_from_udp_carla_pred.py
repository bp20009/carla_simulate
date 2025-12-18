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
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, TextIO, Tuple

import carla

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


class MissingTargetDataError(ValueError):
    """Raised when a message does not contain mandatory target coordinates."""


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
    last_observed_location: Optional[carla.Location] = None
    last_observed_time: Optional[float] = None
    throttle_pid: Optional[PIDController] = None
    steering_pid: Optional[PIDController] = None
    max_speed: float = 10.0
    autopilot_enabled: bool = False
    control_mode: str = "direct"


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


def normalise_message(message: Mapping[str, object]) -> IncomingState:
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
    )


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

        # autopilot 有効化済みかどうか
        self._autopilot_enabled = False

    @property
    def timing_enabled(self) -> bool:
        return self._timing_enabled

    def enable_autopilot(self, traffic_manager: carla.TrafficManager) -> None:
        """現在の vehicle / bicycle アクターを CARLA の autopilot 配下に移行する．"""
        if self._autopilot_enabled:
            return
        self._autopilot_enabled = True

        tm_port = traffic_manager.get_port()
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

    def end_frame(self, *, completed: bool, frame_id: Optional[int]) -> None:
        if not self._timing_enabled or self._frame_start_ns is None:
            return

        frame_duration_ns = time.perf_counter_ns() - self._frame_start_ns
        actor_count = len(self._actor_timings)
        frame_sequence = self._active_frame_sequence or ""
        carla_frame = frame_id if frame_id is not None else ""
        if completed:
            LOGGER.info(
                "Frame processed in %.3f ms (%d actor updates)",
                frame_duration_ns / 1e6,
                actor_count,
            )
        else:
            LOGGER.info(
                "Aborted frame after %.3f ms (%d actor updates)",
                frame_duration_ns / 1e6,
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
                        "carla_frame",
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
                    carla_frame,
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
                        carla_frame,
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
                actor.set_autopilot(True)
                record.autopilot_enabled = True
                record.control_mode = "autopilot"
            except RuntimeError:
                LOGGER.warning("Failed to enable autopilot for late join '%s'", state.object_id)

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
        if not record.autopilot_enabled:
            record.control_mode = "tracking"

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

    def destroy_all(self) -> None:
        for object_id, record in list(self._entities.items()):
            actor = record.actor
            if actor.is_alive:
                LOGGER.info("Destroying actor '%s'", object_id)
                actor.destroy()
            self._entities.pop(object_id, None)

    def step_all(self, dt: float) -> None:
        if not self._entities:
            return

        for record in self._entities.values():
            actor = record.actor
            if not actor.is_alive:
                continue

            target = (
                record.predicted_target if self._use_lstm_target else record.target
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
            "first_frame": None,
            "switch_frame": None,
            "end_frame": None,
            "lead_time_seconds": None,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    manager = EntityManager(
        world,
        blueprint_library,
        enable_timing=args.measure_update_times,
        timing_output=timing_file,
        enable_completion=args.enable_completion,
    )

    control_broadcaster = ControlStateBroadcaster(args.control_state_file)

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
    switch_frame: Optional[int] = None
    end_frame: Optional[int] = None
    first_frame: Optional[int] = None

    # 追加: トラッキング開始時刻と future モードフラグ
    has_received_first_data = False
    tracking_start_time: Optional[float] = None
    future_mode = False

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
                                    try:
                                        state = normalise_message(raw_message)
                                    except MissingTargetDataError as exc:
                                        LOGGER.warning("Incomplete target data: %s", exc)
                                        continue
                                    except (TypeError, ValueError) as exc:
                                        LOGGER.debug("Ignoring invalid message: %s", exc)
                                        continue

                                    applied = manager.apply_state(
                                        state, time.monotonic()
                                    )
                                    if applied and not has_received_first_data:
                                        LOGGER.info(
                                            "Received first complete tracking update"
                                        )
                                        has_received_first_data = True
                                        tracking_start_time = time.monotonic()

                        # トラッキング開始から一定時間経過したら future モードへ
                        if (
                            not future_mode
                            and has_received_first_data
                            and tracking_start_time is not None
                            and (now - tracking_start_time) >= TRACKING_PHASE_DURATION
                        ):
                            LOGGER.info(
                                "Switching to future simulation mode (autopilot) "
                                "after %.1f seconds of tracking",
                                TRACKING_PHASE_DURATION,
                            )
                            future_mode = True
                            manager.enable_autopilot(traffic_manager)
                            switch_wall_time = time.monotonic()

                    current_time = time.monotonic()
                    elapsed = current_time - last_step_time
                    if args.fixed_delta > 0.0:
                        if elapsed < args.fixed_delta:
                            time.sleep(args.fixed_delta - elapsed)
                        current_time = time.monotonic()
                        elapsed = args.fixed_delta

                    # トラッキングフェーズ中だけ自前 PID で target へ追従
                    if not future_mode:
                        manager.step_all(elapsed)
                    # future_mode のときは Traffic Manager / autopilot が制御するので
                    # ここでは何もしない（world.tick() だけ進める）

                    carla_frame_id = world.tick()
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

                    now = last_step_time
                    # stale アクタの自動 destroy は future_mode では無効化しておく
                    if not future_mode and now >= next_cleanup:
                        manager.destroy_stale(now, args.stale_timeout)
                        next_cleanup = now + max(args.stale_timeout * 0.5, 0.5)

                    control_broadcaster.publish(manager.control_state_snapshot())
                finally:
                    if manager.timing_enabled:
                        manager.end_frame(
                            completed=frame_completed, frame_id=carla_frame_id
                        )

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        manager.destroy_all()
        sock.close()
        if timing_file is not None:
            timing_file.close()

    if metadata is not None and metadata_path is not None:
        metadata["first_frame"] = first_frame
        metadata["switch_frame"] = switch_frame
        metadata["end_frame"] = end_frame
        metadata["lead_time_seconds"] = metadata.get("lead_time_seconds") or (
            switch_wall_time - tracking_start_time
            if switch_wall_time is not None and tracking_start_time is not None
            else None
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    sys.exit(run())
