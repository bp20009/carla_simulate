"""Replay external tracking data inside a CARLA simulation via UDP."""

from __future__ import annotations

import argparse
import json
import logging
import socket
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Mapping, Optional

import carla

LOGGER = logging.getLogger(__name__)


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
    return parser.parse_args(list(argv) if argv is not None else None)


TYPE_FILTERS: Mapping[str, List[str]] = {
    "vehicle": ["vehicle.tesla.model3", "vehicle.lincoln.mkz_2017", "vehicle.*"],
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


class MissingTargetDataError(ValueError):
    """Raised when a message does not contain mandatory target coordinates."""


@dataclass
class TrackedActor:
    actor: carla.Actor
    last_update: float


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
                yield json.loads(stripped)
            except json.JSONDecodeError:
                LOGGER.debug("Discarding invalid JSON fragment: %s", stripped)
        return

    if isinstance(data, list):
        for item in data:
            if isinstance(item, Mapping):
                yield item
            else:
                LOGGER.debug("Unsupported list element type: %r", item)
    elif isinstance(data, Mapping):
        yield data
    else:
        LOGGER.debug("Unsupported JSON payload: %r", data)


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

    def _extract_rotation(key: str) -> float:
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
                        return float(value)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Invalid numeric value '{value}' for rotation '{candidate}'"
                        ) from None
        return 0.0

    roll = _extract_rotation("roll")
    pitch = _extract_rotation("pitch")
    yaw = _extract_rotation("yaw")

    return IncomingState(
        object_id=str(object_id),
        object_type=object_type,
        x=x,
        y=y,
        z=z,
        roll=roll,
        pitch=pitch,
        yaw=yaw,
    )


class ActorManager:
    def __init__(self, world: carla.World, blueprint_library: carla.BlueprintLibrary) -> None:
        self._world = world
        self._blueprint_library = blueprint_library
        self._actors: Dict[str, TrackedActor] = {}

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

    def _spawn_actor(self, state: IncomingState) -> Optional[carla.Actor]:
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
        else:
            LOGGER.info(
                "Spawned %s '%s'", state.object_type, state.object_id
            )
            try:
                actor.set_simulate_physics(False)
            except RuntimeError:
                LOGGER.debug("Unable to disable physics for actor '%s'", state.object_id)
        return actor

    def apply_state(self, state: IncomingState, timestamp: float) -> bool:
        tracked = self._actors.get(state.object_id)
        transform = carla.Transform(
            carla.Location(x=state.x, y=state.y, z=state.z),
            carla.Rotation(roll=state.roll, pitch=state.pitch, yaw=state.yaw),
        )

        if tracked is None or not tracked.actor.is_alive:
            actor = self._spawn_actor(state)
            if actor is None:
                return False
            tracked = TrackedActor(actor=actor, last_update=timestamp)
            self._actors[state.object_id] = tracked
        else:
            actor = tracked.actor

        actor.set_transform(transform)
        tracked.last_update = timestamp
        return True

    def destroy_stale(self, now: float, timeout: float) -> None:
        if timeout <= 0:
            return
        stale_ids = [
            object_id
            for object_id, tracked in self._actors.items()
            if now - tracked.last_update > timeout
        ]
        for object_id in stale_ids:
            actor = self._actors.pop(object_id).actor
            if actor.is_alive:
                LOGGER.info("Destroying stale actor '%s'", object_id)
                actor.destroy()

    def destroy_all(self) -> None:
        for object_id, tracked in list(self._actors.items()):
            actor = tracked.actor
            if actor.is_alive:
                LOGGER.info("Destroying actor '%s'", object_id)
                actor.destroy()
            self._actors.pop(object_id, None)


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
    manager = ActorManager(world, blueprint_library)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.listen_host, args.listen_port))
    sock.settimeout(args.poll_interval)
    LOGGER.info(
        "Listening for UDP packets on %s:%d", args.listen_host, args.listen_port
    )

    start_time = time.monotonic()
    next_cleanup = start_time

    try:
        with synchronous_mode(world, args.fixed_delta):
            has_received_first_data = False
            while True:
                now = time.monotonic()
                if args.max_runtime and now - start_time >= args.max_runtime:
                    LOGGER.info("Reached maximum runtime (%.1f s)", args.max_runtime)
                    break

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

                            applied = manager.apply_state(state, time.monotonic())
                            if applied and not has_received_first_data:
                                LOGGER.info("Received first complete tracking update")
                                has_received_first_data = True

                world.tick()

                now = time.monotonic()
                if now >= next_cleanup:
                    manager.destroy_stale(now, args.stale_timeout)
                    next_cleanup = now + max(args.stale_timeout * 0.5, 0.5)

    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user")
    finally:
        manager.destroy_all()
        sock.close()

    return 0


if __name__ == "__main__":
    sys.exit(run())
