"""Helpers for applying incoming tracking state updates to CARLA actors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import carla

if TYPE_CHECKING:  # pragma: no cover
    from .replay_from_udp import EntityManager, IncomingState

LOGGER = logging.getLogger(__name__)


def _clone_location(location: carla.Location) -> carla.Location:
    return carla.Location(location.x, location.y, location.z)


def _clone_rotation(rotation: carla.Rotation) -> carla.Rotation:
    return carla.Rotation(roll=rotation.roll, pitch=rotation.pitch, yaw=rotation.yaw)


def apply_state_angle_only(
    manager: "EntityManager", state: "IncomingState", timestamp: float
) -> bool:
    """Apply an incoming state to an existing actor while only adjusting its angle.

    Actors that have not yet been spawned are created in the requested location, just
    like the original :meth:`EntityManager.apply_state`. For already tracked actors the
    position target is preserved and only the rotation is refreshed so that downstream
    PID controllers keep driving towards the existing waypoint while allowing their
    facing direction to stay aligned with the incoming data feed.
    """

    record = manager._entities.get(state.object_id)  # type: ignore[attr-defined]

    desired_location = carla.Location(x=state.x, y=state.y, z=state.z)
    desired_rotation = carla.Rotation(roll=state.roll, pitch=state.pitch, yaw=state.yaw)

    if record is None or not record.actor.is_alive:
        transform = carla.Transform(desired_location, desired_rotation)
        record = manager._spawn_actor(state)  # type: ignore[attr-defined]
        if record is None:
            return False
        manager._entities[state.object_id] = record  # type: ignore[attr-defined]
        try:
            record.actor.set_transform(transform)
        except RuntimeError:
            LOGGER.debug("Failed to place new actor '%s'", state.object_id)
        record.last_update = timestamp
        record.target = _clone_location(transform.location)
        record.target_rotation = _clone_rotation(transform.rotation)
        return True

    actor = record.actor
    try:
        current_transform = actor.get_transform()
    except RuntimeError:
        LOGGER.debug("Unable to fetch transform for '%s'", state.object_id)
        return False

    new_transform = carla.Transform(current_transform.location, desired_rotation)
    try:
        actor.set_transform(new_transform)
    except RuntimeError:
        LOGGER.debug("Failed to update rotation for actor '%s'", state.object_id)
        return False

    record.last_update = timestamp
    record.target_rotation = _clone_rotation(new_transform.rotation)
    return True
