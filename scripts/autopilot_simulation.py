"""Spawn autopilot vehicle, walker, and cyclist in a CARLA world.

This script connects to a CARLA 0.10.0 server, spawns a vehicle on autopilot
and two pedestrian-style actors (a walker and a cyclist) controlled by AI
controllers, waits for a configurable duration, and then cleans up every actor.

The script is intentionally light-weight so that it can be used as a quick
smoke test for a running CARLA instance. For repeatable behaviour, pass
``--seed`` when executing the script.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from typing import Iterable, List, Tuple

import carla


def parse_arguments(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA server port")
    parser.add_argument(
        "--timeout",
        default=10.0,
        type=float,
        help="Client connection timeout in seconds",
    )
    parser.add_argument(
        "--duration",
        default=30.0,
        type=float,
        help="How long to keep the simulation running before cleanup",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible actor selection and placement",
    )
    return parser.parse_args(list(argv))


def find_blueprint(
    blueprint_library: carla.BlueprintLibrary, *filters: str
) -> carla.ActorBlueprint:
    """Return a random blueprint matching the given filters."""
    for pattern in filters:
        matches = blueprint_library.filter(pattern)
        if matches:
            return random.choice(matches)
    raise RuntimeError(
        "No blueprints found for filters: {}".format(", ".join(filters))
    )


def get_random_nav_location(world: carla.World) -> carla.Location:
    for _ in range(10):
        location = world.get_random_location_from_navigation()
        if location is not None:
            return location
    raise RuntimeError("Unable to find a valid navigation location.")


def spawn_vehicle(
    world: carla.World, blueprint_library: carla.BlueprintLibrary
) -> carla.Actor:
    vehicle_bp = find_blueprint(
        blueprint_library,
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2017",
        "vehicle.*",
    )
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available for vehicles.")
    random.shuffle(spawn_points)

    for transform in spawn_points:
        vehicle = world.try_spawn_actor(vehicle_bp, transform)
        if vehicle is not None:
            vehicle.set_autopilot(True)
            return vehicle

    raise RuntimeError("Failed to spawn vehicle at any available spawn point.")


def spawn_walker(
    world: carla.World, blueprint_library: carla.BlueprintLibrary
) -> Tuple[carla.Actor, carla.WalkerAIController]:
    walker_bp = find_blueprint(
        blueprint_library,
        "walker.pedestrian.0010",
        "walker.pedestrian.*",
    )
    for _ in range(15):
        transform = carla.Transform(
            get_random_nav_location(world),
            carla.Rotation(yaw=random.uniform(-180.0, 180.0)),
        )
        walker = world.try_spawn_actor(walker_bp, transform)
        if walker is not None:
            break
    else:
        raise RuntimeError("Failed to spawn walker after multiple attempts.")

    controller_bp = blueprint_library.find("controller.ai.walker")
    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
    controller.start()
    destination = get_random_nav_location(world)
    controller.go_to_location(destination)
    controller.set_max_speed(random.uniform(0.9, 1.4))
    return walker, controller


def spawn_cyclist(
    world: carla.World, blueprint_library: carla.BlueprintLibrary
) -> Tuple[carla.Actor, carla.WalkerAIController]:
    bicycle_bp = find_blueprint(
        blueprint_library,
        "vehicle.bh.crossbike",
        "vehicle.diamondback.century",
        "walker.bicycle.*",
        "vehicle.*bike*",
    )
    for _ in range(15):
        transform = carla.Transform(
            get_random_nav_location(world),
            carla.Rotation(yaw=random.uniform(-180.0, 180.0)),
        )
        cyclist = world.try_spawn_actor(bicycle_bp, transform)
        if cyclist is not None:
            break
    else:
        raise RuntimeError("Failed to spawn cyclist after multiple attempts.")

    controller_bp = blueprint_library.find("controller.ai.walker")
    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=cyclist)
    controller.start()
    destination = get_random_nav_location(world)
    controller.go_to_location(destination)
    controller.set_max_speed(random.uniform(1.2, 3.0))
    return cyclist, controller


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_arguments(argv or sys.argv[1:])

    if args.seed is not None:
        random.seed(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    actors: List[carla.Actor] = []
    controllers: List[carla.WalkerAIController] = []

    try:
        vehicle = spawn_vehicle(world, blueprint_library)
        actors.append(vehicle)

        walker, walker_controller = spawn_walker(world, blueprint_library)
        actors.append(walker)
        controllers.append(walker_controller)

        cyclist, cyclist_controller = spawn_cyclist(world, blueprint_library)
        actors.append(cyclist)
        controllers.append(cyclist_controller)

        time.sleep(args.duration)
    finally:
        for controller in controllers:
            try:
                controller.stop()
            except RuntimeError:
                pass
            finally:
                try:
                    controller.destroy()
                except RuntimeError:
                    pass
        for actor in actors:
            try:
                actor.destroy()
            except RuntimeError:
                pass


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
