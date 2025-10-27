"""Spawn autopilot vehicle, walker, and cyclist in a CARLA world."""

import random
import time
from typing import Tuple

import carla


def find_blueprint(blueprint_library: carla.BlueprintLibrary, *filters: str) -> carla.ActorBlueprint:
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


def spawn_vehicle(world: carla.World, blueprint_library: carla.BlueprintLibrary) -> carla.Actor:
    vehicle_bp = find_blueprint(
        blueprint_library,
        "vehicle.tesla.model3",
        "vehicle.lincoln.mkz_2017",
        "vehicle.*",
    )
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available for vehicles.")
    transform = random.choice(spawn_points)
    vehicle = world.spawn_actor(vehicle_bp, transform)
    vehicle.set_autopilot(True)
    return vehicle


def spawn_walker(
    world: carla.World, blueprint_library: carla.BlueprintLibrary
) -> Tuple[carla.Actor, carla.WalkerAIController]:
    walker_bp = find_blueprint(
        blueprint_library,
        "walker.pedestrian.0010",
        "walker.pedestrian.*",
    )
    transform = carla.Transform(get_random_nav_location(world))
    walker = world.spawn_actor(walker_bp, transform)

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
    transform = carla.Transform(get_random_nav_location(world))
    cyclist = world.spawn_actor(bicycle_bp, transform)

    controller_bp = blueprint_library.find("controller.ai.walker")
    controller = world.spawn_actor(controller_bp, carla.Transform(), attach_to=cyclist)
    controller.start()
    destination = get_random_nav_location(world)
    controller.go_to_location(destination)
    controller.set_max_speed(random.uniform(1.2, 3.0))
    return cyclist, controller


def main() -> None:
    client = carla.Client("localhost", 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprint_library = world.get_blueprint_library()

    actors = []
    controllers = []

    try:
        vehicle = spawn_vehicle(world, blueprint_library)
        actors.append(vehicle)

        walker, walker_controller = spawn_walker(world, blueprint_library)
        actors.append(walker)
        controllers.append(walker_controller)

        cyclist, cyclist_controller = spawn_cyclist(world, blueprint_library)
        actors.append(cyclist)
        controllers.append(cyclist_controller)

        time.sleep(30)
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


if __name__ == "__main__":
    main()
