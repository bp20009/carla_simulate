"""Measure CARLA time acceleration across actor counts and rendering modes.

This script connects to a running CARLA server, optionally disables rendering,
spawns a configurable number of autopilot vehicles, and measures how much
simulation time advances within a wall-clock budget. Results are printed and
optionally saved to CSV.

Example:
    python exp_future/measure_time_acceleration.py --duration 15 --output accel.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import carla


@dataclass
class BenchmarkResult:
    rendering: str
    requested_actors: int
    spawned_actors: int
    ticks: int
    sim_seconds: float
    real_seconds: float
    fixed_delta: float | None

    @property
    def speedup(self) -> float:
        return self.sim_seconds / self.real_seconds if self.real_seconds else 0.0


def parse_actor_counts(value: str) -> List[int]:
    if ":" in value:
        parts = [p.strip() for p in value.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError(
                "Range syntax must be start:end:step (e.g. 0:50:10)."
            )
        start, end, step = (int(p) for p in parts)
        if step <= 0:
            raise argparse.ArgumentTypeError("Step must be positive.")
        if end < start:
            raise argparse.ArgumentTypeError("End must be >= start.")
        return list(range(start, end + 1, step))

    counts = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        counts.append(int(item))
    if not counts:
        raise argparse.ArgumentTypeError("Provide at least one actor count.")
    return counts


def parse_arguments(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost", help="CARLA server host")
    parser.add_argument("--port", default=2000, type=int, help="CARLA server port")
    parser.add_argument(
        "--timeout", default=10.0, type=float, help="Client connection timeout"
    )
    parser.add_argument(
        "--duration",
        default=10.0,
        type=float,
        help="Wall-clock seconds to run each benchmark",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        help=(
            "Optional fixed delta seconds for synchronous mode. If omitted, "
            "the current world value is preserved."
        ),
    )
    parser.add_argument(
        "--actor-counts",
        type=parse_actor_counts,
        default=parse_actor_counts("0,10,20,30,40,50"),
        help="Comma-separated list (e.g. 0,10,20) or range start:end:step",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for deterministic spawn selection",
    )
    parser.add_argument(
        "--tm-port",
        type=int,
        default=8000,
        help="Traffic Manager port",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV output path",
    )
    parser.add_argument(
        "--warmup-ticks",
        type=int,
        default=1,
        help="Ticks to advance after spawning before timing",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--render-only",
        action="store_true",
        help="Run only the rendering-enabled benchmark",
    )
    group.add_argument(
        "--no-render-only",
        action="store_true",
        help="Run only the no-rendering benchmark",
    )
    return parser.parse_args(list(argv))


def apply_sync_settings(
    world: carla.World, no_rendering: bool, fixed_delta: float | None
) -> Tuple[carla.WorldSettings, carla.WorldSettings]:
    original = world.get_settings()

    # ベースは「現在設定のコピー」にする．新規 WorldSettings() は使わない．
    new_settings = world.get_settings()

    new_settings.synchronous_mode = True
    new_settings.no_rendering_mode = no_rendering

    # ベンチ用途なら固定を強く推奨．省略時も current を保持するか，デフォルト値を入れる．
    if fixed_delta is not None:
        new_settings.fixed_delta_seconds = fixed_delta
    else:
        # warning を避けたいならデフォルトを入れる（例: 0.05）
        # new_settings.fixed_delta_seconds = 0.05
        pass

    return original, new_settings


def spawn_autopilot_vehicles(
    world: carla.World,
    blueprint_library: carla.BlueprintLibrary,
    traffic_manager: carla.TrafficManager,
    count: int,
    rng: random.Random,
) -> List[carla.Actor]:
    if count <= 0:
        return []

    vehicle_blueprints = blueprint_library.filter("vehicle.*")
    if not vehicle_blueprints:
        raise RuntimeError("No vehicle blueprints available in this map.")

    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available in this map.")

    spawn_points = list(spawn_points)
    rng.shuffle(spawn_points)

    actors: List[carla.Actor] = []
    for transform in spawn_points:
        if len(actors) >= count:
            break
        blueprint = rng.choice(vehicle_blueprints)
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", "autopilot")
        actor = world.try_spawn_actor(blueprint, transform)
        if actor is None:
            continue
        actor.set_autopilot(True, traffic_manager.get_port())
        actors.append(actor)

    return actors


def run_benchmark(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    duration: float,
    no_rendering: bool,
    fixed_delta: float | None,
    actor_count: int,
    warmup_ticks: int,
    rng: random.Random,
) -> BenchmarkResult:
    original_settings, synced_settings = apply_sync_settings(
        world, no_rendering=no_rendering, fixed_delta=fixed_delta
    )
    world.apply_settings(synced_settings)
    traffic_manager.set_synchronous_mode(True)

    actors: List[carla.Actor] = []
    try:
        blueprint_library = world.get_blueprint_library()
        actors = spawn_autopilot_vehicles(
            world, blueprint_library, traffic_manager, actor_count, rng
        )

        for _ in range(max(warmup_ticks, 0)):
            world.tick()

        world.tick()
        start_snapshot = world.get_snapshot()
        start_elapsed = start_snapshot.timestamp.elapsed_seconds
        start_wall = time.perf_counter()

        ticks = 0
        latest_snapshot = start_snapshot
        while time.perf_counter() - start_wall < duration:
            world.tick()
            ticks += 1
            latest_snapshot = world.get_snapshot()

        end_elapsed = latest_snapshot.timestamp.elapsed_seconds
        sim_elapsed = end_elapsed - start_elapsed
        real_elapsed = time.perf_counter() - start_wall
    finally:
        for actor in actors:
            try:
                actor.destroy()
            except RuntimeError:
                pass
        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)

    return BenchmarkResult(
        rendering="no_rendering" if no_rendering else "rendering",
        requested_actors=actor_count,
        spawned_actors=len(actors),
        ticks=ticks,
        sim_seconds=sim_elapsed,
        real_seconds=real_elapsed,
        fixed_delta=fixed_delta,
    )


def print_results(results: Sequence[BenchmarkResult]) -> None:
    print(
        "Mode         | Actors(req/spawn) |   Sim [s] |  Real [s] | Ticks | Speedup"
    )
    print(
        "------------ | ----------------- | --------- | --------- | ----- | -------"
    )
    for result in results:
        print(
            f"{result.rendering:>12} | {result.requested_actors:3d}/{result.spawned_actors:3d} "
            f"         | {result.sim_seconds:9.2f} | {result.real_seconds:9.2f} "
            f"| {result.ticks:5d} | {result.speedup:7.2f}x"
        )


def write_csv(results: Sequence[BenchmarkResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rendering",
                "requested_actors",
                "spawned_actors",
                "ticks",
                "sim_seconds",
                "real_seconds",
                "speedup",
                "fixed_delta",
            ]
        )
        for result in results:
            writer.writerow(
                [
                    result.rendering,
                    result.requested_actors,
                    result.spawned_actors,
                    result.ticks,
                    f"{result.sim_seconds:.6f}",
                    f"{result.real_seconds:.6f}",
                    f"{result.speedup:.6f}",
                    "" if result.fixed_delta is None else f"{result.fixed_delta:.6f}",
                ]
            )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_arguments(argv or sys.argv[1:])

    rng = random.Random(args.seed)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)

    render_modes: List[bool] = []
    if not args.no_render_only:
        render_modes.append(False)
    if not args.render_only:
        render_modes.append(True)

    results: List[BenchmarkResult] = []
    for no_rendering in render_modes:
        for actor_count in args.actor_counts:
            results.append(
                run_benchmark(
                    world,
                    traffic_manager,
                    duration=args.duration,
                    no_rendering=no_rendering,
                    fixed_delta=args.fixed_delta,
                    actor_count=actor_count,
                    warmup_ticks=args.warmup_ticks,
                    rng=rng,
                )
            )

    print_results(results)

    if args.output:
        write_csv(results, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
