"""Measure CARLA time acceleration across actor counts and rendering modes.

This script connects to a running CARLA server, optionally disables rendering,
spawns a configurable number of vehicles, enables autopilot, and measures how
much simulation time advances within a wall-clock budget. Results are printed
and optionally saved to CSV.

Example:
    python exp_future/measure_time_acceleration.py --duration 15 --fixed-delta 0.05 \
        --actor-counts 0:50:10 --output .\\results\\time_accel_benchmark.csv
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
    fixed_delta_actual: float | None

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

    counts: List[int] = []
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
    parser.add_argument("--timeout", default=10.0, type=float, help="Client timeout")
    parser.add_argument(
        "--duration",
        default=10.0,
        type=float,
        help="Wall-clock seconds to run each benchmark",
    )
    parser.add_argument(
        "--fixed-delta",
        type=float,
        help="Fixed delta seconds for synchronous mode (recommended).",
    )
    parser.add_argument(
        "--actor-counts",
        type=parse_actor_counts,
        default=parse_actor_counts("0,10,20,30,40,50"),
        help="Comma-separated list (e.g. 0,10,20) or range start:end:step",
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
    parser.add_argument("--output", type=Path, help="Optional CSV output path")
    parser.add_argument(
        "--warmup-ticks",
        type=int,
        default=1,
        help="Ticks to advance after autopilot enable before timing",
    )
    parser.add_argument(
        "--debug-steps",
        action="store_true",
        help="Print [STEP] logs for crash localization",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--render-only", action="store_true", help="Rendering only")
    group.add_argument("--no-render-only", action="store_true", help="No-render only")
    return parser.parse_args(list(argv))


def _log_step(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[STEP] {msg}", flush=True)


def apply_sync_settings(
    world: carla.World, no_rendering: bool, fixed_delta: float | None
) -> Tuple[carla.WorldSettings, carla.WorldSettings]:
    """Return (original_settings, new_sync_settings).

    IMPORTANT: Do NOT create a fresh carla.WorldSettings() here.
    Use world.get_settings() as a base to avoid undefined/unsupported fields
    that can trigger aborts in native code.
    """
    original = world.get_settings()
    new_settings = world.get_settings()

    new_settings.synchronous_mode = True
    new_settings.no_rendering_mode = no_rendering

    if fixed_delta is not None:
        new_settings.fixed_delta_seconds = fixed_delta

        # Guard: if substepping is enabled, ensure max_substep_delta_time * max_substeps >= fixed_delta
        if getattr(new_settings, "substepping", False):
            msdt = getattr(new_settings, "max_substep_delta_time", 0.01) or 0.01
            mss = getattr(new_settings, "max_substeps", 10) or 10
            if msdt * mss < fixed_delta:
                needed = int((fixed_delta / msdt) + 0.999999)
                new_settings.max_substeps = max(mss, needed)

    return original, new_settings


def spawn_vehicles(
    world: carla.World,
    blueprint_library: carla.BlueprintLibrary,
    count: int,
    rng: random.Random,
) -> List[carla.Actor]:
    if count <= 0:
        return []

    vehicle_blueprints = blueprint_library.filter("vehicle.*")
    if not vehicle_blueprints:
        raise RuntimeError("No vehicle blueprints available in this map.")

    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No spawn points available in this map.")

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

        actors.append(actor)

    return actors


def enable_autopilot(actors: List[carla.Actor], traffic_manager: carla.TrafficManager) -> None:
    tm_port = traffic_manager.get_port()
    for a in actors:
        a.set_autopilot(True, tm_port)


def run_benchmark(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    duration: float,
    no_rendering: bool,
    fixed_delta: float | None,
    actor_count: int,
    warmup_ticks: int,
    rng: random.Random,
    debug_steps: bool,
) -> BenchmarkResult:
    original_settings, synced_settings = apply_sync_settings(
        world, no_rendering=no_rendering, fixed_delta=fixed_delta
    )

    actors: List[carla.Actor] = []
    tm_sync_enabled = False

    try:
        _log_step(debug_steps, "apply_settings(sync)")
        world.apply_settings(synced_settings)

        # Ensure settings are applied before we start spawning/autopilot.
        _log_step(debug_steps, "tick_after_settings")
        world.tick()

        # Only enable TM sync when we actually use TM (actors > 0).
        if actor_count > 0:
            _log_step(debug_steps, "tm_sync_on")
            traffic_manager.set_synchronous_mode(True)
            tm_sync_enabled = True

            # One more tick after TM sync to stabilize.
            _log_step(debug_steps, "tick_after_tm_sync")
            world.tick()

        blueprint_library = world.get_blueprint_library()

        _log_step(debug_steps, f"spawn_vehicles(count={actor_count})")
        actors = spawn_vehicles(world, blueprint_library, actor_count, rng)

        _log_step(debug_steps, "tick_after_spawn")
        world.tick()

        if actors:
            _log_step(debug_steps, "enable_autopilot")
            enable_autopilot(actors, traffic_manager)

            _log_step(debug_steps, "tick_after_autopilot")
            world.tick()

        for _ in range(max(warmup_ticks, 0)):
            _log_step(debug_steps, "warmup_tick")
            world.tick()

        # Timing
        _log_step(debug_steps, "start_timing")
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

        # Record actual fixed delta in effect (not just the CLI arg).
        actual_fixed_delta = world.get_settings().fixed_delta_seconds

    finally:
        _log_step(debug_steps, "cleanup_destroy_actors")
        for actor in actors:
            try:
                actor.destroy()
            except RuntimeError:
                pass

        if tm_sync_enabled:
            _log_step(debug_steps, "tm_sync_off")
            try:
                traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass

        _log_step(debug_steps, "restore_world_settings")
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

    return BenchmarkResult(
        rendering="no_rendering" if no_rendering else "rendering",
        requested_actors=actor_count,
        spawned_actors=len(actors),
        ticks=ticks,
        sim_seconds=sim_elapsed,
        real_seconds=real_elapsed,
        fixed_delta_actual=actual_fixed_delta,
    )


def print_results(results: Sequence[BenchmarkResult]) -> None:
    print("Mode         | Actors(req/spawn) |   Sim [s] |  Real [s] | Ticks | Speedup | fixed_delta")
    print("------------ | ----------------- | --------- | --------- | ----- | ------- | ----------")
    for r in results:
        fd = "" if r.fixed_delta_actual is None else f"{r.fixed_delta_actual:.6f}"
        print(
            f"{r.rendering:>12} | {r.requested_actors:3d}/{r.spawned_actors:3d}         "
            f"| {r.sim_seconds:9.2f} | {r.real_seconds:9.2f} | {r.ticks:5d} | {r.speedup:7.2f}x | {fd}"
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
                "fixed_delta_actual",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.rendering,
                    r.requested_actors,
                    r.spawned_actors,
                    r.ticks,
                    f"{r.sim_seconds:.6f}",
                    f"{r.real_seconds:.6f}",
                    f"{r.speedup:.6f}",
                    "" if r.fixed_delta_actual is None else f"{r.fixed_delta_actual:.6f}",
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
        render_modes.append(False)  # rendering enabled
    if not args.render_only:
        render_modes.append(True)   # no_rendering

    results: List[BenchmarkResult] = []
    for no_rendering in render_modes:
        for actor_count in args.actor_counts:
            print(f"[RUN] no_rendering={no_rendering} actors={actor_count}", flush=True)
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
                    debug_steps=args.debug_steps,
                )
            )

    print_results(results)

    if args.output:
        write_csv(results, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
