"""Benchmark how much CARLA simulation time advances within a real-time budget.

The script connects to a CARLA server, optionally toggles rendering, and runs the
world in synchronous mode for a configurable wall-clock duration. For each run it
reports how many simulation seconds elapsed, the number of ticks executed, and
the average simulation speedup factor (simulation seconds per real second).

Example:
    python exp_future/time_acceleration_benchmark.py --duration 15

This will run two back-to-back benchmarks: first with rendering enabled, then
with rendering disabled. Use ``--render-only`` or ``--no-render-only`` to run a
single configuration.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import carla


@dataclass
class BenchmarkResult:
    mode: str
    sim_seconds: float
    real_seconds: float
    ticks: int

    @property
    def speedup(self) -> float:
        return self.sim_seconds / self.real_seconds if self.real_seconds else 0.0


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

    def _copy_if_exists(target: carla.WorldSettings, source: carla.WorldSettings, name: str) -> None:
        if hasattr(target, name) and hasattr(source, name):
            setattr(target, name, getattr(source, name))

    new_settings = carla.WorldSettings()
    # Preserve original tunables so only the relevant sync settings change.
    for field in (
        "substepping",
        "max_substep_delta_time",
        "max_substeps",
        "max_culling_distance",
        "deterministic_ragdolls",
        "tile_stream_distance",
        "actor_active_distance",
        "spectator_as_ego",
    ):
        _copy_if_exists(new_settings, original, field)

    new_settings.synchronous_mode = True
    new_settings.no_rendering_mode = no_rendering
    if fixed_delta is not None:
        new_settings.fixed_delta_seconds = fixed_delta
    else:
        _copy_if_exists(new_settings, original, "fixed_delta_seconds")

    return original, new_settings


def run_benchmark(
    world: carla.World, duration: float, no_rendering: bool, fixed_delta: float | None
) -> BenchmarkResult:
    original_settings, synced_settings = apply_sync_settings(
        world, no_rendering=no_rendering, fixed_delta=fixed_delta
    )
    world.apply_settings(synced_settings)

    try:
        world.tick()
        start_snapshot = world.get_snapshot()
        start_elapsed = start_snapshot.timestamp.elapsed_seconds
        start_wall = time.time()

        ticks = 0
        latest_snapshot = start_snapshot
        while time.time() - start_wall < duration:
            world.tick()
            ticks += 1
            latest_snapshot = world.get_snapshot()

        end_elapsed = latest_snapshot.timestamp.elapsed_seconds
        sim_elapsed = end_elapsed - start_elapsed
        real_elapsed = time.time() - start_wall
    finally:
        world.apply_settings(original_settings)

    return BenchmarkResult(
        mode="no_rendering" if no_rendering else "rendering",
        sim_seconds=sim_elapsed,
        real_seconds=real_elapsed,
        ticks=ticks,
    )


def format_result(result: BenchmarkResult) -> str:
    return (
        f"{result.mode:>12} | sim: {result.sim_seconds:8.2f}s | "
        f"real: {result.real_seconds:8.2f}s | ticks: {result.ticks:5d} | "
        f"speedup: {result.speedup:5.2f}x"
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_arguments(argv or sys.argv[1:])

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()

    results: list[BenchmarkResult] = []
    if not args.no_render_only:
        results.append(
            run_benchmark(
                world, duration=args.duration, no_rendering=False, fixed_delta=args.fixed_delta
            )
        )

    if not args.render_only:
        results.append(
            run_benchmark(
                world, duration=args.duration, no_rendering=True, fixed_delta=args.fixed_delta
            )
        )

    print("Mode         |    Simulated |      Real | Ticks | Speedup")
    print("------------ | ----------- | --------- | ----- | -------")
    for result in results:
        print(format_result(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
