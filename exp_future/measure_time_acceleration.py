from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import carla

SCRIPT_VERSION = "20260203_repeat_median_autopilot"


@dataclass
class TrialResult:
    rendering: str
    requested_actors: int
    spawned_actors: int
    trial: int
    ticks: int
    sim_seconds: float
    real_seconds: float
    fixed_delta_actual: float | None
    no_rendering_applied: bool

    @property
    def speedup(self) -> float:
        return self.sim_seconds / self.real_seconds if self.real_seconds else 0.0


@dataclass
class SummaryResult:
    rendering: str
    requested_actors: int
    spawned_actors_median: float
    ticks_median: float
    sim_seconds_median: float
    real_seconds_median: float
    speedup_median: float
    fixed_delta_actual: float | None
    applied_no_render_all_true: bool


def parse_actor_counts(value: str) -> List[int]:
    if ":" in value:
        parts = [p.strip() for p in value.split(":")]
        if len(parts) != 3:
            raise argparse.ArgumentTypeError("Range syntax must be start:end:step (e.g. 0:50:10).")
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", default=2000, type=int)
    p.add_argument("--timeout", default=10.0, type=float)
    p.add_argument("--duration", default=10.0, type=float)
    p.add_argument("--fixed-delta", type=float, required=True)
    p.add_argument("--actor-counts", type=parse_actor_counts, default=parse_actor_counts("0,10,20,30,40,50"))
    p.add_argument("--seed", type=int, default=0, help="Base seed for deterministic spawns.")
    p.add_argument("--tm-port", type=int, default=8000)
    p.add_argument("--output", type=Path, help="Raw CSV output path (all trials).")
    p.add_argument(
        "--summary-output",
        type=Path,
        help="Optional summary CSV output path (median per condition). If omitted and --output is set, '<output>_summary.csv' is used.",
    )
    p.add_argument("--warmup-ticks", type=int, default=1)
    p.add_argument("--repeat", type=int, default=3, help="Number of trials per condition (median is reported).")
    p.add_argument("--debug-steps", action="store_true")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--render-only", action="store_true")
    g.add_argument("--no-render-only", action="store_true")
    return p.parse_args(list(argv))


def log_step(enabled: bool, msg: str) -> None:
    if enabled:
        print(f"[STEP] {msg}", flush=True)


def apply_sync_settings(world: carla.World, no_rendering: bool, fixed_delta: float) -> Tuple[carla.WorldSettings, carla.WorldSettings]:
    original = world.get_settings()
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.no_rendering_mode = no_rendering
    new_settings.fixed_delta_seconds = fixed_delta

    # substepping safety
    if getattr(new_settings, "substepping", False):
        msdt = getattr(new_settings, "max_substep_delta_time", 0.01) or 0.01
        mss = getattr(new_settings, "max_substeps", 10) or 10
        if msdt * mss < fixed_delta:
            needed = int((fixed_delta / msdt) + 0.999999)
            new_settings.max_substeps = max(mss, needed)

    return original, new_settings


def spawn_vehicles(world: carla.World, blueprint_library: carla.BlueprintLibrary, count: int, rng: random.Random) -> List[carla.Actor]:
    if count <= 0:
        return []
    bps = blueprint_library.filter("vehicle.*")
    if not bps:
        raise RuntimeError("No vehicle blueprints available in this map.")
    spawn_points = list(world.get_map().get_spawn_points())
    if not spawn_points:
        raise RuntimeError("No spawn points available in this map.")

    rng.shuffle(spawn_points)

    actors: List[carla.Actor] = []
    for tr in spawn_points:
        if len(actors) >= count:
            break
        bp = rng.choice(bps)
        if bp.has_attribute("role_name"):
            bp.set_attribute("role_name", "autopilot")
        a = world.try_spawn_actor(bp, tr)
        if a is None:
            continue
        actors.append(a)
    return actors


def enable_autopilot(actors: List[carla.Actor], traffic_manager: carla.TrafficManager) -> None:
    tm_port = traffic_manager.get_port()
    for a in actors:
        a.set_autopilot(True, tm_port)


def disable_autopilot(actors: List[carla.Actor]) -> None:
    for a in actors:
        try:
            a.set_autopilot(False)
        except RuntimeError:
            pass


def destroy_actors_batch_sync(client: carla.Client, actors: List[carla.Actor]) -> None:
    if not actors:
        return
    cmds = [carla.command.DestroyActor(a.id) for a in actors]
    resps = client.apply_batch_sync(cmds, True)
    errs = [r.error for r in resps if r.error]
    if errs:
        raise RuntimeError("DestroyActor batch errors: " + " | ".join(errs))


def check_applied_settings(world: carla.World) -> bool:
    s = world.get_settings()
    print(
        f"[CHECK] applied no_rendering_mode={getattr(s, 'no_rendering_mode', None)} "
        f"sync={getattr(s, 'synchronous_mode', None)} fixed_delta={getattr(s, 'fixed_delta_seconds', None)}",
        flush=True,
    )
    return bool(getattr(s, "no_rendering_mode", False))


def run_trial(
    client: carla.Client,
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    duration: float,
    no_rendering: bool,
    fixed_delta: float,
    actor_count: int,
    warmup_ticks: int,
    rng: random.Random,
    debug_steps: bool,
    trial_idx: int,
) -> TrialResult:
    original_settings, synced_settings = apply_sync_settings(world, no_rendering, fixed_delta)

    actors: List[carla.Actor] = []
    tm_sync_enabled = False
    no_rendering_applied = False

    try:
        log_step(debug_steps, "apply_settings(sync)")
        world.apply_settings(synced_settings)

        # confirm applied
        no_rendering_applied = check_applied_settings(world)

        log_step(debug_steps, "tick_after_settings")
        world.tick()

        if actor_count > 0:
            log_step(debug_steps, "tm_sync_on")
            traffic_manager.set_synchronous_mode(True)
            tm_sync_enabled = True

            log_step(debug_steps, "tick_after_tm_sync")
            world.tick()

        log_step(debug_steps, f"spawn_vehicles(count={actor_count})")
        actors = spawn_vehicles(world, world.get_blueprint_library(), actor_count, rng)

        log_step(debug_steps, "tick_after_spawn")
        world.tick()

        if actors:
            log_step(debug_steps, "enable_autopilot")
            enable_autopilot(actors, traffic_manager)

            log_step(debug_steps, "tick_after_autopilot")
            world.tick()

        for _ in range(max(warmup_ticks, 0)):
            log_step(debug_steps, "warmup_tick")
            world.tick()

        log_step(debug_steps, "start_timing")
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
        fixed_delta_actual = world.get_settings().fixed_delta_seconds

        return TrialResult(
            rendering="no_rendering" if no_rendering else "rendering",
            requested_actors=actor_count,
            spawned_actors=len(actors),
            trial=trial_idx,
            ticks=ticks,
            sim_seconds=sim_elapsed,
            real_seconds=real_elapsed,
            fixed_delta_actual=fixed_delta_actual,
            no_rendering_applied=no_rendering_applied,
        )

    finally:
        # cleanup
        log_step(debug_steps, "cleanup_disable_autopilot")
        disable_autopilot(actors)

        log_step(debug_steps, "cleanup_tick_before_destroy")
        try:
            world.tick()
        except RuntimeError:
            pass

        log_step(debug_steps, "cleanup_destroy_actors_batch_sync")
        try:
            destroy_actors_batch_sync(client, actors)
        except RuntimeError:
            pass

        log_step(debug_steps, "cleanup_tick_after_destroy")
        try:
            world.tick()
        except RuntimeError:
            pass

        if tm_sync_enabled:
            log_step(debug_steps, "tm_sync_off")
            try:
                traffic_manager.set_synchronous_mode(False)
            except RuntimeError:
                pass

        log_step(debug_steps, "restore_world_settings")
        try:
            world.apply_settings(original_settings)
        except RuntimeError:
            pass

        log_step(debug_steps, "tick_after_restore")
        try:
            world.tick()
        except RuntimeError:
            pass


def median(values: Sequence[float]) -> float:
    # statistics.median handles even/odd
    return float(statistics.median(values))


def summarize(trials: Sequence[TrialResult]) -> List[SummaryResult]:
    grouped: Dict[Tuple[str, int], List[TrialResult]] = {}
    for t in trials:
        grouped.setdefault((t.rendering, t.requested_actors), []).append(t)

    summaries: List[SummaryResult] = []
    for (rendering, actor_count), ts in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        spawned = median([float(x.spawned_actors) for x in ts])
        ticks = median([float(x.ticks) for x in ts])
        sim_s = median([x.sim_seconds for x in ts])
        real_s = median([x.real_seconds for x in ts])
        spd = median([x.speedup for x in ts])

        # fixed_delta_actual should be identical; keep the first non-None
        fd = next((x.fixed_delta_actual for x in ts if x.fixed_delta_actual is not None), None)

        # applied flag sanity: expect all True for no_rendering, all False for rendering
        applied_all_true = all(x.no_rendering_applied for x in ts)

        summaries.append(
            SummaryResult(
                rendering=rendering,
                requested_actors=actor_count,
                spawned_actors_median=spawned,
                ticks_median=ticks,
                sim_seconds_median=sim_s,
                real_seconds_median=real_s,
                speedup_median=spd,
                fixed_delta_actual=fd,
                applied_no_render_all_true=applied_all_true,
            )
        )

    return summaries


def print_summary_table(summaries: Sequence[SummaryResult]) -> None:
    print(
        "Mode         | Actors(req) | Spawned_med | Sim_med[s] | Real_med[s] | Ticks_med | Speedup_med | fixed_delta | applied_all_true"
    )
    print(
        "------------ | ---------- | ---------- | ---------- | ----------- | --------- | ----------- | ---------- | ---------------"
    )
    for s in summaries:
        fd = "" if s.fixed_delta_actual is None else f"{s.fixed_delta_actual:.6f}"
        print(
            f"{s.rendering:>12} | {s.requested_actors:10d} | {s.spawned_actors_median:10.1f} | "
            f"{s.sim_seconds_median:10.2f} | {s.real_seconds_median:11.2f} | {s.ticks_median:9.1f} | "
            f"{s.speedup_median:11.2f}x | {fd:>10} | {str(s.applied_no_render_all_true):>15}"
        )


def write_raw_csv(trials: Sequence[TrialResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rendering",
                "requested_actors",
                "spawned_actors",
                "trial",
                "ticks",
                "sim_seconds",
                "real_seconds",
                "speedup",
                "fixed_delta_actual",
                "no_rendering_applied",
            ]
        )
        for t in trials:
            w.writerow(
                [
                    t.rendering,
                    t.requested_actors,
                    t.spawned_actors,
                    t.trial,
                    t.ticks,
                    f"{t.sim_seconds:.6f}",
                    f"{t.real_seconds:.6f}",
                    f"{t.speedup:.6f}",
                    "" if t.fixed_delta_actual is None else f"{t.fixed_delta_actual:.6f}",
                    str(t.no_rendering_applied),
                ]
            )


def write_summary_csv(summaries: Sequence[SummaryResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rendering",
                "requested_actors",
                "spawned_actors_median",
                "sim_seconds_median",
                "real_seconds_median",
                "ticks_median",
                "speedup_median",
                "fixed_delta_actual",
                "applied_no_render_all_true",
            ]
        )
        for s in summaries:
            w.writerow(
                [
                    s.rendering,
                    s.requested_actors,
                    f"{s.spawned_actors_median:.1f}",
                    f"{s.sim_seconds_median:.6f}",
                    f"{s.real_seconds_median:.6f}",
                    f"{s.ticks_median:.1f}",
                    f"{s.speedup_median:.6f}",
                    "" if s.fixed_delta_actual is None else f"{s.fixed_delta_actual:.6f}",
                    str(s.applied_no_render_all_true),
                ]
            )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_arguments(argv or sys.argv[1:])

    print(f"[INFO] script_version={SCRIPT_VERSION}", flush=True)
    print(f"[INFO] repeat={args.repeat} duration={args.duration} fixed_delta={args.fixed_delta}", flush=True)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)

    render_modes: List[bool] = []
    if not args.no_render_only:
        render_modes.append(False)  # rendering
    if not args.render_only:
        render_modes.append(True)   # no_rendering

    trials: List[TrialResult] = []

    # Condition-level deterministic seed: make repeats comparable within each condition.
    # We vary by (mode, actor_count, trial) but keep stable base.
    for no_rendering in render_modes:
        for actor_count in args.actor_counts:
            for trial_idx in range(1, args.repeat + 1):
                # stable seed per condition+trial
                seed = (args.seed * 1000003) ^ (actor_count * 9176) ^ ((1 if no_rendering else 0) * 131071) ^ (trial_idx * 524287)
                rng = random.Random(seed)

                print(
                    f"[RUN] no_rendering={no_rendering} actors={actor_count} trial={trial_idx}/{args.repeat} seed={seed}",
                    flush=True,
                )

                trials.append(
                    run_trial(
                        client=client,
                        world=world,
                        traffic_manager=traffic_manager,
                        duration=args.duration,
                        no_rendering=no_rendering,
                        fixed_delta=args.fixed_delta,
                        actor_count=actor_count,
                        warmup_ticks=args.warmup_ticks,
                        rng=rng,
                        debug_steps=args.debug_steps,
                        trial_idx=trial_idx,
                    )
                )

    summaries = summarize(trials)
    print_summary_table(summaries)

    if args.output:
        write_raw_csv(trials, args.output)
        if args.summary_output is None:
            # e.g., foo.csv -> foo_summary.csv
            args.summary_output = args.output.with_name(args.output.stem + "_summary" + args.output.suffix)
        write_summary_csv(summaries, args.summary_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
