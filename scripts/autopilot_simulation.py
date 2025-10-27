"""CARLA autopilot simulation with trajectory logging.

This module provides a runnable entry point that launches a CARLA autopilot
simulation, records the poses of interesting actors (vehicles, pedestrians and
bicycles) at every server tick, and persists those trajectories for later
inspection.  Optionally, the recorded trajectories can be plotted using
matplotlib for a quick visual validation.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple

import carla


LOGGER = logging.getLogger(__name__)


def actor_category(type_id: str) -> str:
    """Return a coarse actor category for the given CARLA type id.

    Parameters
    ----------
    type_id:
        The actor blueprint identifier reported by CARLA (for example
        ``"vehicle.tesla.model3"``).

    Returns
    -------
    str
        One of ``"vehicle"``, ``"pedestrian"``, ``"bicycle"`` or ``"other"``.
    """

    lowered = type_id.lower()
    if lowered.startswith("walker.pedestrian"):
        return "pedestrian"
    if lowered.startswith("vehicle."):
        if ".bicycle" in lowered or lowered.endswith("bike") or "bike" in lowered:
            return "bicycle"
        return "vehicle"
    return "other"


@dataclass
class TransformRecord:
    """Container that stores the pose of an actor at a given timestamp."""

    actor_id: int
    actor_category: str
    type_id: str
    timestamp: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    def to_csv_row(self) -> List[object]:
        """Return a CSV-compatible list preserving the record data."""

        return [
            self.timestamp,
            self.actor_id,
            self.actor_category,
            self.type_id,
            self.x,
            self.y,
            self.z,
            self.roll,
            self.pitch,
            self.yaw,
        ]

    def to_dict(self) -> Dict[str, object]:
        """Return a JSON-serialisable dictionary representation."""

        return asdict(self)


class TrajectoryLogger:
    """Collect and persist per-actor trajectory data."""

    CSV_HEADER: Sequence[str] = (
        "timestamp",
        "actor_id",
        "actor_category",
        "type_id",
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
    )

    def __init__(
        self,
        output_dir: Path,
        *,
        csv_filename: str = "trajectories.csv",
        json_filename: str = "trajectories.json",
        output_formats: Sequence[str] = ("csv", "json"),
    ) -> None:
        self._output_dir = Path(output_dir)
        self._csv_path = self._output_dir / csv_filename
        self._json_path = self._output_dir / json_filename
        self._output_formats = tuple(fmt.lower() for fmt in output_formats)
        self._records: List[TransformRecord] = []
        self._records_by_actor: DefaultDict[int, List[TransformRecord]] = defaultdict(list)
        self._start_time = datetime.utcnow()

    @property
    def records(self) -> Sequence[TransformRecord]:
        """Expose a read-only view over the recorded transforms."""

        return tuple(self._records)

    def log(self, actor: carla.Actor, transform: carla.Transform, timestamp: float) -> None:
        """Append a new record for the provided actor.

        Parameters
        ----------
        actor:
            The CARLA actor whose transform is being logged.
        transform:
            The transform sampled from the actor. The caller is responsible for
            aligning the call with CARLA's tick to guarantee temporal
            consistency.
        timestamp:
            Simulation time reported by CARLA.
        """

        category = actor_category(actor.type_id)
        record = TransformRecord(
            actor_id=actor.id,
            actor_category=category,
            type_id=actor.type_id,
            timestamp=timestamp,
            x=float(transform.location.x),
            y=float(transform.location.y),
            z=float(transform.location.z),
            roll=float(transform.rotation.roll),
            pitch=float(transform.rotation.pitch),
            yaw=float(transform.rotation.yaw),
        )
        self._records.append(record)
        self._records_by_actor[actor.id].append(record)

    def _ensure_output_dir(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)

    def write_files(self) -> Tuple[Path, ...]:
        """Persist the collected trajectories using the requested formats."""

        self._ensure_output_dir()
        written_files: List[Path] = []

        if "csv" in self._output_formats:
            with self._csv_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(self.CSV_HEADER)
                for record in self._records:
                    writer.writerow(record.to_csv_row())
            written_files.append(self._csv_path)
            LOGGER.info("CSV trajectory log written to %s", self._csv_path)

        if "json" in self._output_formats:
            serialised: Dict[str, Dict[str, object]] = {}
            for actor_id, records in self._records_by_actor.items():
                serialised[str(actor_id)] = {
                    "actor_category": records[0].actor_category if records else "unknown",
                    "type_id": records[0].type_id if records else "unknown",
                    "trajectory": [record.to_dict() for record in records],
                }
            with self._json_path.open("w", encoding="utf-8") as json_file:
                json.dump(
                    {
                        "created_utc": self._start_time.isoformat() + "Z",
                        "records": serialised,
                    },
                    json_file,
                    indent=2,
                )
            written_files.append(self._json_path)
            LOGGER.info("JSON trajectory log written to %s", self._json_path)

        return tuple(written_files)

    def plot_trajectories(self, output_path: Path | None = None, *, show: bool = False) -> None:
        """Visualise the recorded trajectories using matplotlib.

        Parameters
        ----------
        output_path:
            Optional location where the plot will be written.  If omitted, the
            figure is not saved to disk.
        show:
            When set to :data:`True`, display the plot in an interactive window
            after drawing it.  This requires an available GUI backend.
        """

        if not self._records_by_actor:
            LOGGER.warning("No trajectory data available to plot.")
            return

        import matplotlib.pyplot as plt  # Imported lazily to keep the dependency optional.

        colours = {
            "vehicle": "tab:blue",
            "pedestrian": "tab:orange",
            "bicycle": "tab:green",
            "other": "tab:gray",
        }

        fig, ax = plt.subplots(figsize=(10, 10))
        for actor_id, records in self._records_by_actor.items():
            xs = [record.x for record in records]
            ys = [record.y for record in records]
            category = records[0].actor_category if records else "other"
            colour = colours.get(category, "tab:gray")
            label = f"{category}:{actor_id}"
            ax.plot(xs, ys, marker=".", linestyle="-", linewidth=1.25, color=colour, label=label)

        ax.set_title("Actor trajectories")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", fontsize="small")

        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, bbox_inches="tight", dpi=200)
            LOGGER.info("Trajectory plot saved to %s", output_path)

        if show:
            plt.show()

        plt.close(fig)


def get_random_blueprint(blueprints: Sequence[carla.ActorBlueprint]) -> carla.ActorBlueprint:
    blueprint = random.choice(list(blueprints))
    if blueprint.has_attribute("color"):
        colour = random.choice(blueprint.get_attribute("color").recommended_values)
        blueprint.set_attribute("color", colour)
    if blueprint.has_attribute("driver_id"):
        driver_id = random.choice(blueprint.get_attribute("driver_id").recommended_values)
        blueprint.set_attribute("driver_id", driver_id)
    blueprint.set_attribute("role_name", "autopilot")
    return blueprint


def spawn_vehicles(
    world: carla.World,
    blueprint_library: carla.BlueprintLibrary,
    amount: int,
    spawn_points: Sequence[carla.Transform],
) -> List[carla.Vehicle]:
    vehicles: List[carla.Vehicle] = []
    available_blueprints = list(blueprint_library.filter("vehicle.*"))
    shuffled_spawns = list(spawn_points)
    random.shuffle(shuffled_spawns)

    if amount > len(shuffled_spawns):
        LOGGER.warning(
            "Requested %d vehicles but only %d spawn points are available",
            amount,
            len(shuffled_spawns),
        )

    for spawn_point in shuffled_spawns[:amount]:
        blueprint = get_random_blueprint(available_blueprints)
        vehicle = world.try_spawn_actor(blueprint, spawn_point)
        if vehicle:
            vehicles.append(vehicle)
        else:
            LOGGER.debug("Failed to spawn vehicle at %s", spawn_point)

    return vehicles


def configure_world(
    world: carla.World, *, synchronous: bool, fps: float | None
) -> Tuple[carla.WorldSettings, carla.WorldSettings]:
    current_settings = world.get_settings()
    new_settings = carla.WorldSettings()
    new_settings.no_rendering_mode = current_settings.no_rendering_mode
    new_settings.synchronous_mode = synchronous
    new_settings.fixed_delta_seconds = (
        1.0 / fps if synchronous and fps else current_settings.fixed_delta_seconds
    )
    new_settings.substepping = current_settings.substepping
    new_settings.max_substeps = current_settings.max_substeps
    new_settings.max_substep_delta_time = current_settings.max_substep_delta_time
    return current_settings, new_settings


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CARLA autopilot simulation with trajectory logging.")
    parser.add_argument("--host", default="127.0.0.1", help="CARLA server hostname")
    parser.add_argument("--port", type=int, default=2000, help="CARLA server TCP port")
    parser.add_argument("--tm-port", type=int, default=8000, help="Traffic Manager port")
    parser.add_argument("--timeout", type=float, default=10.0, help="Client connection timeout in seconds")
    parser.add_argument("--vehicles", type=int, default=20, help="Number of autopilot vehicles to spawn")
    parser.add_argument("--duration", type=float, default=60.0, help="Simulation duration in seconds (0 runs indefinitely)")
    parser.add_argument("--fps", type=float, default=20.0, help="Target FPS when running in synchronous mode")
    parser.add_argument("--asynchronous", action="store_true", help="Disable synchronous mode (not recommended for logging)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Directory used for trajectory logs and plots",
    )
    parser.add_argument(
        "--output-formats",
        nargs="+",
        default=("csv", "json"),
        choices=("csv", "json"),
        help="File formats to emit after the simulation",
    )
    parser.add_argument(
        "--csv-name",
        default="trajectories.csv",
        help="Filename for the generated CSV log",
    )
    parser.add_argument(
        "--json-name",
        default="trajectories.json",
        help="Filename for the generated JSON log",
    )
    parser.add_argument(
        "--plot-trajectories",
        action="store_true",
        help="Render a matplotlib figure showing recorded trajectories",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=None,
        help="Optional path to save the trajectory plot (requires --plot-trajectories)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logging verbosity",
    )
    return parser.parse_args()


def collect_relevant_actors(actors: Iterable[carla.Actor]) -> List[carla.Actor]:
    relevant: List[carla.Actor] = []
    for actor in actors:
        category = actor_category(actor.type_id)
        if category in {"vehicle", "pedestrian", "bicycle"} and actor.is_alive:
            relevant.append(actor)
    return relevant


def main() -> None:
    args = parse_arguments()
    logging.basicConfig(level=getattr(logging, args.log_level))
    LOGGER.info("Connecting to CARLA at %s:%s", args.host, args.port)

    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(args.tm_port)

    synchronous = not args.asynchronous
    original_settings, desired_settings = configure_world(world, synchronous=synchronous, fps=args.fps)

    vehicles: List[carla.Vehicle] = []
    logger = TrajectoryLogger(
        args.output_dir,
        csv_filename=args.csv_name,
        json_filename=args.json_name,
        output_formats=args.output_formats,
    )

    try:
        if synchronous:
            LOGGER.info("Enabling synchronous mode at %.2f FPS", args.fps)
            world.apply_settings(desired_settings)
        else:
            LOGGER.warning(
                "Running in asynchronous mode. Trajectory timestamps may not align perfectly with the server ticks."
            )

        traffic_manager.set_synchronous_mode(synchronous)

        spawn_points = world.get_map().get_spawn_points()
        blueprint_library = world.get_blueprint_library()
        vehicles = spawn_vehicles(world, blueprint_library, args.vehicles, spawn_points)
        for vehicle in vehicles:
            vehicle.set_autopilot(True, traffic_manager.get_port())
        LOGGER.info("Spawned %d autopilot vehicles", len(vehicles))

        frame_limit = None
        if args.duration > 0 and args.fps:
            frame_limit = int(args.duration * args.fps)

        frame_count = 0
        LOGGER.info("Starting simulation loop")
        while frame_limit is None or frame_count < frame_limit:
            snapshot = world.wait_for_tick()
            timestamp = snapshot.timestamp.elapsed_seconds
            relevant_actors = collect_relevant_actors(world.get_actors())
            for actor in relevant_actors:
                transform = actor.get_transform()
                logger.log(actor, transform, timestamp)
            frame_count += 1

        LOGGER.info("Simulation complete: recorded %d frames", frame_count)

    except KeyboardInterrupt:
        LOGGER.info("Simulation interrupted by user")
    finally:
        written_files = logger.write_files()
        LOGGER.info("Generated trajectory files: %s", ", ".join(str(path) for path in written_files))

        if args.plot_trajectories:
            try:
                logger.plot_trajectories(args.plot_path, show=args.plot_path is None)
            except ModuleNotFoundError:
                LOGGER.warning("matplotlib is not installed; skipping trajectory plot")

        if synchronous:
            world.apply_settings(original_settings)
            traffic_manager.set_synchronous_mode(False)

        for vehicle in vehicles:
            if vehicle.is_alive:
                vehicle.destroy()

        LOGGER.info("Cleaned up simulation actors")


if __name__ == "__main__":
    main()

