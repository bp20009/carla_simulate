#!/usr/bin/env python3
"""Plot CARLA actor trajectories stored by vehicle_state_stream.py.

The script now accepts multiple CSV inputs. You can pass several paths
directly, or supply a directory/glob pattern to aggregate trajectories across
files. When merging runs, the script prefers the ``carla_actor_id`` column when
present (falling back to ``id``) to avoid conflicts from per-file numbering::

    python plot_vehicle_trajectories.py run1.csv run2.csv
    python plot_vehicle_trajectories.py --dir logs/
    python plot_vehicle_trajectories.py --glob "logs/*.csv"

For reproducible figures with simplified styling suitable for publications,
combine ``--paper`` with ``--save`` to write the output directly to disk::

    python plot_vehicle_trajectories.py --dir logs/ --paper --save trajectories.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, TextIO, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


Point = Tuple[float, float, float]  # (frame, x, y)
TrajectoryKey = Tuple[Path, int]


def _strip_null_bytes(lines: Iterable[str]) -> Iterator[str]:
    """Yield each line with embedded NUL characters removed.

    Some CARLA logs can contain stray ``\x00`` bytes which cause ``csv`` to
    raise ``Error: line contains NULL byte``.  The CSV content is otherwise
    valid, so we filter the characters on the fly while keeping streaming
    behavior to avoid loading the entire file into memory.
    """

    for line in lines:
        yield line.replace("\x00", "")


def _open_with_fallback(csv_path: Path, newline: str = "") -> TextIO:
    """Open a CSV file using the first compatible encoding.

    Attempts UTF-8 (with BOM handling) first, then falls back to UTF-16
    variants if necessary, preserving newline handling for ``csv``.
    """

    encodings = ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be")
    last_error: UnicodeError | None = None

    for encoding in encodings:
        try:
            return csv_path.open("r", encoding=encoding, newline=newline)
        except UnicodeError as exc:
            last_error = exc

    if last_error:
        raise last_error


def load_trajectories(
    csv_paths: Iterable[Path],
    allowed_kinds: Iterable[str] | None = None,
):
    """Load trajectories from one or more vehicle_state_stream CSV files.

    Supports UTF-8 (including BOM) and UTF-16 encodings to accommodate
    different CARLA logging configurations.
    """
    allowed_prefixes = None
    if allowed_kinds:
        allowed_prefixes = tuple(f"{kind}." for kind in allowed_kinds)

    trajectories: defaultdict[TrajectoryKey, List[Point]] = defaultdict(list)
    actor_types: Dict[TrajectoryKey, str] = {}

    for csv_path in csv_paths:
        with _open_with_fallback(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)

            for row in reader:
                actor_type = row["type"]
                if allowed_prefixes and not actor_type.startswith(allowed_prefixes):
                    continue

                actor_identifier = row.get("carla_actor_id") or row["id"]
                actor_id = int(actor_identifier)
                traj_key: TrajectoryKey = (csv_path, actor_id)
                frame = float(row["frame"])
                x = float(row["location_x"])
                y = float(row["location_y"])

                if traj_key in actor_types and actor_types[traj_key] != actor_type:
                    raise ValueError(
                        f"Actor id {actor_id} has inconsistent types: "
                        f"'{actor_types[traj_key]}' vs '{actor_type}' in {csv_path}"
                    )

                trajectories[traj_key].append((frame, x, y))
                actor_types[traj_key] = actor_type

    for points in trajectories.values():
        points.sort(key=lambda item: item[0])

    return trajectories, actor_types



def plot_trajectories(
    trajectories: Dict[TrajectoryKey, List[Point]],
    actor_types: Dict[TrajectoryKey, str],
    show_ids: bool,
    mark_endpoints: bool,
    title: str,
    paper: bool,
):
    """Draw each trajectory on a shared XY plane."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = get_cmap("tab20")
    base_color = cmap(1)  # slightly darker for better contrast in paper mode

    effective_show_ids = show_ids and not paper
    effective_mark_endpoints = mark_endpoints and not paper

    for idx, (traj_key, points) in enumerate(sorted(trajectories.items())):
        csv_path, actor_id = traj_key
        xs = [pt[1] for pt in points]
        ys = [pt[2] for pt in points]
        color = base_color if paper else cmap(idx % cmap.N)
        label = f"{csv_path.name}: {actor_types[traj_key]} (id={actor_id})"

        ax.plot(
            xs,
            ys,
            color=color,
            label=label if not paper else None,
            linewidth=1.8 if paper else 1.5,
            alpha=0.8 if paper else 1.0,
        )

        if effective_mark_endpoints:
            ax.scatter(xs[0], ys[0], color=color, marker="o", s=30, edgecolors="white")
            ax.scatter(xs[-1], ys[-1], color=color, marker="X", s=50, edgecolors="black")

        if effective_show_ids:
            ax.text(
                xs[-1],
                ys[-1],
                f"{csv_path.name}:{actor_id}",
                color=color,
                fontsize=8,
            )

    if not paper:
        ax.set_title(title)
    label_fontsize = 22 if paper else None
    tick_fontsize = 20 if paper else None

    ax.set_xlabel("X [m]", fontsize=label_fontsize)
    ax.set_ylabel("Y [m]", fontsize=label_fontsize)
    if tick_fontsize:
        ax.tick_params(labelsize=tick_fontsize)
        ax.tick_params(direction="in")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    if not paper:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig, ax


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        type=Path,
        nargs="*",
        help="One or more vehicle_state_stream output CSVs.",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        dest="csv_dir",
        help="Directory containing CSV logs to combine (non-recursive).",
    )
    parser.add_argument(
        "--glob",
        dest="csv_glob",
        help="Glob pattern (e.g., 'logs/*.csv') for CSV files to include.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("vehicle", "walker"),
        help="Plot only the selected actor categories.",
    )
    parser.add_argument("--hide-ids", action="store_true", help="Suppress id text labels.")
    parser.add_argument(
        "--no-endpoints",
        action="store_true",
        help="Do not mark start/end points of each trajectory.",
    )
    parser.add_argument("--title", default="CARLA trajectories", help="Figure title.")
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path to save the figure instead of opening an interactive window.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=(
            "Use simplified styling (single color, no labels) for publication-ready images; "
            "combine with --save for reproducible figures."
        ),
    )
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.paper:
        mpl.rcParams["font.family"] = "Times New Roman"

    csv_paths = list(args.csv)

    if args.csv_dir:
        csv_paths.extend(sorted(args.csv_dir.glob("*.csv")))

    if args.csv_glob:
        csv_paths.extend(sorted(Path().glob(args.csv_glob)))

    if not csv_paths:
        parser.error("Provide at least one CSV via positional args, --dir, or --glob.")

    trajectories, actor_types = load_trajectories(csv_paths, args.only)
    if not trajectories:
        parser.error("No trajectories matched the provided filters.")

    fig, _ = plot_trajectories(
        trajectories,
        actor_types,
        show_ids=not args.hide_ids,
        mark_endpoints=not args.no_endpoints,
        title=args.title,
        paper=args.paper,
    )

    if args.save:
        fig.savefig(args.save, dpi=200)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
