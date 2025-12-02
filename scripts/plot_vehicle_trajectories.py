#!/usr/bin/env python3
"""Plot CARLA actor trajectories stored by vehicle_state_stream.py.

The script now accepts multiple CSV inputs. You can pass several paths
directly, or supply a directory/glob pattern to aggregate trajectories across
files. When merging runs, the script prefers the ``carla_actor_id`` column when
present (falling back to ``id``) to avoid conflicts from per-file numbering::

    python plot_vehicle_trajectories.py run1.csv run2.csv
    python plot_vehicle_trajectories.py --dir logs/
    python plot_vehicle_trajectories.py --glob "logs/*.csv"
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, TextIO, Tuple

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


Point = Tuple[float, float, float]  # (frame, x, y)


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

    trajectories = defaultdict(list)
    actor_types = {}

    for csv_path in csv_paths:
        with _open_with_fallback(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)

            for row in reader:
                actor_type = row["type"]
                if allowed_prefixes and not actor_type.startswith(allowed_prefixes):
                    continue

                actor_identifier = row.get("carla_actor_id") or row["id"]
                actor_id = int(actor_identifier)
                frame = float(row["frame"])
                x = float(row["location_x"])
                y = float(row["location_y"])

                if actor_id in actor_types and actor_types[actor_id] != actor_type:
                    raise ValueError(
                        f"Actor id {actor_id} has inconsistent types: "
                        f"'{actor_types[actor_id]}' vs '{actor_type}' in {csv_path}"
                    )

                trajectories[actor_id].append((frame, x, y))
                actor_types[actor_id] = actor_type

    for points in trajectories.values():
        points.sort(key=lambda item: item[0])

    return trajectories, actor_types



def plot_trajectories(
    trajectories: Dict[int, List[Point]],
    actor_types: Dict[int, str],
    show_ids: bool,
    mark_endpoints: bool,
    title: str,
):
    """Draw each trajectory on a shared XY plane."""
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = get_cmap("tab20")

    for idx, (actor_id, points) in enumerate(sorted(trajectories.items())):
        xs = [pt[1] for pt in points]
        ys = [pt[2] for pt in points]
        color = cmap(idx % cmap.N)
        label = f"{actor_types[actor_id]} (id={actor_id})"

        ax.plot(xs, ys, color=color, label=label, linewidth=1.5)

        if mark_endpoints:
            ax.scatter(xs[0], ys[0], color=color, marker="o", s=30, edgecolors="white")
            ax.scatter(xs[-1], ys[-1], color=color, marker="X", s=50, edgecolors="black")

        if show_ids:
            ax.text(xs[-1], ys[-1], str(actor_id), color=color, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
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
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

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
    )

    if args.save:
        fig.savefig(args.save, dpi=200)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
