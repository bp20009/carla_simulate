#!/usr/bin/env python3
"""Plot CARLA actor trajectories stored by vehicle_state_stream.py."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

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


def load_trajectories(
    csv_path: Path,
    allowed_kinds: Iterable[str] | None = None,
) -> Tuple[Dict[int, List[Point]], Dict[int, str]]:
    """Load trajectories grouped by actor ID."""
    allowed_prefixes = None
    if allowed_kinds:
        allowed_prefixes = tuple(f"{kind}." for kind in allowed_kinds)

    trajectories: Dict[int, List[Point]] = defaultdict(list)
    actor_types: Dict[int, str] = {}

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(_strip_null_bytes(fh))
        for row in reader:
            actor_type = row["type"]
            if allowed_prefixes and not actor_type.startswith(allowed_prefixes):
                continue

            actor_id = int(row["id"])
            frame = float(row["frame"])
            x = float(row["location_x"])
            y = float(row["location_y"])

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
    parser.add_argument("csv", type=Path, help="Path to vehicle_state_stream output CSV")
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

    trajectories, actor_types = load_trajectories(args.csv, args.only)
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
