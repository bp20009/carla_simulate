#!/usr/bin/env python3
"""Create a video animation of CARLA actor trajectories from vehicle_state_stream.py CSV output."""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.cm import get_cmap

from plot_vehicle_trajectories import Point, load_trajectories


@dataclass
class ActorSeries:
    actor_id: int
    label: str
    frames: Sequence[float]
    xs: Sequence[float]
    ys: Sequence[float]
    line: Any  # matplotlib Line2D
    marker: Any  # matplotlib Line2D representing the current position


def prepare_actor_series(
    trajectories: Dict[int, List[Point]],
    actor_types: Dict[int, str],
    ax: plt.Axes,
):
    cmap = get_cmap("tab20")
    series: List[ActorSeries] = []

    for idx, (actor_id, points) in enumerate(sorted(trajectories.items())):
        frames = [pt[0] for pt in points]
        xs = [pt[1] for pt in points]
        ys = [pt[2] for pt in points]
        color = cmap(idx % cmap.N)
        label = f"{actor_types[actor_id]} (id={actor_id})"

        (line,) = ax.plot([], [], color=color, linewidth=1.5, label=label)
        (marker,) = ax.plot([], [], color=color, marker="o", markersize=4)

        series.append(
            ActorSeries(
                actor_id=actor_id,
                label=label,
                frames=frames,
                xs=xs,
                ys=ys,
                line=line,
                marker=marker,
            )
        )

    return series


def compute_frame_sequence(trajectories: Dict[int, List[Point]]) -> List[float]:
    frame_values = {frame for points in trajectories.values() for frame, *_ in points}
    if not frame_values:
        raise ValueError("No frames available to animate.")
    return sorted(frame_values)


def configure_axes(ax: plt.Axes, trajectories: Dict[int, List[Point]], title: str) -> None:
    xs = [pt[1] for points in trajectories.values() for pt in points]
    ys = [pt[2] for points in trajectories.values() for pt in points]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    x_span = xmax - xmin
    y_span = ymax - ymin
    margin = 0.05

    if x_span == 0:
        x_span = 1.0
    if y_span == 0:
        y_span = 1.0

    ax.set_xlim(xmin - margin * x_span, xmax + margin * x_span)
    ax.set_ylim(ymin - margin * y_span, ymax + margin * y_span)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title)


def animate_trajectories(
    csv_path: Path,
    output_path: Path,
    allowed_kinds: Iterable[str] | None,
    fps: int,
    history: int | None,
    dpi: int,
    title: str,
) -> None:
    trajectories, actor_types, _ = load_trajectories(csv_path, allowed_kinds)
    if not trajectories:
        raise SystemExit("No trajectories matched the provided filters.")

    frame_sequence = compute_frame_sequence(trajectories)

    fig, ax = plt.subplots(figsize=(10, 8))
    configure_axes(ax, trajectories, title)

    series = prepare_actor_series(trajectories, actor_types, ax)
    ax.legend(loc="upper right", fontsize=8)
    frame_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.6),
    )

    def init():
        artists = []
        for actor in series:
            actor.line.set_data([], [])
            actor.marker.set_data([], [])
            artists.extend([actor.line, actor.marker])
        frame_text.set_text("")
        artists.append(frame_text)
        return artists

    def update(frame_index: int):
        frame_value = frame_sequence[frame_index]
        artists = []

        for actor in series:
            idx = bisect.bisect_right(actor.frames, frame_value)
            if idx == 0:
                actor.line.set_data([], [])
                actor.marker.set_data([], [])
            else:
                start = max(0, idx - history) if history is not None else 0
                actor.line.set_data(actor.xs[start:idx], actor.ys[start:idx])
                actor.marker.set_data([actor.xs[idx - 1]], [actor.ys[idx - 1]])
            artists.extend([actor.line, actor.marker])

        frame_text.set_text(f"frame: {frame_value:g}")
        artists.append(frame_text)
        return artists

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_sequence),
        init_func=init,
        interval=1000 / fps,
        blit=True,
        repeat=False,
    )

    fig.tight_layout()
    anim.save(output_path, fps=fps, dpi=dpi)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Path to vehicle_state_stream output CSV")
    parser.add_argument(
        "output",
        type=Path,
        help="Video file to write (e.g. animation.mp4 or animation.gif).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("vehicle", "walker"),
        help="Animate only the selected actor categories.",
    )
    parser.add_argument("--fps", type=int, default=20, help="Frames per second for the video.")
    parser.add_argument(
        "--history",
        type=int,
        default=None,
        help="Number of recent points to keep in the trail (default: full history).",
    )
    parser.add_argument("--dpi", type=int, default=150, help="Dots per inch for the saved video.")
    parser.add_argument("--title", default="CARLA trajectories", help="Figure title.")
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.history is not None and args.history < 0:
        parser.error("--history must be zero or positive")

    try:
        animate_trajectories(
            csv_path=args.csv,
            output_path=args.output,
            allowed_kinds=args.only,
            fps=args.fps,
            history=args.history,
            dpi=args.dpi,
            title=args.title,
        )
    except Exception as exc:  # pragma: no cover - CLI convenience
        parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
