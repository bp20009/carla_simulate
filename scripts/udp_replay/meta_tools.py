import argparse
import csv
import json
import sys
from pathlib import Path


def load_meta(meta_path: str) -> dict:
    try:
        return json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_accident_payload_frames(meta: dict):
    accidents = meta.get("accidents") or []
    for accident in accidents:
        pf = accident.get("payload_frame")
        if pf is None:
            continue
        try:
            yield int(pf)
        except (TypeError, ValueError):
            continue


def cmd_first_accident_pf(args: argparse.Namespace) -> int:
    """Return earliest accident payload_frame in meta.json (min)."""
    data = load_meta(args.meta_path)
    pfs = list(_iter_accident_payload_frames(data))
    if pfs:
        sys.stdout.write(str(min(pfs)))
    return 0


def cmd_first_accident_pf_after_switch(args: argparse.Namespace) -> int:
    """Return earliest accident payload_frame with pf >= switch_pf."""
    data = load_meta(args.meta_path)
    try:
        switch_pf = int(args.switch_pf)
    except ValueError:
        switch_pf = 0

    pfs = [pf for pf in _iter_accident_payload_frames(data) if pf >= switch_pf]
    if pfs:
        sys.stdout.write(str(min(pfs)))
    return 0


def cmd_switch_pf(args: argparse.Namespace) -> int:
    try:
        accident_pf = int(args.accident_pf)
        lead_sec = float(args.lead_sec)
        fixed_delta = float(args.fixed_delta)
    except ValueError:
        return 1
    frames = int(round(lead_sec / fixed_delta))
    switch_pf = max(accident_pf - frames, 0)
    sys.stdout.write(str(switch_pf))
    return 0


def cmd_accident_after_switch(args: argparse.Namespace) -> int:
    """Use the given switch_pf for evaluation (do NOT override)."""
    data = load_meta(args.meta_path)
    try:
        switch_pf = int(args.switch_pf)
    except ValueError:
        switch_pf = 0
    hit = 0
    for pf in _iter_accident_payload_frames(data):
        if pf >= switch_pf:
            hit = 1
            break
    sys.stdout.write(str(hit))
    return 0


def cmd_accident_after_observed_switch(args: argparse.Namespace) -> int:
    """Evaluate using switch_payload_frame_observed stored in meta.json."""
    data = load_meta(args.meta_path)
    observed_switch_pf = data.get("switch_payload_frame_observed")
    try:
        switch_pf = int(observed_switch_pf)
    except (TypeError, ValueError):
        switch_pf = 0
    hit = 0
    for pf in _iter_accident_payload_frames(data):
        if pf >= switch_pf:
            hit = 1
            break
    sys.stdout.write(str(hit))
    return 0


def cmd_accident_pf_from_collisions(args: argparse.Namespace) -> int:
    try:
        with Path(args.collisions_path).open(
            encoding="utf-8", errors="ignore", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            min_frame = None
            for row in reader:
                if str(row.get("is_accident", "")).strip() != "1":
                    continue

                frame_value = row.get("payload_frame")
                if frame_value is None or str(frame_value).strip() == "":
                    frame_value = row.get("carla_frame")
                if frame_value is None or str(frame_value).strip() == "":
                    frame_value = row.get("frame")
                try:
                    frame_int = int(float(frame_value))
                except (TypeError, ValueError):
                    continue
                if min_frame is None or frame_int < min_frame:
                    min_frame = frame_int
    except OSError:
        return 1
    if min_frame is not None:
        sys.stdout.write(str(min_frame))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    first_accident = subparsers.add_parser("first_accident_pf")
    first_accident.add_argument("meta_path")
    first_accident.set_defaults(func=cmd_first_accident_pf)

    first_accident_after_switch = subparsers.add_parser(
        "first_accident_pf_after_switch"
    )
    first_accident_after_switch.add_argument("meta_path")
    first_accident_after_switch.add_argument("switch_pf")
    first_accident_after_switch.set_defaults(func=cmd_first_accident_pf_after_switch)

    switch_pf = subparsers.add_parser("switch_pf")
    switch_pf.add_argument("accident_pf")
    switch_pf.add_argument("lead_sec")
    switch_pf.add_argument("fixed_delta")
    switch_pf.set_defaults(func=cmd_switch_pf)

    accident_after_switch = subparsers.add_parser("accident_after_switch")
    accident_after_switch.add_argument("meta_path")
    accident_after_switch.add_argument("switch_pf")
    accident_after_switch.set_defaults(func=cmd_accident_after_switch)

    accident_after_observed_switch = subparsers.add_parser(
        "accident_after_observed_switch"
    )
    accident_after_observed_switch.add_argument("meta_path")
    accident_after_observed_switch.set_defaults(func=cmd_accident_after_observed_switch)

    accident_pf = subparsers.add_parser("accident_pf_from_collisions")
    accident_pf.add_argument("collisions_path")
    accident_pf.set_defaults(func=cmd_accident_pf_from_collisions)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
