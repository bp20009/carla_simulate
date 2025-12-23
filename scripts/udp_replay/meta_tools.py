import argparse
import json
import sys
from pathlib import Path


def load_meta(meta_path: str) -> dict:
    try:
        return json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def cmd_first_accident_pf(args: argparse.Namespace) -> int:
    data = load_meta(args.meta_path)
    accidents = data.get("accidents") or []
    if accidents:
        pf = accidents[0].get("payload_frame", "")
        sys.stdout.write(str(pf))
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
    data = load_meta(args.meta_path)
    accidents = data.get("accidents") or []
    try:
        switch_pf = int(args.switch_pf)
    except ValueError:
        switch_pf = 0
    hit = 0
    for accident in accidents:
        payload_frame = accident.get("payload_frame")
        if payload_frame is None:
            continue
        try:
            if int(payload_frame) >= switch_pf:
                hit = 1
                break
        except (TypeError, ValueError):
            continue
    sys.stdout.write(str(hit))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    first_accident = subparsers.add_parser("first_accident_pf")
    first_accident.add_argument("meta_path")
    first_accident.set_defaults(func=cmd_first_accident_pf)

    switch_pf = subparsers.add_parser("switch_pf")
    switch_pf.add_argument("accident_pf")
    switch_pf.add_argument("lead_sec")
    switch_pf.add_argument("fixed_delta")
    switch_pf.set_defaults(func=cmd_switch_pf)

    accident_after_switch = subparsers.add_parser("accident_after_switch")
    accident_after_switch.add_argument("meta_path")
    accident_after_switch.add_argument("switch_pf")
    accident_after_switch.set_defaults(func=cmd_accident_after_switch)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
