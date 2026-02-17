import argparse
from pathlib import Path

from scripts.udp_replay import meta_tools


def test_cmd_switch_pf_outputs_expected_value(capsys) -> None:
    args = argparse.Namespace(accident_pf="250", lead_sec="3", fixed_delta="0.1")
    rc = meta_tools.cmd_switch_pf(args)
    assert rc == 0
    assert capsys.readouterr().out == "220"


def test_cmd_first_accident_pf_after_switch(capsys, tmp_path: Path) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text(
        '{"accidents":[{"payload_frame":120},{"payload_frame":150},{"payload_frame":140}]}',
        encoding="utf-8",
    )
    args = argparse.Namespace(meta_path=str(meta), switch_pf="130")
    rc = meta_tools.cmd_first_accident_pf_after_switch(args)
    assert rc == 0
    assert capsys.readouterr().out == "140"


def test_cmd_accident_pf_from_collisions_uses_payload_then_fallback(capsys, tmp_path: Path) -> None:
    collisions = tmp_path / "collisions.csv"
    collisions.write_text(
        "is_accident,payload_frame,carla_frame,frame\n"
        "0,100,100,100\n"
        "1,,210,\n"
        "1,205,205,205\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(collisions_path=str(collisions))
    rc = meta_tools.cmd_accident_pf_from_collisions(args)
    assert rc == 0
    assert capsys.readouterr().out == "205"
