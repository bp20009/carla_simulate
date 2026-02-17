from pathlib import Path

from scripts.extract_future_accidents import (
    RunKey,
    iter_runs,
    parse_run_key_from_collisions_path,
    safe_float,
    safe_int,
)


def test_safe_int_and_safe_float() -> None:
    assert safe_int("10") == 10
    assert safe_int("10.0") == 10
    assert safe_int("x") is None
    assert safe_float("1.25") == 1.25
    assert safe_float("x") is None


def test_parse_run_key_from_collisions_path() -> None:
    path = Path("/tmp/results/autopilot/lead_3/rep_8/logs/collisions.csv")
    key = parse_run_key_from_collisions_path(path)
    assert key == RunKey(method="autopilot", lead_sec=3, rep=8)


def test_iter_runs_finds_only_paths_with_meta(tmp_path: Path) -> None:
    valid_logs = tmp_path / "autopilot" / "lead_2" / "rep_1" / "logs"
    valid_logs.mkdir(parents=True)
    (valid_logs / "collisions.csv").write_text("frame\n1\n", encoding="utf-8")
    (valid_logs / "meta.json").write_text("{}", encoding="utf-8")

    invalid_logs = tmp_path / "lstm" / "lead_4" / "rep_2" / "logs"
    invalid_logs.mkdir(parents=True)
    (invalid_logs / "collisions.csv").write_text("frame\n1\n", encoding="utf-8")

    runs = list(iter_runs(tmp_path))
    assert len(runs) == 1
    key, collisions_csv, meta_json = runs[0]
    assert key.method == "autopilot"
    assert collisions_csv.name == "collisions.csv"
    assert meta_json.name == "meta.json"
