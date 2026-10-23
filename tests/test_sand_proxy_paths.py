"""Terrain isolation across planning, downstream refits and deployment backups."""

import hashlib
import json
from pathlib import Path

import pytest

from sand_proxy_lab import paths
from sand_proxy_lab.calibration import __main__ as calibration
from sand_proxy_lab.shadow import refit


def test_planning_two_terrains_keeps_data_and_backbone_paths_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "TERRAINS_ROOT", tmp_path / "terrains")
    config = Path(__file__).resolve().parents[1] / "sand_proxy_lab/configs/m20_foot_intruder.json"
    for name in ("soft_sand", "dense_sand"):
        monkeypatch.setattr("sys.argv", ["calibration", "plan", "--terrain", name, "--config", str(config)])
        calibration.main()
        terrain = tmp_path / "terrains" / name
        assert (terrain / "intruder/config.json").is_file()
        assert len(json.loads((terrain / "intruder/cases.json").read_text())) == 179
        assert paths.backbone_directory(terrain / "intruder") == terrain / "fit/backbone"
    assert paths.backbone_directory(tmp_path / "custom") == tmp_path / "custom/backbone_fit"


def test_refit_defaults_follow_input_terrain_instead_of_global_default(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "TERRAINS_ROOT", tmp_path / "terrains")
    source = tmp_path / "terrains/new_sand/shadow/run_001/traces.npz"
    captured = []
    monkeypatch.setattr(refit, "fit_trace", lambda *args: captured.append(args))
    monkeypatch.setattr("sys.argv", ["refit", "--trace", str(source)])
    refit.main()
    trace, output, _, _, _, terrain = captured[0]
    assert trace == source
    assert output.parent == tmp_path / "terrains/new_sand/fit/contact"
    assert terrain == "new_sand"


def test_deployment_backup_stays_with_candidate_terrain(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "TERRAINS_ROOT", tmp_path / "terrains")
    candidate = tmp_path / "terrains/new_sand/fit/contact/run_001/proxy_model.json"
    candidate.parent.mkdir(parents=True)
    candidate.write_bytes(refit.DEFAULT_MODEL_PATH.read_bytes())
    destination = tmp_path / "active.json"
    previous = b'{"previous": true}\n'
    destination.write_bytes(previous)
    refit.deploy_model(candidate, destination)
    sha = hashlib.sha256(previous).hexdigest()[:12]
    assert (tmp_path / f"terrains/new_sand/deployment_backups/m20_foot_sand_{sha}.json").read_bytes() == previous
    assert destination.read_bytes() == candidate.read_bytes()
    assert not (tmp_path / "terrains" / paths.DEFAULT_TERRAIN).exists()


def test_terrain_argument_supports_names_and_rejects_paths():
    import argparse

    parser = argparse.ArgumentParser()
    paths.add_terrain_argument(parser)
    assert parser.parse_args(["--terrain", "沙地_01"]).terrain == "沙地_01"
    for value in ("../other", "/tmp/other", "one/two", "..", ""):
        with pytest.raises(SystemExit):
            parser.parse_args(["--terrain", value])
