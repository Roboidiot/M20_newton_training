"""Contact calibration recovery, held-out data isolation and foot-only data provenance."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sand_proxy_lab.calibration.common import identity
from sand_proxy_lab.calibration.config import COL, COLUMNS, case_dict, cases
from sand_proxy_lab.calibration.foot_contact_fit import fit_foot_parameters, fit_rate, load_foot_data


@pytest.fixture(scope="module")
def suite():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "sand_proxy_lab/configs/m20_foot_intruder.json").read_text())
    experiments = []
    for case in cases(config):
        q = np.linspace(0, 1, 121)
        a = np.zeros((len(q), len(COLUMNS)))
        d = (
            np.linspace(0.01, 0.2, len(q))
            if case.motion == "press"
            else case.depth - (case.depth + 0.02) * q
            if case.motion == "pull"
            else np.full(len(q), case.depth)
        )
        a[:, COL["time"]] = np.arange(len(q)) * 0.005
        a[:, COL["depth"]] = d
        a[:, COL["progress"]] = q
        a[:, COL["measure"]] = 1
        if case.motion == "press":
            a[:, COL["vz"]] = -case.speed
            a[:, COL["fz"]] = 10000 * d**1.5 * (1 + 2 * case.speed)
        elif case.motion == "pull":
            a[:, COL["vz"]] = case.speed
            support = 10000 * case.depth**1.5
            a[:, COL["fz"]] = np.maximum(
                0, support - 1e5 * case.depth**0.7 * (case.depth - d) - 20 * case.depth**0.5 * case.speed
            )
            hold = a[:1].copy()
            hold[:, COL["measure"]] = 0
            hold[:, COL["vz"]] = 0
            hold[:, COL["fz"]] = support
            hold[:, COL["time"]] = -0.005
            a = np.concatenate((hold, a))
        else:
            suffix = "x" if case.motion == "forward" else "y"
            a[:, COL[suffix]] = q * 0.08
            a[:, COL[f"v{suffix}"]] = case.speed
            a[:, COL[f"f{suffix}"]] = -3000 * d**1.8 * (1 + case.speed)
            a[:, COL["fz"]] = 10000 * d**1.5 * np.exp(-q * 0.08 / 0.005) + 0.4 * np.abs(a[:, COL[f"f{suffix}"]])
        experiments.append({"case": case, "data": a, "metadata": {"data_origin": "synthetic_unit_test"}})
    return config, experiments


def test_positive_depth_and_rate_coefficients_recovered_without_policy_or_gpu():
    d, v = np.meshgrid([0.01, 0.05, 0.10, 0.20], [0.05, 0.15, 0.30])
    force = 32000 * d**1.6 * (1 + 0.8 * v)
    law = fit_rate(d.ravel(), v.ravel(), force.ravel(), np.ones(force.size))
    np.testing.assert_allclose([law["a"], law["n"], law["b"]], [32000, 1.6, 0.8], rtol=1e-7)


def test_validation_targets_cannot_change_backbone_coefficients(suite):
    config, experiments = suite
    reference = fit_foot_parameters(config, experiments)
    altered = copy.deepcopy(experiments)
    for e in altered:
        if e["case"].split == "validation":
            e["data"][:, COL["fx"] : COL["fz"] + 1] = 1e12
    assert reference == fit_foot_parameters(config, altered)
    assert len(reference["fit_cases"]) == 114
    np.testing.assert_allclose([reference["bearing"][k] for k in ("a", "n", "b")], [10000, 1.5, 2], rtol=1e-7)


def test_loader_requires_only_foot_files_and_keeps_raw_samples_unchanged(suite, tmp_path):
    config, experiments = suite
    (tmp_path / "config.json").write_text(json.dumps(config))
    for e in experiments:
        if e["case"].geometry != "foot":
            continue
        metadata = {
            **e["metadata"],
            "columns": list(COLUMNS),
            "config_id": identity(config),
            "case": case_dict(e["case"]),
        }
        np.savez_compressed(tmp_path / f"{e['case'].name}.npz", data=e["data"], metadata=json.dumps(metadata))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.npz")}
    _, loaded = load_foot_data(tmp_path)
    assert len(loaded) == 179
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.glob("*.npz")}
    assert all(e["metadata"]["data_origin"] == "synthetic_unit_test" for e in loaded)
