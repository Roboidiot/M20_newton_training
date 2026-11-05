"""Physics API precedence, real configuration wiring and validation without a GPU."""

import json
import subprocess
import sys

import pytest

from sand_pinn_lab.collect import load_physics_config
from sand_pinn_lab.collect.config import DATA_ROOT, environment_config, parse_args


def test_partial_json_and_cli_precedence_do_not_mutate_defaults(tmp_path):
    path = tmp_path / "physics.json"
    path.write_text(
        json.dumps(
            {
                "material": {"friction": 0.6, "density": 1500.0},
                "mpm": {"max_iterations": 500},
                "coupling": {"mode": "lagged"},
            }
        )
    )
    resolved = load_physics_config(path, ["material.friction=0.8", "mpm.collider_basis=S2"])
    assert resolved.material["friction"] == 0.8
    assert resolved.material["density"] == 1500
    assert resolved.mpm["max_iterations"] == 500 and resolved.mpm["tolerance"] == 1e-5
    assert resolved.mpm["collider_basis"] == "S2" and resolved.coupling["mode"] == "lagged"
    assert load_physics_config().material["friction"] == 0.7
    json.dumps(resolved.to_dict(), allow_nan=False)


def test_physics_overrides_reach_material_solver_particles_and_timing():
    args = parse_args(
        [
            "--set",
            "material.density=1800",
            "--set",
            "material.young_modulus=1e8",
            "--set",
            "material.friction=0.85",
            "--set",
            "material.yield_stress=2000",
            "--set",
            "mpm.max_iterations=500",
            "--set",
            "mpm.tolerance=1e-6",
            "--set",
            "mpm.collider_basis=S2",
            "--set",
            "mpm.voxel_size=0.025",
            "--set",
            'mpm.solver=["conjugate-residual","gauss-seidel"]',
            "--set",
            "particles.particles_per_cell=2",
            "--set",
            "particles.jitter_fraction=0.5",
            "--set",
            "coupling.mode=lagged",
            "--set",
            "coupling.mass_scale=0.5",
            "--set",
            "coupling.iterations=3",
            "--set",
            "coupling.robot_substeps=4",
            "--set",
            "time.physics_dt_s=0.0025",
        ]
    )
    cfg = environment_config(args)
    grid = cfg.scene.sand.spawn
    assert grid.material.density == 1800 and grid.material.young_modulus == 1e8
    assert grid.material.friction == 0.85 and grid.material.yield_stress == 2000
    assert grid.voxel_size == 0.025 and grid.particles_per_cell == 2
    assert grid.jitter == pytest.approx(0.00625)
    assert grid.upper[2] == pytest.approx(0.3)
    mpm = next(e for e in cfg.sim.physics.solver_cfg.entries if e.name == "sand").solver_cfg
    assert mpm.voxel_size == grid.voxel_size and mpm.collider_basis == "S2"
    assert mpm.max_iterations == 500 and mpm.tolerance == 1e-6
    assert mpm.solver == ("conjugate-residual", "gauss-seidel")
    assert cfg.sim.dt == 0.0025 and cfg.decimation == 8
    coupled = cfg.sim.physics.solver_cfg
    assert coupled.iterations == 3
    assert coupled.proxies[0].mode == "lagged" and coupled.proxies[0].mass_scale == 0.5
    assert next(e for e in coupled.entries if e.name == "robot").substeps == 4
    default = environment_config(parse_args([]))
    assert default.scene.sand.spawn.material.density == 1700
    assert default.scene.sand.spawn.voxel_size == 0.02
    assert DATA_ROOT.name == "data" and DATA_ROOT.parent.name == "sand_pinn_lab"


@pytest.mark.parametrize(
    "override",
    [
        "material.fricton=0.8",
        "unknown.x=1",
        "material.density=0",
        "material.friction=nan",
        "material.poisson_ratio=0.5",
        "mpm.max_iterations=1.5",
        "mpm.max_iterations=true",
        "mpm.tolerance=-1",
        "mpm.collider_velocity_mode=typo",
        "mpm.separate_worlds=1",
        "time.physics_dt_s=0.003",
        "coupling.mode=typo",
        "coupling.iterations=0",
        "particles.particles_per_cell=0",
        "mpm.solver=[]",
    ],
)
def test_invalid_physics_fails_before_launch(override):
    with pytest.raises(ValueError):
        load_physics_config(overrides=[override])


def test_programmatic_changes_revalidated():
    physics = load_physics_config()
    physics.material["density"] = -1
    with pytest.raises(ValueError, match="density"):
        physics.validate()


def test_default_numeric_json_spelling_does_not_change_parameter_type(tmp_path, monkeypatch):
    from sand_pinn_lab.collect import physics as interface

    values = load_physics_config().to_dict()
    values["material"]["friction"] = 1
    path = tmp_path / "default.json"
    path.write_text(json.dumps(values))
    monkeypatch.setattr(interface, "DEFAULT_CONFIG", path)
    assert interface.load_physics_config(overrides=["material.friction=0.8"]).material["friction"] == 0.8


def test_new_and_compatible_cli_print_config_without_loading_simulator():
    for module in ("sand_pinn_lab.collect", "sand_pinn_lab"):
        result = subprocess.run(
            [sys.executable, "-m", module, "--set", "material.friction=0.9", "--print-config"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert json.loads(result.stdout)["material"]["friction"] == 0.9
