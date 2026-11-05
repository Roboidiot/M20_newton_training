"""Public physics interface: load_physics_config() and apply_physics_config()."""

import copy
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "default.json"


@dataclass
class PhysicsConfig:
    time: dict
    material: dict
    mpm: dict
    particles: dict
    coupling: dict

    @property
    def physics_dt_s(self):
        return self.time["physics_dt_s"]

    @property
    def control_dt_s(self):
        return self.time["control_dt_s"]

    def to_dict(self):
        return asdict(self)

    def validate(self):
        """Reject typos, nonfinite values and invalid ranges before simulator launch."""
        defaults = json.loads(DEFAULT_CONFIG.read_text())
        values = self.to_dict()
        integer_fields = {
            "mpm.max_iterations",
            "mpm.grid_padding",
            "mpm.max_active_cell_count",
            "mpm.max_leaf_node_count",
            "mpm.max_lower_node_count",
            "mpm.max_upper_node_count",
            "coupling.iterations",
            "coupling.robot_substeps",
        }
        boolean_fields = {"mpm.separate_worlds", "mpm.collider_normal_from_sdf_gradient"}
        numeric_fields = {
            "mpm.tolerance",
            "mpm.voxel_size",
            "mpm.critical_fraction",
            "mpm.air_drag",
            "coupling.mass_scale",
        }
        for section, expected in defaults.items():
            actual = values[section]
            if not isinstance(actual, dict) or actual.keys() != expected.keys():
                raise ValueError(f"Unexpected/missing fields in {section}")
            for key in expected:
                value = actual[key]
                name = f"{section}.{key}"
                if name == "mpm.solver":
                    continue
                if name in boolean_fields:
                    valid = isinstance(value, bool)
                elif name in integer_fields:
                    valid = isinstance(value, int) and not isinstance(value, bool)
                elif section in ("time", "material", "particles") or name in numeric_fields:
                    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
                else:
                    valid = isinstance(value, str)
                if not valid:
                    raise ValueError(f"Invalid type or nonfinite value for {name}: {value!r}")
        for section, keys in (
            ("time", ("physics_dt_s", "control_dt_s")),
            ("material", ("density", "young_modulus")),
            ("mpm", ("voxel_size", "tolerance", "max_iterations")),
            ("particles", ("particles_per_cell",)),
            ("coupling", ("mass_scale", "iterations", "robot_substeps")),
        ):
            for key in keys:
                if values[section][key] <= 0:
                    raise ValueError(f"{section}.{key} must be positive")
        ratio = self.control_dt_s / self.physics_dt_s
        if ratio < 1 or not math.isclose(ratio, round(ratio), rel_tol=0, abs_tol=1e-8):
            raise ValueError("time.control_dt_s must be an integer multiple of time.physics_dt_s")
        for key, value in self.material.items():
            if key not in ("poisson_ratio", "dilatancy") and value < 0:
                raise ValueError(f"material.{key} must be nonnegative")
        if not -1 < self.material["poisson_ratio"] < 0.5:
            raise ValueError("material.poisson_ratio must be between -1 and 0.5 (exclusive)")
        for name, value in (
            ("mpm.critical_fraction", self.mpm["critical_fraction"]),
            ("particles.jitter_fraction", self.particles["jitter_fraction"]),
            ("material.tensile_yield_ratio", self.material["tensile_yield_ratio"]),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.mpm["grid_padding"] < 0 or self.mpm["air_drag"] < 0:
            raise ValueError("mpm.grid_padding and mpm.air_drag must be nonnegative")
        for key in ("max_active_cell_count", "max_leaf_node_count", "max_lower_node_count", "max_upper_node_count"):
            if self.mpm[key] != -1 and self.mpm[key] <= 0:
                raise ValueError(f"mpm.{key} must be -1 or positive")
        for section, key, choices in (
            ("mpm", "grid_type", ("sparse", "dense", "fixed")),
            ("mpm", "warmstart_mode", ("none", "auto", "particles", "grid", "smoothed")),
            ("mpm", "collider_velocity_mode", ("forward", "backward")),
            ("mpm", "transfer_scheme", ("apic", "pic")),
            ("mpm", "integration_scheme", ("pic", "gimp")),
            ("mpm", "velocity_basis", ("Q1", "B2", "B3")),
            ("coupling", "mode", ("staggered", "lagged")),
        ):
            if values[section][key] not in choices:
                raise ValueError(f"{section}.{key} must be one of {choices}")
        for key in ("collider_basis", "strain_basis"):
            if not re.fullmatch(r"pic(?:[1-9]\d*)?|[PQBS]\d+d?", self.mpm[key]):
                raise ValueError(f"Invalid mpm.{key}: {self.mpm[key]}")
        solvers = self.mpm["solver"]
        allowed = {
            "auto",
            "gs",
            "gauss-seidel",
            "gs-soa",
            "gauss-seidel-soa",
            "gs-batched",
            "gauss-seidel-batched",
            "jacobi",
            "cg",
            "conjugate-gradient",
            "cr",
            "conjugate-residual",
            "gmres",
            "generalized-minimal-residual",
        }
        sequence = [solvers] if isinstance(solvers, str) else solvers
        if (
            not isinstance(sequence, (list, tuple))
            or not sequence
            or any(not isinstance(solver, str) or solver not in allowed for solver in sequence)
        ):
            raise ValueError("mpm.solver must name a supported solver or a nonempty sequence of solvers")
        return self


def load_physics_config(path=None, overrides=()):
    """Merge defaults < partial JSON file < repeated 'section.key=value' overrides."""
    values = json.loads(DEFAULT_CONFIG.read_text())
    if path is not None:
        patch = json.loads(Path(path).read_text())
        if not isinstance(patch, dict):
            raise ValueError("Physics JSON must contain an object")
        for section, fields in patch.items():
            if section not in values or not isinstance(fields, dict):
                raise ValueError(f"Unknown physics section or invalid object: {section}")
            for key, value in fields.items():
                if key not in values[section]:
                    raise ValueError(f"Unknown physics field: {section}.{key}")
                values[section][key] = value
    for assignment in overrides:
        name, separator, encoded = assignment.partition("=")
        parts = name.split(".")
        if not separator or len(parts) != 2 or parts[0] not in values or parts[1] not in values[parts[0]]:
            raise ValueError(f"Expected a known section.key=value, got {assignment!r}")
        try:
            value = json.loads(encoded)
        except json.JSONDecodeError:
            value = encoded
        values[parts[0]][parts[1]] = value
    return PhysicsConfig(**values).validate()


def apply_physics_config(cfg, physics):
    """Apply validated values to a fresh M20 task configuration, before building physics."""
    import numpy as np

    physics.validate()
    cfg.sim.dt = physics.physics_dt_s
    cfg.decimation = round(physics.control_dt_s / physics.physics_dt_s)
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.physics.use_cuda_graph = False
    coupled = cfg.sim.physics.solver_cfg
    sand = next(entry for entry in coupled.entries if entry.name == "sand")
    robot = next(entry for entry in coupled.entries if entry.name == "robot")
    for name, value in physics.mpm.items():
        if not hasattr(sand.solver_cfg, name):
            raise ValueError(f"MPM backend does not support parameter {name!r}")
        setattr(sand.solver_cfg, name, tuple(value) if name == "solver" and isinstance(value, list) else value)
    robot.substeps = physics.coupling["robot_substeps"]
    coupled.iterations = physics.coupling["iterations"]
    proxy = next(p for p in coupled.proxies if p.source == "robot" and p.destination == "sand")
    proxy.mode, proxy.mass_scale = physics.coupling["mode"], physics.coupling["mass_scale"]
    grid = cfg.scene.sand.spawn
    grid.material = copy.deepcopy(grid.material)
    for name, value in physics.material.items():
        if not hasattr(grid.material, name):
            raise ValueError(f"Material backend does not support parameter {name!r}")
        setattr(grid.material, name, value)
    grid.voxel_size = physics.mpm["voxel_size"]
    grid.particles_per_cell = physics.particles["particles_per_cell"]
    spacing = grid.voxel_size / grid.particles_per_cell
    grid.jitter = physics.particles["jitter_fraction"] * spacing
    # Preserve the nominal bed thickness and fit whole particle layers, as the shared scene does.
    nominal_top = float(np.nextafter(np.float32(grid.upper[2]), np.float32(np.inf)))
    layers = math.floor((nominal_top - grid.lower[2]) / spacing + 1e-9)
    if layers < 1:
        raise ValueError("Particle spacing exceeds the sand bed thickness")
    upper = list(grid.upper)
    upper[2] = float(np.nextafter(np.float32(grid.lower[2] + layers * spacing), np.float32(grid.lower[2])))
    grid.upper = tuple(upper)
    return cfg
