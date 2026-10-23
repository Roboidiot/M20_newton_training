"""Calibrate an M20 foot-only Newton-terrain-style contact law and replay held-out MPM tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import least_squares

from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH, FootSandModel
from sand_proxy_lab.paths import backbone_directory

from .common import identity
from .config import COL, COLUMNS, case_dict, cases, validate_config
from .data import remove_repeated_terminal_sample


def load_foot_data(directory):
    config = json.loads((directory / "config.json").read_text())
    validate_config(config)
    result = []
    for case in (c for c in cases(config) if c.geometry == "foot"):
        path = directory / f"{case.name}.npz"
        with np.load(path, allow_pickle=False) as saved:
            a, metadata = saved["data"].copy(), json.loads(saved["metadata"].item())
        if metadata["config_id"] != identity(config) or metadata["case"] != case_dict(case):
            raise ValueError(f"Mixed calibration data: {path}")
        if metadata["columns"] != list(COLUMNS):
            raise ValueError(f"Unexpected columns: {path}")
        a, removed = remove_repeated_terminal_sample(a)
        result.append(
            {
                "case": case,
                "data": a,
                "metadata": metadata,
                "removed": removed,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return config, result


def fit_power(depth, target):
    d, y = np.asarray(depth), np.asarray(target)
    good = (d > 0) & (y > 0)
    if good.sum() < 2 or np.ptp(d[good]) < 1e-6:
        raise ValueError("Insufficient positive depth/force variation")
    n, log_a = np.polyfit(np.log(d[good]), np.log(y[good]), 1)
    return {"a": float(np.exp(log_a)), "n": float(n)}


def fit_rate(depth, speed, target, weight):
    d, v, f, w = map(np.asarray, (depth, speed, target, weight))
    initial = fit_power(d, np.maximum(f, 0.01))

    def residual(x):
        prediction = np.exp(x[0]) * d ** x[1] * (1 + x[2] * v)
        return (prediction - f) * w

    optimum = least_squares(
        residual,
        [np.log(initial["a"]), np.clip(initial["n"], 0.5, 4), 0.5],
        bounds=([-20, 0.5, 0], [30, 4, 30]),
        max_nfev=300,
    )
    if not optimum.success:
        raise RuntimeError(optimum.message)
    return {"a": float(np.exp(optimum.x[0])), "n": float(optimum.x[1]), "b": float(optimum.x[2])}


def fit_foot_parameters(config, experiments):
    experiments = [e for e in experiments if e["case"].geometry == "foot"]
    train = [e for e in experiments if e["case"].split == "train"]
    press = [e for e in train if e["case"].motion == "press"]
    ds, vs, fs, ws = [], [], [], []
    for e in press:
        a = e["data"]
        a = a[(a[:, COL["measure"]] > 0.5) & (a[:, COL["depth"]] >= 0.01 - 1e-8) & (a[:, COL["depth"]] <= 0.2 + 1e-8)]
        f = a[:, COL["fz"]]
        ds.extend(a[:, COL["depth"]])
        vs.extend(np.abs(a[:, COL["vz"]]))
        fs.extend(f)
        ws.extend(np.full(len(a), 1 / (np.sqrt(len(a)) * max(5, np.sqrt(np.mean(f * f))))))
    bearing = fit_rate(ds, vs, fs, ws)
    parameters = {
        "schema_version": 1,
        "model_type": "m20_foot_elastoplastic_sand",
        "source_config_id": identity(config),
        "source_data_origins": sorted({e["metadata"].get("data_origin", "newton_mpm") for e in experiments}),
        "geometry": config["wheel_shape"],
        "material": config["material"],
        "bearing": bearing,
        "domain": {"max_depth_m": 0.20, "max_speed_m_s": 0.30},
        "history": {"slots": 16, "diameter_m": 2 * config["wheel_radius"], "default_keep": 0.7},
        "low_speed_epsilon_m_s": 0.02,
        "fit_cases": [e["case"].name for e in train],
        "unfitted_settings": [
            "16 shared footprint slots and keep=0.7 follow reference assumptions",
            "low-speed regularization epsilon=0.02 m/s and explicit-force stability limiter",
            "force application point is the wheel's lowest-point contact approximation",
        ],
        "limitations": [
            "Foot forces only: no shin, thigh or belly force",
            "Fixed-orientation translational calibration; rolling, tilt, reverse and combined motions extrapolate",
            "Steady drag uses final 2/3 as in newton-terrain; full transient errors are reported separately",
            "Force strength saturates at max calibrated depth/speed; coverage flags are available at runtime",
            "MPM intruder replay is not a whole-robot policy validation",
        ],
    }
    for motion, axis, suffix in (("forward", "fx", "x"), ("lateral", "fy", "y")):
        nodes, ratios, lengths = [], [], []
        for e in train:
            if e["case"].motion != motion:
                continue
            a = e["data"]
            a = a[a[:, COL["measure"]] > 0.5]
            steady = a[a[:, COL["progress"]] >= 1 / 3]
            depth, speed = e["case"].depth, e["case"].speed
            drag = max(0.01, -float(steady[:, COL[axis]].mean()))
            # Fit the unregularized amplitude to reproduce the runtime velocity smoothing.
            amplitude = drag * np.sqrt(speed**2 + 0.02**2) / speed
            nodes.append((depth, speed, amplitude, 1 / max(5, drag)))
            lift = max(0.01, float(steady[:, COL["fz"]].mean()))
            ratios.append((depth, lift / drag))
            excess = a[:, COL["fz"]] - lift
            coordinate = a[:, COL["x" if suffix == "x" else "y"]]
            travel = coordinate - coordinate[0]
            mask = (travel > 0) & (travel < 0.03) & (excess > max(1, 0.05 * excess[0]))
            if mask.sum() >= 3 and excess[0] > 5:
                slope = np.polyfit(travel[mask], np.log(excess[mask] / excess[0]), 1)[0]
                if slope < 0:
                    lengths.append(float(np.clip(-1 / slope, 0.001, 0.03)))
        parameters[f"drag_{suffix}"] = fit_rate(*np.asarray(nodes).T)
        parameters[f"lift_ratio_{suffix}"] = fit_power(*np.asarray(ratios).T)
        parameters[f"shear_length_{suffix}_m"] = float(np.median(lengths)) if lengths else 0.005
    stiffness, damping, pulling = [], [], []
    for e in train:
        if e["case"].motion != "pull":
            continue
        c, a = e["case"], e["data"]
        measured = a[a[:, COL["measure"]] > 0.5]
        hold = a[(a[:, COL["measure"]] < 0.5) & (np.abs(a[:, COL["vz"]]) < 1e-9)]
        f = measured[:, COL["fz"]]
        displacement = c.depth - measured[:, COL["depth"]]
        mask = (displacement > 1e-8) & (displacement < 0.02) & (f > 0.2 * f[0])
        if c.speed <= 0.1 and mask.any() and f[0] > 1:
            k = float(np.median((f[0] - f[mask]) / displacement[mask]))
            if k > 0:
                stiffness.append((c.depth, k))
        if c.speed <= 0.1 and len(hold) and f[0] > 1:
            cup = (float(np.median(hold[-10:, COL["fz"]])) - f[0]) / c.speed
            if cup > 0:
                damping.append((c.depth, cup))
        mask = measured[:, COL["depth"]] > 0.01
        pulling.extend((-f[mask] / measured[mask, COL["depth"]]).tolist())
    parameters["unloading_stiffness"] = fit_power(*np.asarray(stiffness).T)
    parameters["unloading_damping"] = fit_power(*np.asarray(damping).T)
    parameters["pull_p_N_per_m"] = max(0, float(np.median(pulling)))
    parameters["calibration"] = {
        "training_cases": len(train),
        "validation_cases": sum(e["case"].split == "validation" for e in experiments),
        "scope": "foot-only MPM reference; whole-robot validation pending",
    }
    return parameters


def replay(parameters, experiments, dt):
    """Replay full preparation/hold/measurement paths; measure held-out errors without fitting them."""
    model = FootSandModel(parameters, len(experiments), num_feet=1)
    max_steps = max(len(e["data"]) for e in experiments)
    padded = np.stack([np.pad(e["data"], ((0, max_steps - len(e["data"])), (0, 0)), mode="edge") for e in experiments])
    data = torch.tensor(padded, dtype=torch.float32)
    predictions = torch.zeros((len(experiments), max_steps, 3))
    with torch.inference_mode():
        for i in range(max_steps):
            depth = data[:, i, COL["depth"]][:, None]
            velocity = data[:, i, COL["vx"] : COL["vz"] + 1][:, None, :]
            xy = data[:, i, COL["x"] : COL["y"] + 1][:, None, :]
            predictions[:, i] = model.step(depth, velocity, xy, dt)[:, 0]
    predictions = predictions.numpy()
    results = []
    for index, e in enumerate(experiments):
        a = e["data"]
        mask = a[:, COL["measure"]] > 0.5
        if e["case"].motion == "press":
            mask &= (a[:, COL["depth"]] >= 0.01 - 1e-8) & (a[:, COL["depth"]] <= 0.2 + 1e-8)
        actual = a[mask, COL["fx"] : COL["fz"] + 1]
        predicted = predictions[index, : len(a)][mask]
        if not np.isfinite(predicted).all():
            raise ValueError("Nonfinite replay force")
        main = {"press": "Fz", "pull": "Fz", "forward": "Fx", "lateral": "Fy"}[e["case"].motion]
        metrics = {}
        for axis, component in enumerate(("Fx", "Fy", "Fz")):
            f, estimate = actual[:, axis], predicted[:, axis]
            rmse = float(np.sqrt(np.mean((f - estimate) ** 2)))
            rms = float(np.sqrt(np.mean(f * f)))
            metrics[component] = {
                "rmse_N": rmse,
                "nrmse": rmse / max(1, rms),
                "passed": rmse <= max(5, 0.25 * rms),
                "measured_mean_N": float(f.mean()),
                "predicted_mean_N": float(estimate.mean()),
            }
        results.append({**case_dict(e["case"]), "component": main, "metrics": metrics})
    return results, predictions


def fit_directory(directory: Path, model_out: Path | None = None):
    config, experiments = load_foot_data(directory)
    parameters = fit_foot_parameters(config, experiments)
    results, predictions = replay(parameters, experiments, config["dt"])
    validation = [r for r in results if r["split"] == "validation"]
    passed = sum(r["metrics"][r["component"]]["passed"] for r in validation)
    parameters["calibration"].update(
        primary_passed=passed, primary_validation_total=len(validation), status="experimental_calibrated_contact_model"
    )
    output = backbone_directory(directory)
    output.mkdir(parents=True, exist_ok=True)
    # A calibration intermediate must never silently replace the active schema-2 model.
    model_out = model_out or output / "proxy_model.json"
    if model_out.resolve() == DEFAULT_MODEL_PATH.resolve():
        raise ValueError("Backbone is a calibration intermediate; deploy the final shadow-refitted model instead")
    text = json.dumps(parameters, indent=2) + "\n"
    (output / "proxy_model.json").write_text(text)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    model_out.write_text(text)
    (output / "report.json").write_text(json.dumps({"parameters": parameters, "results": results}, indent=2) + "\n")
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "raw_files_modified": False,
                "source_config_id": identity(config),
                "files": {e["case"].name + ".npz": e["sha256"] for e in experiments},
                "duplicate_terminal_samples_removed": {
                    e["case"].name: e["removed"] for e in experiments if e["removed"]
                },
            },
            indent=2,
        )
        + "\n"
    )
    lines = [
        "# M20 foot-only contact proxy",
        "",
        f"{parameters['calibration']['training_cases']} training / {len(validation)} validation foot experiments; "
        f"primary full-history pass {passed}/{len(validation)}.",
        "Replay includes preparation, hold, unloading and shear; no shin data or forces.",
        "Constitutive strengths use only training data. Footprint retention and stabilizers are explicit settings.",
        "Runtime limits explicit drag/damping slopes using timestep and effective mass; "
        "that limiter is off in prescribed-path replay.",
        "Experimental model: closed-loop checks and force-domain coverage are required before large training runs.",
        "",
        "| Motion | Component | Mean RMSE (N) | Mean NRMSE | Passed |",
        "|---|---|---:|---:|---:|",
    ]
    for motion in ("press", "pull", "forward", "lateral"):
        group = [r for r in validation if r["motion"] == motion]
        c = group[0]["component"]
        m = [r["metrics"][c] for r in group]
        lines.append(
            f"| {motion} | {c} | {np.mean([x['rmse_N'] for x in m]):.2f} | "
            f"{np.mean([x['nrmse'] for x in m]):.2%} | {sum(x['passed'] for x in m)}/{len(m)} |"
        )
    lines += ["", "![Held-out prescribed paths](validation.png)", ""]
    (output / "report.md").write_text("\n".join(lines))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 2, figsize=(12, 12))
    for row, motion in enumerate(("press", "pull", "forward", "lateral")):
        candidates = [
            (i, e) for i, e in enumerate(experiments) if e["case"].split == "validation" and e["case"].motion == motion
        ]
        i, e = min(candidates, key=lambda x: abs(x[1]["case"].depth - 0.125) + abs(x[1]["case"].speed - 0.175))
        a = e["data"]
        mask = a[:, COL["measure"]] > 0.5
        t = a[mask, COL["time"]]
        component = {"press": "fz", "pull": "fz", "forward": "fx", "lateral": "fy"}[motion]
        for col, c in enumerate((component, "fz")):
            axis = axes[row, col]
            axis.plot(t, a[mask, COL[c]], label="MPM")
            axis.plot(t, predictions[i, : len(a)][mask, COL[c] - COL["fx"]], label="foot proxy")
            axis.set(title=f"{motion}: {c}", xlabel="time (s)", ylabel="N")
            axis.grid(alpha=0.2)
            axis.legend()
    figure.tight_layout()
    figure.savefig(output / "validation.png", dpi=140)
    plt.close(figure)
    print(
        f"Foot-only model: {model_out}; primary validation {passed}/{len(validation)}; report: {output / 'report.md'}"
    )
    return parameters, results
