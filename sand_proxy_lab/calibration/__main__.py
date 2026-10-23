"""Foot-only MPM calibration: plan, collect, fit-backbone, run, status."""

from __future__ import annotations

import argparse
import fcntl
import json
from collections import Counter
from pathlib import Path

from sand_proxy_lab.paths import add_terrain_argument, backbone_directory, terrain_directory

from .common import identity
from .config import case_dict, cases, training_snapshot, validate_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "collect", "fit-backbone", "run", "status"))
    add_terrain_argument(parser)
    parser.add_argument("--output-dir", type=Path, help="Override the terrain's intruder data directory")
    parser.add_argument("--config", type=Path, help="Frozen material/geometry settings; default reads current M20/Sand")
    parser.add_argument("--case", nargs="+")
    parser.add_argument("--motion", choices=("press", "pull", "forward", "lateral"))
    parser.add_argument("--split", choices=("train", "validation"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-out", type=Path, help="Override backbone JSON; default terrain/fit/backbone")
    args = parser.parse_args()
    directory = (args.output_dir or terrain_directory(args.terrain) / "intruder").resolve()
    if args.command in ("fit-backbone", "status"):
        if args.config or args.case or args.motion or args.split:
            parser.error("fit-backbone/status read the complete frozen dataset; collection filters are not accepted")
        if args.command == "fit-backbone":
            from .foot_contact_fit import fit_directory

            fit_directory(directory, args.model_out)
        else:
            config = json.loads((directory / "config.json").read_text())
            validate_config(config)
            suite = cases(config)
            counts = Counter(c.split for c in suite if (directory / f"{c.name}.npz").is_file())
            print(f"Foot data: {sum(counts.values())}/{len(suite)}; by split: {dict(counts)}")
            print(f"Backbone candidate: {backbone_directory(directory) / 'proxy_model.json'}")
        return
    snapshot = directory / "config.json"
    config = (
        json.loads(args.config.read_text())
        if args.config
        else (json.loads(snapshot.read_text()) if snapshot.is_file() else training_snapshot())
    )
    validate_config(config)
    full_suite = cases(config)
    selected = full_suite
    if args.case:
        unknown = set(args.case) - {c.name for c in full_suite}
        if unknown:
            parser.error(f"Unknown foot cases: {sorted(unknown)}")
        selected = [c for c in selected if c.name in args.case]
    for key in ("motion", "split"):
        if getattr(args, key):
            selected = [c for c in selected if getattr(c, key) == getattr(args, key)]
    if not selected:
        parser.error("No foot cases match")
    if args.command == "run" and (args.case or args.motion or args.split):
        parser.error("run requires the full 179-case suite; use collect for selected cases")
    directory.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and identity(json.loads(snapshot.read_text())) != identity(config):
        raise ValueError("Output directory belongs to different settings; use a new directory")
    print(f"Config {identity(config)}; {len(selected)} foot cases; {dict(Counter(c.split for c in selected))}")
    with (directory / ".collection.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        snapshot.write_text(json.dumps(config, indent=2) + "\n")
        (directory / "cases.json").write_text(json.dumps([case_dict(c) for c in full_suite], indent=2) + "\n")
        if args.command == "plan":
            return
        import warp as wp

        from .collect import run_case

        wp.init()
        if not wp.get_device(args.device).is_cuda:
            raise RuntimeError("MPM collection requires CUDA; plan/fit-backbone/status can run on CPU")
        for case in selected:
            run_case(config, case, directory, args.device)
        if args.command == "run":
            from .foot_contact_fit import fit_directory

            fit_directory(directory, args.model_out)


if __name__ == "__main__":
    main()
