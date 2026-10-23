"""Extract and analyze one recorded random-agent episode without restarting simulation."""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def extract_episode(trace, episode, output):
    trace, output = Path(trace), Path(output)
    with np.load(trace, allow_pickle=False) as saved:
        metadata = json.loads(saved["metadata"].item())
        mask = saved["episode_id"] == episode
        if not mask.any():
            raise ValueError(f"No recorded episode {episode}")
        data = {key: saved[key][mask].copy() for key in saved.files if key != "metadata"}
    source_entry = next(ep for ep in metadata["episodes"] if ep["episode_id"] == episode)
    entry = copy.deepcopy(source_entry)
    entry.update(episode_id=0, recorded_start_s=0, recorded_end_s=float(data["episode_time_s"][-1]))
    data["time_s"] = data["episode_time_s"].copy()
    data["episode_id"][:] = 0
    data["control_step"] -= data["control_step"][0]
    metadata.update(
        status="extracted_episode",
        stop_reason=source_entry.get("reset_reason", "partial episode"),
        source_trace=str(trace.resolve()),
        source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),
        source_episode_id=episode,
        source_duration_requested_s=metadata.get("duration_requested_s"),
        duration_requested_s=float(data["dt_s"].sum()),
        episodes=[entry],
        physics_samples=int(mask.sum()),
        recorded_duration_s=float(data["dt_s"].sum()),
    )
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output / "traces.npz", **data, metadata=json.dumps(metadata, ensure_ascii=False))
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    shutil.copy2(trace.parent / "model_snapshot.json", output / "model_snapshot.json")
    return output / "traces.npz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trace = extract_episode(args.trace, args.episode, args.output_dir)
    from .diagnostics import analyze as diagnostics
    from .random_report import analyze

    analyze(trace)
    diagnostics(trace.parent)


if __name__ == "__main__":
    main()
