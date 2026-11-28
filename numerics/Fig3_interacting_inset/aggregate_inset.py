"""Rebuild the Fig. 3 inset from the five N=5, E=0.1 periodic results."""
import argparse
import json
from pathlib import Path

import pandas as pd


def aggregate(source, output):
    records = []
    for rep in range(5):
        path = source / f"N005_E0.1_r{rep}_result.json"
        run = json.loads(path.read_text())
        if (run["N"], run["E"], run["rep"]) != (5, 0.1, rep):
            raise ValueError(f"Unexpected realization in {path}")
        for grid in (5, 10, 20, 40, 80, 160):
            rate = run[f"estimator_K{grid}"]
            records.append(dict(
                N=run["N"], E=run["E"], K=grid, rep=rep,
                estimator_rate=rate, normalized_estimator=rate / run["E"]**2,
                true_rate=run["true_rate"], lag=run[f"lag_K{grid}"],
                phase=run[f"phase_K{grid}"], cap=run[f"cap_K{grid}"],
                source_file=path.name, method=run["selection_method"],
            ))
    table = pd.DataFrame(records).sort_values(["K", "rep"])
    summary = []
    for grid, group in table.groupby("K"):
        summary.append(dict(
            N=5, E=0.1, K=grid, boxes_total=grid**2,
            estimator_rate=group.estimator_rate.mean(),
            estimator_rate_sd=group.estimator_rate.std(ddof=1),
            normalized_estimator=group.normalized_estimator.mean(),
            normalized_estimator_sd=group.normalized_estimator.std(ddof=1),
            realizations=len(group), lag_min=group.lag.min(), lag_max=group.lag.max(),
            phases=",".join(sorted(set(group.phase))),
            selection_method=",".join(sorted(set(group.method))),
        ))
    output.mkdir(parents=True, exist_ok=True)
    table.drop(columns="method").to_csv(output / "inset_interacting_replicates.csv", index=False)
    pd.DataFrame(summary).to_csv(output / "inset_interacting.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    aggregate(args.source, args.output)
