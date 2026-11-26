#!/usr/bin/env python3
"""Reproduce the physical-current moment checks in moment_tests.json.

The tests compare the C++ count kernel with direct particle-pair counting,
check the exact one-particle and rigid-cluster limits, and refine the Sobol
integration for 16 interacting configurations. The two pilot trajectories
must be available as N020_E0.1_r0 and N100_E1.0_r0 in --runs. Their metadata,
snapshots, and period statistics are loaded by steady_core.load_run.

All integration seeds and budgets are fixed below. Wall-clock timings vary
between executions. The full quadrature-refinement test is computationally
more expensive than the analytic and direct-count checks.

Example:
    python validate_moments.py --runs ../engine/raw --output moment_tests.json
"""
from pathlib import Path
import argparse
import json

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

from moment_kernel import current_weights, moments
from positive_kernel import gaussian_bin_kernel
from steady_core import current_mean, histogram, load_run, probabilities, tur_rate


HERE = Path(__file__).resolve().parent
ANALYTIC_CASES = (
    (2, 1, .1, .05, 5),
    (2, 1, 1., 1., 5),
    (200, 100, .1, .05, 5),
    (200, 100, 1., 1., 5),
    (200, 100, 1., .05, 20),
)
PILOT_NAMES = ("N020_E0.1_r0", "N100_E1.0_r0")
PILOT_LAGS = (.05, .5, 5., 20.)
PILOT_BUDGETS = ((11, 256, 1731), (13, 512, 1731), (13, 512, 2917))


def exact_one_particle(Gamma, Q, E, tau, K):
    """Analytic mean and second moment of the normalized one-particle J."""
    h = 5. / K
    sigma = np.sqrt(2 * tau / Gamma)
    px = gaussian_bin_kernel(np.arange(K) * h - Q * E * tau / Gamma, h, sigma)
    py = gaussian_bin_kernel(np.arange(K) * h, h, sigma)
    p = px[:, None] * py[None, :]
    return float(np.sum(p * current_weights(K))), float(1 - p[0, 0])


def analytic_tests():
    """Original unsymmetrized Sobol convergence check, with fixed seed 1731."""
    rows = []
    for Gamma, Q, E, tau, K in ANALYTIC_CASES:
        mean, second = exact_one_particle(Gamma, Q, E, tau, K)
        for power in (9, 12, 16, 20):
            r = moments(np.zeros((2, 1, 2)), 1, K, Gamma, Q, E, tau,
                        origins=1, power=power, seed=1731, reflect_y=False)
            rows.append(dict(Gamma=Gamma, Q=Q, E=E, tau=tau, K=K,
                             power=power, mean=r["mean"],
                             second=r["second_moment"], exact_mean=mean,
                             exact_second=second, mean_error=r["mean"] - mean,
                             second_error=r["second_moment"] - second,
                             seconds=r["seconds"]))
    return rows


def direct_count_reference():
    """Compare every origin with explicit N-squared pair signs."""
    y = np.random.default_rng(313).uniform(-3, 3, (5, 4, 2))
    lag, K, Gamma, Q, E, tau, power, seed = 2, 5, 9, 4, .7, .5, 8, 191
    points = qmc.Sobol(4, scramble=True, seed=seed).random_base2(power)
    X = 5 * points[:, :2]
    delta = np.sqrt(2 * tau / Gamma) * ndtri(points[:, 2:])
    delta[:, 0] += Q * E * tau / Gamma
    rows = []
    for reflect in (False, True):
        r = moments(y, lag, K, Gamma, Q, E, tau, origins=[0, 1, 2],
                    power=power, seed=seed, reflect_y=reflect)
        reference = []
        for t in (0, 1, 2):
            a = (y[t][None, :, :] + X[:, None, :]) % 5
            b = (y[t + lag][None, :, :] + (X + delta)[:, None, :]) % 5
            ia = np.floor(a * K / 5).astype(int)
            ib = np.floor(b * K / 5).astype(int)
            la = ia[:, :, 1] * K + ia[:, :, 0]
            lb = ib[:, :, 1] * K + ib[:, :, 0]
            J = np.sign(lb[:, :, None] - la[:, None, :]).mean((1, 2))
            if reflect:
                lar = (K - 1 - ia[:, :, 1]) * K + ia[:, :, 0]
                lbr = (K - 1 - ib[:, :, 1]) * K + ib[:, :, 0]
                Jr = np.sign(lbr[:, :, None] - lar[:, None, :]).mean((1, 2))
                mu = .5 * (J + Jr)
                sec = .5 * (J * J + Jr * Jr)
            else:
                mu, sec = J, J * J
            reference.append((mu.mean(), sec.mean()))
        reference = np.asarray(reference)
        actual = np.column_stack((r["per_origin_mean"], r["per_origin_second"]))
        error = float(np.max(np.abs(actual - reference)))
        assert error < 1e-14, (reflect, error)
        rows.append(dict(reflect_y=reflect, max_error=error))
    return rows


def pilot_refinement(runs):
    """Refine both origin count and QMC point count, then change the scramble."""
    rows = []
    for name in PILOT_NAMES:
        meta, _, y, _ = load_run(runs / name)
        for tau in PILOT_LAGS:
            shift = round(tau / .05)
            H, _ = histogram(y, shift, 256 if meta["N"] == 20 else 512, stride=10)
            for K in (5, 20):
                print(f"Refining {name}, tau={tau:g}, K={K}", flush=True)
                p = probabilities(H, meta, meta["E"], tau, K)
                mean = current_mean(p)
                for power, origins, seed in PILOT_BUDGETS:
                    r = moments(y, shift, K, meta["Gamma"], meta["Q"], meta["E"],
                                tau, origins=origins, power=power, seed=seed,
                                reflect_y=True)
                    rows.append(dict(name=name, tau=tau, K=K, power=power,
                                     origins=origins, seed=seed, exact_mean=mean,
                                     mean=r["mean"], second=r["second_moment"],
                                     tur=tur_rate(mean, r["second_moment"], tau),
                                     seconds=r["seconds"]))
    return rows


def refinement_summary(rows):
    summary = []
    for name in PILOT_NAMES:
        for K in (5, 20):
            for tau in PILOT_LAGS:
                chosen = [r for r in rows if r["name"] == name and
                          r["K"] == K and r["tau"] == tau]
                a = next(r for r in chosen if r["power"] == 11)
                b = [r for r in chosen if r["power"] == 13]
                ref = np.mean([r["tur"] for r in b])
                summary.append(dict(
                    name=name, K=K, tau=tau,
                    relative_tur_change=float(abs(a["tur"] - ref) / ref),
                    refinement_seed_relative_difference=float(
                        abs(b[0]["tur"] - b[1]["tur"]) / ref)))
    return summary


def rigid_cluster_identity():
    args = dict(lagsteps=1, K=5, Gamma=200, Q=100, E=1., tau=1.,
                origins=1, power=16, seed=381, reflect_y=True)
    r1 = moments(np.zeros((2, 1, 2)), **args)
    r100 = moments(np.zeros((2, 100, 2)), **args)
    result = dict(mean_difference=r1["mean"] - r100["mean"],
                  second_moment_difference=r1["second_moment"] - r100["second_moment"])
    assert result["mean_difference"] == 0.
    assert result["second_moment_difference"] == 0.
    return result


def final_reflection_analytic_tests():
    rows = []
    for Gamma, Q, E, tau, K in ANALYTIC_CASES:
        mean, second = exact_one_particle(Gamma, Q, E, tau, K)
        r = moments(np.zeros((2, 1, 2)), 1, K, Gamma, Q, E, tau,
                    origins=1, power=20, seed=1731, reflect_y=True)
        error = abs(r["second_moment"] - second)
        assert error < 2e-4, (Gamma, Q, E, tau, K, error)
        rows.append(dict(Gamma=Gamma, Q=Q, E=E, tau=tau, K=K,
                         exact_mean=mean, exact_second=second,
                         qmc_mean=r["mean"], qmc_second=r["second_moment"],
                         absolute_second_error=error))
    return rows


def main():
    default_runs = HERE.parent / "raw"
    if not default_runs.exists():
        default_runs = HERE.parent / "engine" / "raw"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=default_runs)
    parser.add_argument("--output", type=Path, default=HERE / "moment_tests.json")
    args = parser.parse_args()
    print("Checking analytic limits and direct count reference", flush=True)
    result = dict(analytic_tests=analytic_tests(),
                  direct_count_reference=direct_count_reference())
    result["pilot_refinement"] = pilot_refinement(args.runs)
    result["refinement_summary"] = refinement_summary(result["pilot_refinement"])
    result["rigid_cluster_identity"] = rigid_cluster_identity()
    result["final_reflection_analytic_tests"] = final_reflection_analytic_tests()
    result["integration_definition"] = (
        "Physical E[J²], analytically paired y-reflection, uniform X0 and Gaussian CM "
        "increment sampled with scrambled Sobol; analytic C supplies final mean.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    summary = result["refinement_summary"]
    print(f"Saved {args.output}")
    print("Maximum relative TUR change:", max(r["relative_tur_change"] for r in summary))
    print("Maximum refined-seed relative difference:",
          max(r["refinement_seed_relative_difference"] for r in summary))


if __name__ == "__main__":
    main()
