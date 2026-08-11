"""
run_noise_experiment.py

Sweeps noise levels, fits (c, k) via least-squares at each level
across multiple random seeds, and records the recovery error.
This produces the data behind the "error vs. noise" figure that
sets up the case for Bayesian inference in step 3: as noise grows,
point estimates drift and least-squares gives no sense of how much
to trust them.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from simulate import OscillatorParams, simulate
from noise import NoiseConfig, corrupt_trajectory
from fit_leastsquares import fit_leastsquares


def run_experiment(
    true_params: OscillatorParams,
    x0: float,
    v0: float,
    t_span: tuple,
    n_points: int,
    sigma_levels: list,
    n_seeds: int = 10,
    missing_frac: float = 0.0,
):
    """
    For each sigma in sigma_levels, fit (c, k) across n_seeds
    independent noise draws and record errors.

    Returns a dict: {sigma: {"c_errors": [...], "k_errors": [...]}}
    """
    t_clean, x_clean, v_clean = simulate(true_params, x0, v0, t_span, n_points)

    results = {}
    for sigma in sigma_levels:
        c_errors, k_errors = [], []
        for seed in range(n_seeds):
            config = NoiseConfig(sigma=sigma, missing_frac=missing_frac, seed=seed)
            t_obs, x_obs = corrupt_trajectory(t_clean, x_clean, config)

            try:
                c_hat, k_hat, result = fit_leastsquares(t_obs, x_obs, x0, v0)
                if not result.success:
                    continue
                c_errors.append(abs(c_hat - true_params.c))
                k_errors.append(abs(k_hat - true_params.k))
            except Exception:
                continue  # skip failed fits rather than crash the sweep

        results[sigma] = {"c_errors": c_errors, "k_errors": k_errors}

    return results


if __name__ == "__main__":
    true_params = OscillatorParams(m=1.0, c=0.3, k=4.0)
    sigma_levels = [0.01, 0.05, 0.1, 0.2, 0.4]

    results = run_experiment(
        true_params,
        x0=1.0, v0=0.0,
        t_span=(0, 20), n_points=300,
        sigma_levels=sigma_levels,
        n_seeds=15,
    )

    print(f"{'sigma':>8} {'mean |c_err|':>14} {'mean |k_err|':>14} {'n_success':>10}")
    for sigma in sigma_levels:
        c_errs = results[sigma]["c_errors"]
        k_errs = results[sigma]["k_errors"]
        print(f"{sigma:8.2f} {np.mean(c_errs):14.4f} {np.mean(k_errs):14.4f} {len(c_errs):10d}")