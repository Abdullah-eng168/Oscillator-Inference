"""
plot_diagnostics.py

Runs the MCMC fit and produces three diagnostic figures:

1. Trace plot -- the single chain's (c, k, log_sigma) over the run, to
   visually check the sampler has converged / mixed well (should
   look like a "fuzzy caterpillar", not wandering or stuck).
2. Corner-style posterior plot -- 1D marginal histograms on the
   diagonal, 2D histograms off-diagonal, to see parameter
   correlations (c and k are typically correlated).
3. Posterior predictive check -- noisy data overlaid with
   trajectories drawn from posterior samples, to sanity-check the
   fit actually explains the data.

(corner isn't installable in this environment, so trace and corner
plots are hand-built with matplotlib.)
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from simulate import OscillatorParams, simulate
from noise import NoiseConfig, corrupt_trajectory
from fit_leastsquares import fit_leastsquares
from fit_mcmc import fit_mcmc, forward_model, summarize

OUTDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(OUTDIR, exist_ok=True)


def plot_trace(chain, burn_in, names=("c", "k", "log_sigma"), path=None):
    n_steps, n_dim = chain.shape
    fig, axes = plt.subplots(n_dim, 1, figsize=(9, 2.2 * n_dim), sharex=True, squeeze = False)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        ax.plot(chain[:, i], color="steelblue", lw=0.5)
        ax.axvline(burn_in, color="firebrick", ls="--", lw=1, label="burn-in cutoff")
        ax.set_ylabel(names[i])
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle("MCMC trace plot")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_corner(samples, names=("c", "k", "sigma"), truths=None, path=None):
    n_dim = samples.shape[1]
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(8, 8), squeeze = False)
    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            if i == j:
                ax.hist(samples[:, i], bins=40, color="steelblue", alpha=0.8)
                if truths is not None:
                    ax.axvline(truths[i], color="firebrick", ls="--", lw=1.5)
                # diagonal panels are 1D marginals: label both axes directly
                # rather than relying on a shared edge label, so every panel
                # (including the top-left one) is self-explanatory
                ax.set_xlabel(names[i])
                ax.set_ylabel("count")
            elif i > j:
                ax.hist2d(samples[:, j], samples[:, i], bins=40, cmap="Blues")
                if truths is not None:
                    ax.axvline(truths[j], color="firebrick", ls="--", lw=1)
                    ax.axhline(truths[i], color="firebrick", ls="--", lw=1)
                ax.set_xlabel(names[j])
                ax.set_ylabel(names[i])
            else:
                ax.axis("off")
    fig.suptitle("Posterior distributions (dashed line = true value)")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_posterior_predictive(
    t_obs, x_obs, x0, v0, samples, true_params, n_draws=60, path=None
):
    fig, ax = plt.subplots(figsize=(9, 5))
    t_fine = np.linspace(t_obs.min(), t_obs.max(), 500)

    idx = np.random.default_rng(1).choice(len(samples), size=n_draws, replace=False)
    for i in idx:
        c, k, sigma = samples[i]
        x_pred = forward_model(c, k, t_fine, x0, v0)
        ax.plot(t_fine, x_pred, color="steelblue", alpha=0.08, lw=1)

    x_true = forward_model(true_params.c, true_params.k, t_fine, x0, v0)
    ax.plot(t_fine, x_true, color="black", lw=1.5, label="true trajectory")
    ax.scatter(t_obs, x_obs, color="firebrick", s=14, zorder=5, label="noisy observations")
    ax.plot([], [], color="steelblue", alpha=0.5, lw=1.5, label=f"{n_draws} posterior draws")

    ax.set_xlabel("t")
    ax.set_ylabel("x(t)")
    ax.set_title("Posterior predictive check")
    ax.legend()
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    true_params = OscillatorParams(m=1.0, c=0.3, k=4.0)
    x0, v0 = 1.0, 0.0
    t, x_clean, v_clean = simulate(true_params, x0, v0, t_span=(0, 20), n_points=300)

    noise_config = NoiseConfig(sigma=0.05, missing_frac=0.05, seed=42)
    t_obs, x_obs = corrupt_trajectory(t, x_clean, noise_config)

    c_ls, k_ls, _ = fit_leastsquares(t_obs, x_obs, x0, v0)

    samples, chain, acc_frac = fit_mcmc(
        t_obs, x_obs, x0, v0,
        init_guess=(c_ls, k_ls, np.log(0.05)),
        step_size=(0.007, 0.013, 0.038),
        n_steps=50000,
        burn_in=5000,
        seed=0,
    )

    print(f"acceptance fraction: {acc_frac:.3f}")
    summarize(samples)

    plot_trace(chain, burn_in=5000, path=os.path.join(OUTDIR, "mcmc_trace.png"))
    plot_corner(
        samples,
        truths=(true_params.c, true_params.k, noise_config.sigma),
        path=os.path.join(OUTDIR, "mcmc_corner.png"),
    )
    plot_posterior_predictive(
        t_obs, x_obs, x0, v0, samples, true_params,
        path=os.path.join(OUTDIR, "mcmc_posterior_predictive.png"),
    )
    print("saved: mcmc_trace.png, mcmc_corner.png, mcmc_posterior_predictive.png")
