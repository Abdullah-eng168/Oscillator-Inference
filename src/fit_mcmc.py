"""
fit_mcmc.py

Recovers oscillator parameters (c, k) -- plus the measurement noise
level sigma -- from noisy position observations via Bayesian MCMC.

Unlike fit_leastsquares.py, this gives you full posterior
distributions (not just point estimates), so you get calibrated
uncertainty on c and k, and can see parameter correlations directly.

Sampler
-------
Plain single-chain random-walk Metropolis-Hastings: propose a new
point by adding Gaussian noise to the current point, then accept or
reject based on the posterior ratio. The proposal is symmetric, so
the Hastings correction term is just 1 (cancels out in log-space).
No walkers, no ensemble -- one chain, tuned by hand via step_size
until the acceptance fraction lands in a healthy 0.2-0.5 range.

Model
-----
theta = [c, k, log_sigma]

Priors (weakly informative, just enough to keep the sampler in a
physically sensible region):
    c         ~ Uniform(0, 10)
    k         ~ Uniform(1e-6, 20)
    log_sigma ~ Uniform(log(1e-4), log(2.0))

Likelihood: Gaussian iid noise on each observed position, with sigma
itself inferred (rather than assumed known, which is unrealistic --
in real life you don't know your sensor noise ahead of time).
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from simulate import OscillatorParams, simulate


# ---------------------------------------------------------------
# Forward model + posterior
# ---------------------------------------------------------------
#
# MCMC needs tens of thousands of forward-model evaluations (n_walkers
# x n_steps), so calling solve_ivp with tight tolerances on a fine
# grid + interpolating (as fit_leastsquares.py does, which is fine
# for a single optimizer run) is far too slow here. Instead we use
# the exact closed-form solution of m*x'' + c*x' + k*x = 0, covering
# all three damping regimes -- it's exact (no interpolation error)
# and vectorized directly at the observed timestamps.

def analytic_position(c, k, t, x0, v0, m=1.0):
    """
    Closed-form position x(t) for m*x'' + c*x' + k*x = 0, valid for
    under-, critically-, and over-damped regimes. `t` can be an array.
    """
    omega0 = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    eps = 1e-6

    if zeta < 1 - eps:
        omega_d = omega0 * np.sqrt(1 - zeta**2)
        A = x0
        B = (v0 + zeta * omega0 * x0) / omega_d
        env = np.exp(-zeta * omega0 * t)
        return env * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

    elif zeta > 1 + eps:
        omega0_shift = omega0 * np.sqrt(zeta**2 - 1)
        r1 = -zeta * omega0 + omega0_shift
        r2 = -zeta * omega0 - omega0_shift
        C1 = (v0 - r2 * x0) / (r1 - r2)
        C2 = x0 - C1
        return C1 * np.exp(r1 * t) + C2 * np.exp(r2 * t)

    else:  # critically damped (or within eps of it)
        r = -omega0
        C1 = x0
        C2 = v0 - r * x0
        return (C1 + C2 * t) * np.exp(r * t)


def forward_model(c, k, t_obs, x0, v0, m_fixed=1.0):
    """Closed-form trajectory for (c, k), evaluated at the observed times."""
    return analytic_position(c, k, t_obs, x0, v0, m_fixed)


# prior bounds -- module-level so both log_prior and the sampler init can see them
C_MIN, C_MAX = 0.0, 10.0
K_MIN, K_MAX = 1e-6, 20.0
LOGSIG_MIN, LOGSIG_MAX = np.log(1e-4), np.log(2.0)


def log_prior(theta):
    c, k, log_sigma = theta
    if not (C_MIN <= c <= C_MAX):
        return -np.inf
    if not (K_MIN <= k <= K_MAX):
        return -np.inf
    if not (LOGSIG_MIN <= log_sigma <= LOGSIG_MAX):
        return -np.inf
    return 0.0  # flat within bounds


def log_likelihood(theta, t_obs, x_obs, x0, v0, m_fixed=1.0):
    c, k, log_sigma = theta
    sigma = np.exp(log_sigma)
    try:
        x_pred = forward_model(c, k, t_obs, x0, v0, m_fixed)
    except Exception:
        return -np.inf
    if not np.all(np.isfinite(x_pred)):
        return -np.inf
    resid = x_obs - x_pred
    n = len(x_obs)
    return -0.5 * np.sum(resid**2) / sigma**2 - n * np.log(sigma) - 0.5 * n * np.log(2 * np.pi)


def log_posterior(theta, t_obs, x_obs, x0, v0, m_fixed=1.0):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t_obs, x_obs, x0, v0, m_fixed)


# ---------------------------------------------------------------
# Single-chain random-walk Metropolis-Hastings
# ---------------------------------------------------------------

def metropolis_hastings(
    log_prob_fn,
    theta_init,
    n_steps,
    step_size,
    rng=None,
    args=(),
):
    """
    Plain random-walk Metropolis-Hastings.

    At each step: propose theta_new = theta_current + N(0, step_size),
    then accept with probability min(1, posterior(new) / posterior(old))
    -- done in log-space as accept if log(uniform) < log_post(new) - log_post(old).
    The proposal is symmetric, so there's no Hastings correction term.

    Parameters
    ----------
    log_prob_fn : callable(theta, *args) -> float
    theta_init : starting point, shape (n_dim,)
    n_steps : number of steps to run
    step_size : scalar or array, std dev of the Gaussian proposal per parameter
    args : extra args passed through to log_prob_fn

    Returns
    -------
    chain : ndarray, shape (n_steps, n_dim)
    log_prob_chain : ndarray, shape (n_steps,)
    acceptance_fraction : float
    """
    rng = np.random.default_rng() if rng is None else rng
    theta_current = np.array(theta_init, dtype=float)
    n_dim = len(theta_current)

    log_prob_current = log_prob_fn(theta_current, *args)

    chain = np.zeros((n_steps, n_dim))
    log_prob_chain = np.zeros(n_steps)
    n_accepted = 0

    for step in range(n_steps):
        theta_proposal = theta_current + rng.normal(0.0, step_size, size=n_dim)
        log_prob_proposal = log_prob_fn(theta_proposal, *args)

        log_accept_ratio = log_prob_proposal - log_prob_current
        if np.log(rng.random()) < log_accept_ratio:
            theta_current = theta_proposal
            log_prob_current = log_prob_proposal
            n_accepted += 1

        chain[step] = theta_current
        log_prob_chain[step] = log_prob_current

    acceptance_fraction = n_accepted / n_steps
    return chain, log_prob_chain, acceptance_fraction


# ---------------------------------------------------------------
# High-level fitting function
# ---------------------------------------------------------------

def fit_mcmc(
    t_obs,
    x_obs,
    x0,
    v0,
    m_fixed=1.0,
    init_guess=(0.3, 4.0, np.log(0.05)),
    step_size=(0.007, 0.013, 0.038),
    n_steps=50000,
    burn_in=5000,
    seed=0,
):
    """
    Fit (c, k, sigma) to noisy observations via single-chain MCMC.

    step_size controls how far each proposal moves from the current
    point -- tune it so acceptance_fraction lands around 0.2-0.5. Too
    small -> accepts almost everything but explores very slowly; too
    large -> rejects almost everything.

    Returns
    -------
    samples : ndarray, shape (n_samples, 3) -- post-burn-in, columns
              are [c, k, sigma] (sigma already exponentiated back out
              of log-space for convenience)
    chain : ndarray, shape (n_steps, 3) -- raw chain in
            [c, k, log_sigma] space, for trace/diagnostic plots
    acceptance_fraction : float
    """
    rng = np.random.default_rng(seed)
    theta_init = np.array(init_guess)

    chain, log_prob_chain, acc_frac = metropolis_hastings(
        log_posterior,
        theta_init,
        n_steps,
        step_size,
        rng=rng,
        args=(t_obs, x_obs, x0, v0, m_fixed),
    )

    post_burn = chain[burn_in:]  # (n_steps - burn_in, 3)
    samples = post_burn.copy()
    samples[:, 2] = np.exp(samples[:, 2])  # log_sigma -> sigma

    return samples, chain, acc_frac


def summarize(samples, names=("c", "k", "sigma")):
    """Print posterior median and 68% credible interval for each parameter."""
    for i, name in enumerate(names):
        lo, med, hi = np.percentile(samples[:, i], [16, 50, 84])
        print(f"{name}: {med:.4f}  (+{hi - med:.4f} / -{med - lo:.4f})  [68% CI]")


if __name__ == "__main__":
    from noise import NoiseConfig, corrupt_trajectory
    from fit_leastsquares import fit_leastsquares

    # ground truth
    true_params = OscillatorParams(m=1.0, c=0.3, k=4.0)
    x0, v0 = 1.0, 0.0
    t, x_clean, v_clean = simulate(true_params, x0, v0, t_span=(0, 20), n_points=300)

    # corrupt with moderate noise
    noise_config = NoiseConfig(sigma=0.05, missing_frac=0.05, seed=42)
    t_obs, x_obs = corrupt_trajectory(t, x_clean, noise_config)

    # least-squares point estimate first, to anchor the walker initialization
    c_ls, k_ls, _ = fit_leastsquares(t_obs, x_obs, x0, v0)
    print(f"least-squares point estimate: c={c_ls:.4f}, k={k_ls:.4f}\n")

    print("running MCMC...")
    samples, chain, acc_frac = fit_mcmc(
        t_obs, x_obs, x0, v0,
        init_guess=(c_ls, k_ls, np.log(0.05)),
        step_size=(0.007, 0.013, 0.038),
        n_steps=50000,
        burn_in=5000,
        seed=0,
    )

    print(f"acceptance fraction: {acc_frac:.3f}  (healthy range is roughly 0.2-0.5)\n")
    print(f"true:      c={true_params.c:.4f}, k={true_params.k:.4f}, sigma={noise_config.sigma:.4f}\n")
    print("posterior summary:")
    summarize(samples)
