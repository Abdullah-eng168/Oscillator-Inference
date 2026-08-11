"""
fit_leastsquares.py

Recovers oscillator parameters (c, k) from noisy position
observations via nonlinear least-squares.

Note on identifiability: given only noisy position data x(t) (no
external forcing, no absolute force/mass measurement), the dynamics
only depend on the ratios c/m and k/m -- you cannot uniquely
recover m on its own. We therefore fix m=1 and only fit c, k.
"""

import numpy as np
from scipy.optimize import least_squares
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from simulate import OscillatorParams, simulate


def residuals(theta, t_obs, x_obs, x0, v0, m_fixed=1.0):
    """
    theta = [c, k] -- the parameters being fit.
    Simulates the trajectory for candidate (c, k) and returns the
    residual against the noisy observations.
    """
    c, k = theta
    if c < 0 or k <= 0:
        # keep the optimizer in a physically valid region
        return np.full_like(x_obs, 1e6)

    params = OscillatorParams(m=m_fixed, c=c, k=k)
    t_span = (t_obs.min(), t_obs.max())

    # simulate on a fine grid, then interpolate onto the observed
    # timestamps -- t_obs may be irregular/sparse due to missingness
    t_fine = np.linspace(t_span[0], t_span[1], max(500, 3 * len(t_obs)))
    _, x_fine, _ = simulate(params, x0, v0, t_span, len(t_fine))
    x_pred = np.interp(t_obs, t_fine, x_fine)

    return x_pred - x_obs


def fit_leastsquares(t_obs, x_obs, x0, v0, initial_guess=(1.0, 1.0), m_fixed=1.0):
    """
    Fit (c, k) to noisy observations via nonlinear least-squares.

    Returns
    -------
    c_hat, k_hat : recovered parameters
    result : the full scipy OptimizeResult (for diagnostics)
    """
    result = least_squares(
        residuals,
        x0=initial_guess,
        args=(t_obs, x_obs, x0, v0, m_fixed),
        bounds=([0, 1e-6], [np.inf, np.inf]),
    )
    c_hat, k_hat = result.x
    return c_hat, k_hat, result


if __name__ == "__main__":
    from noise import NoiseConfig, corrupt_trajectory

    # ground truth
    true_params = OscillatorParams(m=1.0, c=0.3, k=4.0)
    x0, v0 = 1.0, 0.0
    t, x_clean, v_clean = simulate(true_params, x0, v0, t_span=(0, 20), n_points=300)

    # corrupt with moderate noise
    config = NoiseConfig(sigma=0.05, missing_frac=0.05, seed=42)
    t_obs, x_obs = corrupt_trajectory(t, x_clean, config)

    c_hat, k_hat, result = fit_leastsquares(t_obs, x_obs, x0, v0)

    print(f"true:      c={true_params.c:.4f}, k={true_params.k:.4f}")
    print(f"recovered: c={c_hat:.4f}, k={k_hat:.4f}")
    print(f"optimizer converged: {result.success}, cost: {result.cost:.6f}")