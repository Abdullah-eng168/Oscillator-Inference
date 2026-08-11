"""
noise.py

Corrupts a clean simulated trajectory the way a real sensor would:
- Gaussian measurement noise on position readings
- Optional constant sensor bias
- Random missingness (dropped samples)

Real sensors typically only measure position directly (e.g. a
displacement sensor, a camera tracking a marker) -- not velocity --
so this module only touches x, not v. That constraint is deliberate.
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class NoiseConfig:
    sigma: float = 0.05       # std dev of Gaussian measurement noise
    bias: float = 0.0         # constant systematic offset
    missing_frac: float = 0.0  # fraction of samples to randomly drop
    seed: int | None = None   # for reproducibility


def add_measurement_noise(x: np.ndarray, config: NoiseConfig) -> np.ndarray:
    """Add Gaussian noise + constant bias to a clean position signal."""
    rng = np.random.default_rng(config.seed)
    noise = rng.normal(loc=0.0, scale=config.sigma, size=x.shape)
    return x + noise + config.bias


def apply_missingness(t: np.ndarray, x: np.ndarray, config: NoiseConfig):
    """
    Randomly drop a fraction of samples to simulate sensor dropout.

    Returns filtered (t, x) with missing_frac of points removed,
    keeping the original ordering.
    """
    if config.missing_frac <= 0:
        return t, x

    rng = np.random.default_rng(
        None if config.seed is None else config.seed + 1  # decorrelate from noise draw
    )
    n = len(t)
    keep_mask = rng.random(n) >= config.missing_frac
    return t[keep_mask], x[keep_mask]


def corrupt_trajectory(t: np.ndarray, x: np.ndarray, config: NoiseConfig):
    """
    Full pipeline: add measurement noise, then apply missingness.

    -------
    t_obs : ndarray -- observed timestamps (subset of t if missingness > 0)
    x_obs : ndarray -- noisy, possibly-subsampled position readings
    """
    x_noisy = add_measurement_noise(x, config)
    t_obs, x_obs = apply_missingness(t, x_noisy, config)
    return t_obs, x_obs


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from simulate import OscillatorParams, simulate

    params = OscillatorParams(m=1.0, c=0.3, k=4.0)
    t, x, v = simulate(params, x0=1.0, v0=0.0, t_span=(0, 20), n_points=200)

    config = NoiseConfig(sigma=0.05, missing_frac=0.1, seed=42)
    t_obs, x_obs = corrupt_trajectory(t, x, config)

    print(f"clean samples: {len(t)}")
    print(f"observed samples after missingness: {len(t_obs)}")
    print(f"noise std added: {config.sigma}")
    print(f"mean abs deviation from clean signal at shared timestamps: "
          f"{np.mean(np.abs(x_obs[:5] - x[:5])):.4f} (first 5 pts, pre-missingness check)")