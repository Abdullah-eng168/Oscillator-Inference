"""
simulate.py

Simulates a damped harmonic oscillator:

    m * x'' + c * x' + k * x = 0

Rewritten as a first-order system for solve_ivp:
    state = [x, v]
    x' = v
    v' = (-c*v - k*x) / m

Also provides the closed-form analytical solution for the
underdamped case, so we can validate the numerical integrator
against ground truth before doing anything else.
"""
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass


@dataclass
class OscillatorParams:
    m: float  # mass (kg)
    c: float  # damping coefficient (kg/s)
    k: float  # spring constant (N/m)

    @property
    def omega0(self) -> float:
        """Natural (undamped) angular frequency."""
        return np.sqrt(self.k / self.m)

    @property
    def zeta(self) -> float:
        """Damping ratio. zeta < 1: underdamped, =1: critically damped, >1: overdamped."""
        return self.c / (2 * np.sqrt(self.m * self.k))

    @property
    def regime(self) -> str:
        z = self.zeta
        if z < 1:
            return "underdamped"
        elif np.isclose(z, 1.0):
            return "critically damped"
        else:
            return "overdamped"


def equations_of_motion(t, state, params: OscillatorParams):
    """First-order ODE system for solve_ivp."""
    x, v = state
    dxdt = v
    dvdt = (-params.c * v - params.k * x) / params.m
    return [dxdt, dvdt]


def simulate(
    params: OscillatorParams,
    x0: float,
    v0: float,
    t_span: tuple,
    n_points: int,
    method: str = "RK45",
):
    """
    Numerically integrate the oscillator.

    Returns
    -------
    t : ndarray, shape (n_points,)
    x : ndarray, shape (n_points,)  -- position
    v : ndarray, shape (n_points,)  -- velocity
    """
    t_eval = np.linspace(t_span[0], t_span[1], n_points)

    sol = solve_ivp(
        equations_of_motion,
        t_span,
        y0=[x0, v0],
        args=(params,),
        t_eval=t_eval,
        method=method,
        rtol=1e-9,
        atol=1e-9,
    )

    if not sol.success:
        raise RuntimeError(f"Integration failed: {sol.message}")

    return sol.t, sol.y[0], sol.y[1]


def analytical_underdamped(params: OscillatorParams, x0: float, v0: float, t: np.ndarray):
    """
    Closed-form solution for the underdamped case (zeta < 1):

        x(t) = exp(-zeta*omega0*t) * (A*cos(omega_d*t) + B*sin(omega_d*t))

    where omega_d = omega0 * sqrt(1 - zeta^2), and A, B are set by
    the initial conditions x0, v0.
    """
    if params.zeta >= 1:
        raise ValueError(
            f"Analytical solution here only covers the underdamped case; "
            f"got zeta={params.zeta:.3f} ({params.regime})"
        )

    omega0 = params.omega0
    zeta = params.zeta
    omega_d = omega0 * np.sqrt(1 - zeta**2)

    A = x0
    B = (v0 + zeta * omega0 * x0) / omega_d

    envelope = np.exp(-zeta * omega0 * t)
    x = envelope * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

    # velocity via product rule
    dxdt_bracket = -omega_d * A * np.sin(omega_d * t) + omega_d * B * np.cos(omega_d * t)
    v = -zeta * omega0 * x + envelope * dxdt_bracket

    return x, v


if __name__ == "__main__":
    # Quick smoke test: underdamped case, numerical vs analytical
    params = OscillatorParams(m=1.0, c=0.3, k=4.0)
    print(f"omega0={params.omega0:.3f}, zeta={params.zeta:.3f}, regime={params.regime}")

    t, x_num, v_num = simulate(params, x0=1.0, v0=0.0, t_span=(0, 20), n_points=500)
    x_ana, v_ana = analytical_underdamped(params, x0=1.0, v0=0.0, t=t)

    max_err_x = np.max(np.abs(x_num - x_ana))
    max_err_v = np.max(np.abs(v_num - v_ana))
    print(f"max abs error (position): {max_err_x:.2e}")
    print(f"max abs error (velocity): {max_err_v:.2e}")