"""Temporal GMST trajectory generators.

Two flavours:

1. :func:`generate_gmst_trajectory` — phenomenological (linear/exponential/
   logistic trend + AR(1) red noise).
2. :func:`generate_physical_gmst` — 0-D energy balance model integrated via
   :func:`scipy.integrate.solve_ivp`, with radiative forcing from GHGs,
   solar, volcanic pulses, and Ornstein-Uhlenbeck stochastic noise.

The governing ODE is

.. math:: C \\, dT/dt = F(t) - \\lambda T

where :math:`F(t) = F_\\mathrm{ghg}(t) + F_\\mathrm{solar}(t) + F_\\mathrm{volc}(t)
+ \\varepsilon(t)` and :math:`\\varepsilon` is red noise.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import xarray as xr
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d


# ==============================================================================
# 1. TEMPORAL MODULE: GMST & TREND GENERATION
# ==============================================================================


def generate_gmst_trajectory(
    n_years: int = 40,
    start_year: int = 1981,
    trend_type: Literal["linear", "exponential", "logistic"] = "linear",
    noise_std: float = 0.05,
    seed: int = 42,
) -> xr.DataArray:
    """
    Generates a synthetic Global Mean Surface Temperature (GMST) anomaly curve.

    Args:
        trend_type: Shape of the warming curve.
    """
    rng = np.random.default_rng(seed)
    years = np.arange(start_year, start_year + n_years)
    t = np.linspace(0, 1, n_years)  # Normalized time 0 to 1

    # Deterministic Trend Component
    if trend_type == "linear":
        # Simple linear warming: +0.8C over period
        trend = 0.1 + 0.8 * t
    elif trend_type == "exponential":
        # Accelerating warming
        trend = 0.1 + 0.8 * (t**2)
    elif trend_type == "logistic":
        # S-curve (stabilization scenario)
        trend = 1.0 / (1 + np.exp(-10 * (t - 0.5)))

    # Add AR(1) red noise to mimic internal climate variability
    noise = np.zeros(n_years)
    epsilon = rng.normal(0, noise_std, n_years)
    alpha = 0.6  # Autocorrelation factor

    noise[0] = epsilon[0]
    for i in range(1, n_years):
        noise[i] = alpha * noise[i - 1] + epsilon[i]

    gmst = trend + noise

    return xr.DataArray(
        gmst,
        coords={"year": years},
        dims="year",
        name="gmst",
        attrs={"description": f"Synthetic GMST ({trend_type})"},
    )


# ==============================================================================
# ADVANCED TEMPORAL MODULE: ENERGY BALANCE MODEL (ODE)
# ==============================================================================


def generate_physical_gmst(
    n_years: int = 100,
    start_year: int = 1900,
    climate_sensitivity: float = 3.0,  # Equilibrium CS (deg C for 2xCO2)
    ocean_heat_capacity: float = 10.0,  # Effective capacity (Watt-year / m2 / K)
    seed: int = 42,
) -> xr.Dataset:
    """
    Simulates Global Mean Surface Temperature (GMST) using a 0-D Energy Balance Model.

    The evolution of temperature anomaly T(t) is governed by the ODE:

      C * dT/dt = F(t) - lambda * T(t)

    Where:
      C      : Effective heat capacity of the system (ocean mixed layer).
      lambda : Climate feedback parameter (Watts / m2 / K).
      F(t)   : Total Radiative Forcing (Watts / m2).

    The feedback parameter lambda is derived from Equilibrium Climate Sensitivity (ECS):
      lambda = F_2xCO2 / ECS
      (F_2xCO2 is approx 3.7 W/m2)

    Args:
        n_years: Duration of simulation.
        climate_sensitivity: Equilibrium warming for doubled CO2.
        ocean_heat_capacity: Thermal inertia (higher = slower response).

    Returns:
        xr.Dataset containing Temperature, Total Forcing, and Components.
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------------------------
    # 1. Physics Constants & Setup
    # --------------------------------------------------------------------------
    # Exclude the terminal endpoint: `np.linspace(0, n_years, n_years * 12)`
    # includes `t = n_years`, which falls in a floor(t) = n_years bin and
    # produces an extra final-year row with a single sample. Using
    # endpoint=False keeps exactly `n_years` monthly bins per integer year.
    t_eval = np.linspace(0, n_years, n_years * 12, endpoint=False)
    F_2xCO2 = 3.7  # Radiative forcing for doubling CO2 (W/m2)

    # Calculate Feedback Parameter (lambda)
    # At equilibrium: 0 = F_2xCO2 - lambda * ECS  =>  lambda = 3.7 / ECS
    lam = F_2xCO2 / climate_sensitivity

    # --------------------------------------------------------------------------
    # 2. Define Forcing Components F(t)
    # --------------------------------------------------------------------------

    # A. Greenhouse Gases (Logarithmic relation to CO2, Logistic CO2 growth)
    #    CO2(t) ~ Logistic curve from 280ppm to 560ppm (doubling)
    #    F_ghg(t) = 5.35 * ln(CO2(t) / CO2_ref)
    def forcing_ghg(t):
        # Center the logistic rise at year 50 (relative)
        sigmoid = 1 / (1 + np.exp(-0.1 * (t - (n_years / 2))))
        # Scale forcing from 0 to F_2xCO2 approx
        return F_2xCO2 * sigmoid

    # B. Solar Cycles (11-year Schwabe cycle)
    #    Amplitude approx 0.1 W/m2
    def forcing_solar(t):
        return 0.1 * np.sin(2 * np.pi * t / 11.0)

    # C. Volcanic Eruptions (Stochastic Spikes)
    #    modeled as discrete negative impulses decaying exponentially
    n_eruptions = int(n_years / 10)  # Approx 1 per decade
    # Reserve a 5-year buffer at both ends; if the run is shorter than that
    # window (or n_eruptions < 1) fall back to an empty schedule. Without
    # this guard `rng.uniform(5, n_years - 5, ...)` raises ValueError for
    # `n_years < 10` since `high < low`.
    if n_eruptions >= 1 and n_years >= 10:
        eruption_times = np.sort(rng.uniform(5, n_years - 5, n_eruptions))
        eruption_magnitudes = rng.gamma(shape=2.0, scale=1.5, size=n_eruptions)  # W/m2
    else:
        eruption_times = np.empty(0, dtype=float)
        eruption_magnitudes = np.empty(0, dtype=float)

    def forcing_volcano(t):
        val = 0.0
        # Sum effect of all previous eruptions
        for et, mag in zip(eruption_times, eruption_magnitudes, strict=False):
            if t > et:
                # Eruptions cause cooling (negative forcing)
                # Rapid onset, slow decay (e.g. 2 year lifetime)
                dt = t - et
                decay = np.exp(-dt / 2.0)
                val -= mag * decay * (dt * 2)  # Shape pulse
        return val

    # D. Stochastic Weather/Internal Variability (Ornstein-Uhlenbeck / Red Noise)
    #    Since ODE solvers need continuous functions, we pre-generate noise
    #    and interpolate it.
    noise_dt = 0.1  # High res noise generation
    noise_steps = int(n_years / noise_dt)
    noise_t = np.linspace(0, n_years, noise_steps)
    white_noise = rng.normal(0, 0.2, size=noise_steps)

    # Generate Red Noise (AR1)
    red_noise = np.zeros_like(white_noise)
    alpha = 0.95
    for i in range(1, noise_steps):
        red_noise[i] = alpha * red_noise[i - 1] + (1 - alpha) * white_noise[i]

    # Create continuous noise function
    forcing_noise_func = interp1d(
        noise_t, red_noise, kind="linear", fill_value=0.0, bounds_error=False
    )

    # --------------------------------------------------------------------------
    # 3. Solve ODE
    # --------------------------------------------------------------------------

    def system_dynamics(t, y):
        """
        dy/dt = (1/C) * (F_total(t) - lambda * y)
        """
        T = y[0]

        # Aggregate Forcings
        F_g = forcing_ghg(t)
        F_s = forcing_solar(t)
        F_v = forcing_volcano(t)
        F_n = forcing_noise_func(t)

        F_total = F_g + F_s + F_v + F_n

        dT_dt = (F_total - lam * T) / ocean_heat_capacity
        # solve_ivp expects an array-like of the same shape as y (length 1).
        return [dT_dt]

    # Initial Condition: Start at equilibrium (0 anomaly)
    y0 = [0.0]

    sol = solve_ivp(
        system_dynamics, t_span=(0, n_years), y0=y0, t_eval=t_eval, method="RK45"
    )

    # --------------------------------------------------------------------------
    # 4. Packaging Results
    # --------------------------------------------------------------------------

    # Reconstruct forcing components for the output dataset
    # (Expensive loop, but fine for dataset generation)
    f_ghg = [forcing_ghg(t) for t in sol.t]
    f_volc = [forcing_volcano(t) for t in sol.t]
    f_solar = [forcing_solar(t) for t in sol.t]
    f_noise = [forcing_noise_func(t) for t in sol.t]
    f_total = np.array(f_ghg) + np.array(f_volc) + np.array(f_solar) + np.array(f_noise)

    years_abs = start_year + sol.t

    ds = xr.Dataset(
        coords={"time": years_abs},
        data_vars={
            "gmst": (("time",), sol.y[0]),
            "forcing_total": (("time",), f_total),
            "forcing_ghg": (("time",), f_ghg),
            "forcing_volcanic": (("time",), f_volc),
            "forcing_solar": (("time",), f_solar),
            "forcing_stochastic": (("time",), f_noise),
        },
        attrs={
            "description": "Zero-dimensional Energy Balance Model (EBM)",
            "equation": "C * dT/dt = F(t) - lambda * T",
            "climate_sensitivity": f"{climate_sensitivity} K per 2xCO2",
            "heat_capacity": f"{ocean_heat_capacity} W-yr/m2/K",
        },
    )

    # Downsample to annual means (monthly output grouped by integer year).
    # Build an explicit integer year coordinate so the resulting group
    # dimension is named "year" regardless of xarray's groupby-anonymous
    # naming behaviour.
    year_coord = np.floor(ds["time"].values).astype(int)
    ds_annual = ds.assign_coords(year=("time", year_coord)).groupby("year").mean()

    return ds_annual
