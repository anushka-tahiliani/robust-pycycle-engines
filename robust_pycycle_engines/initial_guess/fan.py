"""
Initial guess for an electrically driven fan (no thermal core).

With no core to close, the design mass flow follows directly from the shaft power and the
enthalpy rise across the fan. Off-design, the nozzle flow function and the power balance give two
equations in mass flow and fan pressure ratio, solved by fixed-point iteration.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import numpy as np

GAMMA = 1.4
R = 287.0  # [J/kg-K]
CP = R * GAMMA / (GAMMA - 1.0)  # [J/kg-K]
HP_TO_W = 745.699872
KG_TO_LBM = 2.2046226218
IN2_TO_M2 = 6.4516e-4


def design_mass_flow(fan_power, FPR, eta_pfan, Tamb, M):
    """
    Estimate the design mass flow of a fan.

    Parameters
    ----------
    fan_power : float
        Fan shaft power [hp].
    FPR : float
        Fan pressure ratio.
    eta_pfan : float
        Fan polytropic efficiency.
    Tamb : float
        Ambient static temperature [K].
    M : float
        Freestream Mach number.

    Returns
    -------
    float
        Fan mass flow [lbm/s].
    """
    Tt0 = Tamb * (1 + (GAMMA - 1) / 2 * M**2)
    eta_fan = (FPR ** ((GAMMA - 1) / GAMMA) - 1) / (FPR ** ((GAMMA - 1) / (eta_pfan * GAMMA)) - 1)
    Tt2 = Tt0 * (1 + (FPR ** ((GAMMA - 1) / GAMMA) - 1) / eta_fan)
    return fan_power * HP_TO_W / (CP * (Tt2 - Tt0)) * KG_TO_LBM


def off_design_mass_flow(fan_power, nozzle_area, Tamb, Pamb, M, eta_pfan=0.9, FPR_seed=1.4):
    """
    Estimate the off-design mass flow of a fan from its nozzle area and power balance.

    Parameters
    ----------
    fan_power : float
        Fan shaft power [hp].
    nozzle_area : float
        Design nozzle throat area [inch**2].
    Tamb : float
        Ambient static temperature [K].
    Pamb : float
        Ambient static pressure [Pa].
    M : float
        Freestream Mach number.
    eta_pfan : float
        Assumed fan polytropic efficiency.
    FPR_seed : float
        Starting value of the fan pressure ratio.

    Returns
    -------
    W_lbm : float
        Fan mass flow [lbm/s].
    FPR : float
        Fan pressure ratio.
    """
    P_shaft = fan_power * HP_TO_W
    A_star = nozzle_area * IN2_TO_M2

    isen = 1.0 + (GAMMA - 1.0) / 2.0 * M**2
    Tt2 = Tamb * isen
    Pt2 = Pamb * isen ** (GAMMA / (GAMMA - 1.0))

    x = (GAMMA - 1.0) / (eta_pfan * GAMMA)
    NPR_crit = ((GAMMA + 1.0) / 2.0) ** (GAMMA / (GAMMA - 1.0))
    exponent = -(GAMMA + 1.0) / (2.0 * (GAMMA - 1.0))

    FPR = FPR_seed
    W = P_shaft / (CP * Tt2 * (FPR**x - 1.0))
    for _ in range(5):
        Tt7 = Tt2 * FPR**x
        Pt7 = Pt2 * FPR
        NPR = Pt7 / Pamb

        # Nozzle flow function: the throat Mach number is 1 when choked
        if np.real(NPR) >= np.real(NPR_crit):
            M_th = 1.0
        else:
            M_th = (2.0 / (GAMMA - 1.0) * (NPR ** ((GAMMA - 1.0) / GAMMA) - 1.0)) ** 0.5
        flow_fn = M_th * (GAMMA / R) ** 0.5 * (1.0 + (GAMMA - 1.0) / 2.0 * M_th**2) ** exponent
        W = A_star * Pt7 * flow_fn / Tt7**0.5

        # Update the pressure ratio from the power balance P = W * cp * (Tt7 - Tt2)
        if np.real(W) > 1e-6:
            dT = P_shaft / (W * CP)
            FPR = np.maximum(((Tt2 + dT) / Tt2) ** (1.0 / x), 1.05)

    return W * KG_TO_LBM, FPR
