"""
Map an initial-guess cycle state onto pyCycle balance variables.

The design functions convert the SI cycle state from `turbofan.initial_guess` or
`turboshaft.initial_guess` to the imperial units pyCycle uses. The off-design functions implement
Eqs. (7)-(11) of Tahiliani et al., AIAA AVIATION 2026 (https://doi.org/10.2514/6.2026-4499),
scaling the design state with flight condition and throttle. Each function returns a dictionary
keyed by balance name, suitable for assigning to `outputs` in a pyCycle `guess_nonlinear`.

Geared and direct-drive turbofans share one cycle state and differ only here: a geared engine runs
the LP shaft at its stress-limited speed, with a gearbox connecting it to the fan. A direct-drive
engine runs the LP shaft at the fan speed.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import numpy as np

GAMMA = 1.4
R = 287.0  # [J/kg-K]
PA_TO_PSI = 0.0001450377
K_TO_R = 1.8
KG_TO_LBM = 2.20462
NM_TO_FTLBF = 0.7375621


def _stagnation(M, Tamb, Pamb):
    """Return the isentropic total temperature [K] and total pressure [Pa]."""
    isen = 1.0 + 0.5 * (GAMMA - 1.0) * M**2
    return Tamb * isen, Pamb * isen ** (GAMMA / (GAMMA - 1.0))


def _design_common(cycle, flow):
    """Return the design balances shared by the turbofan and the turboshaft."""
    return {
        'balance.W': cycle[flow] * KG_TO_LBM,
        'balance.hpt_PR': cycle['pi_hpt'],
        'balance.lpt_PR': cycle['pi_lpt'],
        'balance.FAR': cycle['FAR'],
        'balance.cool_NGV_frac_W': cycle['coolNGV_frac_W'],
        'balance.cool_turb_frac_W': cycle['coolRotor1_frac_W']
        + cycle['coolStator2_frac_W']
        + cycle['coolRotor2_frac_W'],
        'fc.balance.Pt': cycle['Pt0'] * PA_TO_PSI,
        'fc.balance.Tt': cycle['Tt0'] * K_TO_R,
    }


def turbofan_design_guess(cycle, geared=False):
    """
    Compute the design balance guesses of a turbofan.

    Parameters
    ----------
    cycle : dict
        Cycle state from `turbofan.initial_guess`.
    geared : bool
        If True, also seed the gearbox torque [ft*lbf] from the fan power and speed.

    Returns
    -------
    dict
        Guess for each design balance.
    """
    guess = _design_common(cycle, 'W_total')
    if geared:
        omega_fan = 2.0 * np.pi * cycle['N_fan'] / 60.0
        guess['balance.gb_trq'] = cycle['Wdot_fan'] / omega_fan * NM_TO_FTLBF
    return guess


def gear_ratio(cycle):
    """
    Compute the gear ratio set by the design LP shaft and fan speeds.

    Parameters
    ----------
    cycle : dict
        Cycle state from `turbofan.initial_guess`.

    Returns
    -------
    float
        Fan speed divided by LP shaft speed, matching `gear_ratio` of pyCycle's Gearbox (N_out/N_in).
    """
    return cycle['N_fan'] / cycle['LP_Nmech_max']


def turbofan_off_design_guess(
    cycle, M, Tamb, Pamb, T4_max, fan_area, geared=False, M_fan=0.6, bypass_k=0.5, N_fan_des=None
):
    """
    Compute the off-design balance guesses of a turbofan.

    Parameters
    ----------
    cycle : dict
        Design cycle state from `turbofan.initial_guess`.
    M : float
        Freestream Mach number.
    Tamb : float
        Ambient static temperature [K].
    Pamb : float
        Ambient static pressure [Pa].
    T4_max : float
        Local turbine inlet temperature target [K]. Throttle is estimated as its ratio to the
        design value.
    fan_area : float
        Design fan face area [m**2].
    geared : bool
        If True, the LP shaft and the fan have separate speeds.
    M_fan : float
        Assumed fan face Mach number.
    bypass_k : float
        Tuning parameter k in the bypass ratio scaling, Eq. (11).
    N_fan_des : float or None
        Converged design fan speed [rpm]. Defaults to the fan speed of the initial guess.

    Returns
    -------
    dict
        Guess for each off-design balance.
    """
    Tt2, Pt2 = _stagnation(M, Tamb, Pamb)
    isen_fan = 1.0 + 0.5 * (GAMMA - 1.0) * M_fan**2
    W = (
        (fan_area * Pt2 / np.sqrt(Tt2))
        * np.sqrt(GAMMA / R)
        * M_fan
        * isen_fan ** (-(GAMMA + 1) / (2 * (GAMMA - 1)))
    )

    throttle = T4_max / cycle['Tt4'] if np.real(cycle['Tt4']) > 0.0 else 1.0
    speed_scale = (Tt2 / cycle['Tt0']) ** 0.5 * throttle

    N_fan = (cycle['N_fan'] if N_fan_des is None else N_fan_des) * speed_scale
    guess = {
        'balance.W': W * KG_TO_LBM,
        'balance.FAR': cycle['FAR'] * throttle,
        'balance.BPR': cycle['BPR'] * (1.0 + bypass_k * (1.0 - throttle)),
        'fc.balance.Pt': Pt2 * PA_TO_PSI,
        'fc.balance.Tt': Tt2 * K_TO_R,
        'balance.HP_Nmech': cycle['HP_Nmech_max'] * speed_scale,
    }
    if geared:
        guess['balance.LP_Nmech'] = cycle['LP_Nmech_max'] * speed_scale
        guess['balance.FAN_Nmech'] = N_fan
    else:
        guess['balance.LP_Nmech'] = N_fan
    return guess


def turboshaft_design_guess(cycle):
    """
    Compute the design balance guesses of a turboshaft.

    Parameters
    ----------
    cycle : dict
        Cycle state from `turboshaft.initial_guess`.

    Returns
    -------
    dict
        Guess for each design balance.
    """
    guess = _design_common(cycle, 'W')
    guess['balance.pt_PR'] = cycle['pi_pt']
    return guess


def turboshaft_off_design_guess(cycle, M, Tamb, Pamb, T4_target):
    """
    Compute the off-design balance guesses of a turboshaft.

    Mass flow holds corrected flow constant and is bounded below by the design flow: at high
    altitude the corrected-flow estimate undershoots and starts the compressors near surge. Shaft
    speeds hold corrected speed constant.

    Parameters
    ----------
    cycle : dict
        Design cycle state from `turboshaft.initial_guess`.
    M : float
        Freestream Mach number.
    Tamb : float
        Ambient static temperature [K].
    Pamb : float
        Ambient static pressure [Pa].
    T4_target : float
        Target turbine inlet temperature [K].

    Returns
    -------
    dict
        Guess for each off-design balance.
    """
    Tt0, Pt0 = _stagnation(M, Tamb, Pamb)
    theta_ratio = Tt0 / cycle['Tt0']
    delta_ratio = Pt0 / cycle['Pt0']
    throttle = T4_target / cycle['Tt4'] if np.real(cycle['Tt4']) > 0.0 else 1.0
    speed_ratio = theta_ratio**0.5

    return {
        'fc.balance.Pt': Pt0 * PA_TO_PSI,
        'fc.balance.Tt': Tt0 * K_TO_R,
        'balance.FAR': cycle['FAR'] * throttle,
        'balance.W': np.maximum(cycle['W_lbs'] * delta_ratio / speed_ratio, cycle['W_lbs']),
        'balance.HP_Nmech': cycle['HP_Nmech_max'] * speed_ratio,
        'balance.IP_Nmech': cycle['IP_Nmech_max'] * speed_ratio,
    }
