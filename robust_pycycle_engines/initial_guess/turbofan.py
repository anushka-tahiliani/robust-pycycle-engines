"""
On-design initial guess for a turbofan (direct-drive or geared).

Implements the physics-based on-design initialization of Tahiliani et al.,
"Robust Approach to Engine Cycle Modeling to Enable Multidisciplinary Optimization of
Advanced Aircraft", AIAA AVIATION 2026 (https://doi.org/10.2514/6.2026-4499), Sec. IV.

For a prescribed mass flow, a Brayton cycle that mirrors the pyCycle model (turbine cooling,
shaft speeds, small-core losses) closes in closed form. The design problem therefore reduces
to a one-dimensional root find in mass flow: a coarse sweep brackets the target thrust, and
the bracket midpoint seeds a Newton solve. The returned state seeds pyCycle's balances.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import numpy as np

from robust_pycycle_engines.initial_guess._gas_path import (
    CP,
    GAMMA,
    HP_TO_W,
    R,
    burner_and_hpt,
    compress_core,
    corrected_flow_function,
    low_pressure_turbine,
)
from robust_pycycle_engines.initial_guess._solver import (
    compressor_exit,
    solve_mass_flow,
    split_pressure_ratio,
)
from robust_pycycle_engines.initial_guess.balances import KG_TO_LBM

# Fractional total pressure losses of the ducts, shared with the cycle model
DUCT_LOSSES = {'duct6': 0.0101, 'duct11': 0.0051, 'duct13': 0.0107}

LBF_TO_N = 4.448221615
FAN_FACE_MACH = 0.661
FAN_TIP_MACH = 1.2
FAN_HUB_TIP = 0.3125


def _nozzle_thrust(W, Tt, Pt, Pamb, v0):
    """Compute the net thrust [N] of a nozzle, which is choked or expands fully to ambient."""
    PR = Pt / Pamb
    PR_crit = ((GAMMA + 1) / 2) ** (GAMMA / (GAMMA - 1))
    M = np.where(
        np.real(PR) > np.real(PR_crit),
        1.0,
        np.sqrt((2 / (GAMMA - 1)) * np.maximum(PR ** ((GAMMA - 1) / GAMMA) - 1, 0.0)),
    )
    flow_function = np.where(
        corrected_flow_function(M, GAMMA) <= 1e-8, 1e-8, corrected_flow_function(M, GAMMA)
    )
    area = W * np.sqrt(CP * Tt) / (flow_function * Pt)
    T_exit = Tt / (1 + (GAMMA - 1) / 2 * M**2)
    v_exit = M * np.sqrt(GAMMA * R * T_exit)
    P_exit = Pamb
    if np.real(M) == 1:
        P_exit = Pt * (2 / (GAMMA + 1)) ** (GAMMA / (GAMMA - 1))
    return area * (P_exit - Pamb) + W * (v_exit - v0)


def initial_guess(
    opr,
    M,
    Tamb,
    Pamb,
    T_metal,
    BPR,
    TIT,
    target_lbf,
    pwr_offtk_hpt,
    pi_fan,
    pi_lpc,
    pi_centrif=1.0,
    eta_pfan=0.9,
    eta_plpc=0.9,
    eta_pcentrif=0.85,
    eta_phpc_init=0.9,
    eta_phpt=0.9,
    eta_plpt=0.9,
    in_dP=0.99,
    com_dPqP=0.054,
    coolIG_frac_W=0.0,
    W_max=1000.0,
):
    """
    Compute an initial thermodynamic cycle state for a given altitude and engine configuration.

    Parameters
    ----------
    opr : float
        Overall pressure ratio. The axial HPC supplies what the fan, LPC and centrifugal stage do
        not: pi_hpc = opr / (pi_fan * pi_lpc * pi_centrif).
    M : float
        Freestream Mach number.
    Tamb : float
        Ambient static temperature [K].
    Pamb : float
        Ambient static pressure [Pa].
    T_metal : float
        Maximum allowable metal temperature [K].
    BPR : float
        Bypass ratio.
    TIT : float
        Turbine inlet temperature [K].
    target_lbf : float
        Target thrust [lbf].
    pwr_offtk_hpt : float
        Power off-take from the HP shaft [hp].
    pi_fan : float
        Fan (root) pressure ratio.
    pi_lpc : float
        Low-pressure compressor pressure ratio.
    pi_centrif : float
        Pressure ratio of a centrifugal compressor after the axial HPC; 1 means there is none.
    eta_pfan, eta_plpc, eta_pcentrif : float
        Polytropic efficiencies of the fan, LPC and centrifugal compressor.
    eta_phpc_init, eta_phpt, eta_plpt : float
        Polytropic efficiencies of the axial HPC (before small-core losses), HPT and LPT.
    in_dP : float
        Total pressure ratio applied to the flow leaving the fan, before the bypass and core split.
    com_dPqP : float
        Combustor fractional total pressure loss.
    coolIG_frac_W : float
        Optional integrated generator: fraction of the core flow bled through the LPC to cool a
        generator housed in the LPC hub (0, the default, means no generator). See the README.
    W_max : float
        Upper end of the total mass flow sweep [kg/s]. Raise it for very large engines.

    Returns
    -------
    dict
        Converged thermodynamic state, pressure ratios, shaft speeds, cooling flows and
        component performance, to initialize the pyCycle model.
    """
    ram = 1 + (GAMMA - 1) / 2 * M**2
    Tt0 = Tamb * ram
    Pt0 = Pamb * ram ** (GAMMA / (GAMMA - 1))
    v0 = M * np.sqrt(GAMMA * R * Tamb)

    pi_hpc = split_pressure_ratio(opr, pi_fan, pi_lpc, pi_centrif)

    def compute_cycle(W_total):
        W_fan = W_total * BPR / (BPR + 1)
        W_core = W_total / (BPR + 1)

        # Fan, then the duct to the bypass and core streams
        Tt2, eta_fanroot = compressor_exit(Tt0, pi_fan, eta_pfan, GAMMA)
        Pt2 = in_dP * (Pt0 * pi_fan)
        Wdot_fan = W_total * CP * (Tt2 - Tt0)

        core = compress_core(
            W_core,
            Tt2,
            Pt2,
            pi_lpc,
            pi_hpc,
            pi_centrif,
            eta_plpc,
            eta_phpc_init,
            eta_pcentrif,
            coolIG_frac_W,
            lpc_duct_loss=DUCT_LOSSES['duct6'],
            tip_clearance_passes=1,
        )

        Wdot_hpt = core['Wdot_hpc'] + core['Wdot_centrif'] + pwr_offtk_hpt * HP_TO_W
        hot = burner_and_hpt(
            core['W'],
            core['Tt3_centrif'],
            core['Pt3_centrif'],
            Wdot_hpt,
            TIT,
            T_metal,
            com_dPqP,
            eta_phpt,
        )

        lpt = low_pressure_turbine(
            hot['W_45'],
            hot['Pt45'],
            hot['Tt45_mixed'],
            core['Wdot_lpc'] + Wdot_fan,
            Pt2,
            eta_plpt,
            inlet_duct_loss=DUCT_LOSSES['duct11'],
            exit_duct_loss=DUCT_LOSSES['duct13'],
        )

        thrust = _nozzle_thrust(W_fan, Tt2, Pt2, Pamb, v0) + _nozzle_thrust(
            lpt['W_lpt'], lpt['Tt45_lpt'], lpt['Pt45_lpt'], Pamb, v0
        )

        # Fan speed from the tip Mach number, with the inlet sized at the fan face Mach number
        A0 = W_total * np.sqrt(CP * Tt0) / (corrected_flow_function(FAN_FACE_MACH, GAMMA) * Pt0)
        r_tip = (A0 / (np.pi * (1.0 - FAN_HUB_TIP**2.0))) ** 0.5
        N_fan = (
            (60 / (2 * np.pi))
            * (
                (FAN_TIP_MACH**2 - FAN_FACE_MACH**2)
                * GAMMA
                * R
                * Tt0
                / (1 + 0.5 * (GAMMA - 1) * FAN_FACE_MACH**2)
            )
            ** 0.5
            / r_tip
        )

        return {
            **core,
            **hot,
            **lpt,
            'W_total': W_total,
            'W_fan': W_fan,
            'W_core': W_core,
            'W_core_lbs': W_core * KG_TO_LBM,
            'BPR': BPR,
            'opr': opr,
            'pi_lpc': pi_lpc,
            'pi_hpc': pi_hpc,
            'pi_centrif': pi_centrif,
            'eta_pcentrif': eta_pcentrif,
            'eta_fanroot': eta_fanroot,
            'Wdot_fan': Wdot_fan,
            'Pt0': Pt0,
            'Tt0': Tt0,
            'Pt2': Pt2,
            'Tt2': Tt2,
            'Thrust_lbf': thrust / LBF_TO_N,
            'LP_Nmech_max': lpt['N_lpt_limit'],
            'N_fan': N_fan,
        }

    W_array = np.linspace(10, W_max, int(W_max))
    return solve_mass_flow(compute_cycle, target_lbf, W_array, output='Thrust_lbf', flow='W_total')
