"""
On-design initial guess for a multi-spool turboshaft with a free power turbine.

Implements the physics-based on-design initialization of Tahiliani et al.,
"Robust Approach to Engine Cycle Modeling to Enable Multidisciplinary Optimization of
Advanced Aircraft", AIAA AVIATION 2026 (https://doi.org/10.2514/6.2026-4499), Sec. IV.

For a prescribed mass flow, a Brayton cycle that mirrors the pyCycle model (turbine cooling,
shaft speeds, small-core losses) closes in closed form. The design problem therefore reduces
to a one-dimensional root find in mass flow: a coarse sweep brackets the target shaft power, and
the bracket midpoint seeds a Newton solve. The returned state seeds pyCycle's balances.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import numpy as np

from robust_pycycle_engines.initial_guess._gas_path import (
    CP,
    CP_T,
    GAMMA,
    GAMMA_T,
    HP_TO_W,
    LPT_EXIT_MACH,
    STEEL_DENSITY,
    STEEL_UTS,
    burner_and_hpt,
    compress_core,
    corrected_flow_function,
    low_pressure_turbine,
    shaft_speed,
    turbine_efficiency,
)
from robust_pycycle_engines.initial_guess._solver import solve_mass_flow, split_pressure_ratio
from robust_pycycle_engines.initial_guess.balances import KG_TO_LBM


def initial_guess(
    opr,
    M,
    Tamb,
    Pamb,
    T_metal,
    Nozzle_PR,
    TIT,
    target_hp,
    pwr_offtk_hpt,
    pi_lpc,
    pi_centrif=1.0,
    eta_plpc=0.9,
    eta_pcentrif=0.85,
    eta_phpc_init=0.9,
    eta_phpt=0.9,
    eta_plpt=0.9,
    eta_ppt=0.9,
    in_dP=1.0,
    com_dPqP=0.05,
    disable_cooling=False,
    coolIG_frac_W=0.0,
    W_max=100.0,
):
    """
    Compute an initial thermodynamic cycle state for a given altitude and engine configuration.

    Parameters
    ----------
    opr : float
        Overall pressure ratio. The axial HPC supplies what the LPC and centrifugal stage do not:
        pi_hpc = opr / (pi_lpc * pi_centrif).
    M : float
        Freestream Mach number.
    Tamb : float
        Ambient static temperature [K].
    Pamb : float
        Ambient static pressure [Pa].
    T_metal : float
        Maximum allowable metal temperature [K].
    Nozzle_PR : float
        Nozzle pressure ratio (>1).
    TIT : float
        Turbine inlet temperature [K].
    target_hp : float
        Target shaft power output [hp].
    pwr_offtk_hpt : float
        Power off-take from the HP shaft [hp].
    pi_lpc : float
        Low-pressure compressor pressure ratio.
    pi_centrif : float
        Pressure ratio of a centrifugal compressor after the axial HPC; 1 means there is none.
    eta_plpc, eta_pcentrif : float
        Polytropic efficiencies of the LPC and centrifugal compressor.
    eta_phpc_init, eta_phpt, eta_plpt, eta_ppt : float
        Polytropic efficiencies of the axial HPC (before small-core losses), HPT, LPT and power
        turbine.
    in_dP : float
        Inlet total pressure ratio.
    com_dPqP : float
        Combustor fractional total pressure loss.
    disable_cooling : bool
        If True, assume an uncooled turbine (small engines such as turboprops).
    coolIG_frac_W : float
        Optional integrated generator: fraction of the core flow bled through the LPC to cool a
        generator housed in the LPC hub (0, the default, means no generator). See the README.
    W_max : float
        Upper end of the core mass flow sweep [kg/s]. Raise it for very large engines.

    Returns
    -------
    dict
        Converged thermodynamic state, pressure ratios, shaft speeds, cooling flows and
        component performance, to initialize the pyCycle model.
    """
    ram = 1 + (GAMMA - 1) / 2 * M**2
    Tt0 = Tamb * ram
    Pt0 = Pamb * ram ** (GAMMA / (GAMMA - 1))
    Pt2 = in_dP * Pt0
    Tt2 = Tt0

    pi_hpc = split_pressure_ratio(opr, pi_lpc, pi_centrif)

    def compute_cycle(W_core):
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
            tip_clearance_passes=2,
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
            cooled=not disable_cooling,
        )

        lpt = low_pressure_turbine(
            hot['W_45'], hot['Pt45'], hot['Tt45_mixed'], core['Wdot_lpc'], Pt2, eta_plpt
        )
        W_lpt, Pt45_lpt, Tt45_lpt = lpt['W_lpt'], lpt['Pt45_lpt'], lpt['Tt45_lpt']

        # Free power turbine expands to the nozzle pressure
        Pt5 = Pamb * Nozzle_PR
        pi_pt = Pt45_lpt / Pt5
        eta_pt = turbine_efficiency(pi_pt, eta_ppt)
        Tt5 = Tt45_lpt * (1 - eta_pt * (1 - pi_pt ** (-(GAMMA_T - 1) / GAMMA_T)))
        power_out = W_lpt * CP_T * (Tt45_lpt - Tt5)

        A5 = W_lpt * np.sqrt(CP * Tt5) / (corrected_flow_function(LPT_EXIT_MACH, GAMMA) * Pt5)

        return {
            **core,
            **hot,
            **lpt,
            'W_core': W_core,
            'W_core_lbs': W_core * KG_TO_LBM,
            'opr': opr,
            'pi_lpc': pi_lpc,
            'pi_hpc': pi_hpc,
            'pi_centrif': pi_centrif,
            'eta_pcentrif': eta_pcentrif,
            'pi_pt': pi_pt,
            'eta_pt': eta_pt,
            'power_out': power_out,
            'hp_out': power_out / HP_TO_W,
            'Pt0': Pt0,
            'Tt0': Tt0,
            'Pt2': Pt2,
            'Tt2': Tt2,
            'Pt5': Pt5,
            'Tt5': Tt5,
            'IP_Nmech_max': lpt['N_lpt_limit'],
            'LP_Nmech_max': shaft_speed(A5, STEEL_DENSITY, STEEL_UTS),
        }

    W_array = np.linspace(1, W_max, 990)
    return solve_mass_flow(compute_cycle, target_hp, W_array, output='hp_out', flow='W_core')
