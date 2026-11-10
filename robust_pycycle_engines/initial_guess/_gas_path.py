"""
Gas-path calculations shared by the turbofan and turboshaft initial guesses.

Stations follow the pyCycle models: 2 compressor face, 24 LPC exit, 3 HPC exit, 4 combustor exit,
45 HPT exit. Every function is complex-step safe, as the mass flow solve differentiates through
them.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import numpy as np
from scipy.optimize import newton

from robust_pycycle_engines.initial_guess._solver import abs_cs, compressor_exit, cs_clip, cs_max
from robust_pycycle_engines.initial_guess.balances import KG_TO_LBM

# Gas properties
GAMMA = 1.4  # compressor-side ratio of specific heats
GAMMA_T = 1.3  # turbine-side ratio of specific heats
R = 287.0  # [J/kg-K] gas constant of air
CP = R * GAMMA / (GAMMA - 1)  # [J/kg-K] compressor-side specific heat
CP_T = R * GAMMA_T / (GAMMA_T - 1)  # [J/kg-K] turbine-side specific heat
LHV = 43e6  # [J/kg] fuel lower heating value (Jet-A equivalent)

# Standard-day reference conditions and unit conversions
T_SLS = 288.15  # [K]
P_SLS = 101325.0  # [Pa]
HP_TO_W = 745.7

# Fraction of the LPC work done on the integrated generator cooling flow before it is bled
COOL_IG_FRAC_WORK = 0.5

# Small-core tip-clearance penalty on the axial HPC
HPC_EXIT_HUB_TIP = 0.93
HPC_EXIT_MACH = 0.25
TIP_CLEARANCE_BLADE_HEIGHT = 13.0  # [mm] blade height below which the penalty applies
TIP_CLEARANCE_COEFF = 0.12
TIP_CLEARANCE_EXPONENT = -0.6

# Burner
FAR_EFFICIENCY_FACTOR = 0.9  # applied to the ideal fuel-air ratio

# Turbine cooling correlation
COOLING_STREAMS = ('NGV', 'Rotor1', 'Stator2', 'Rotor2')
COOLING_COEFF = 0.022 * (4 / 3)  # the 4/3 factor accounts for endwall cooling
COOLING_EXPONENT = 1.25
FILM_COOLING_FACTOR = 0.8
ODTF_NGV = 0.3  # over-design temperature factor of the NGV
ODTF_STAGE = 0.13  # over-design temperature factor of the rotors and second stator
T_SAFETY = 83.3333  # [K] (150 degR) margin on the gas temperature seen by the blades
ROTOR_RELATIVE_TEMPERATURE = 0.92  # relative to absolute gas total temperature at a rotor
COOLING_TOL = 1e-3  # [kg/s] convergence of the cooling flow iterations
COOLING_MAX_ITER = 10
TWO_STAGE_HPT_DELTA_T = 700 * (5 / 9)  # [K] HPT temperature drop above which it has two stages

# Turbine exit Mach numbers and the AN**2 shaft speed limit
HPT_EXIT_MACH = 0.3
LPT_EXIT_MACH = 0.4
SHROUDLESS_STRESS_FACTOR = (3 / 2) * 0.25  # working stress of a shroudless blade, per UTS
NICKEL_DENSITY = 8500.0  # [kg/m^3]
NICKEL_UTS = 1.2e9  # [Pa]
STEEL_DENSITY = 7900.0  # [kg/m^3]
STEEL_UTS = 1.05e9  # [Pa]
BURST_MARGIN = 1.4  # burst speed over design speed


def corrected_flow_function(M, gamma):
    """
    Compute the flow function relating mass flow, area and total conditions to Mach number.

    Parameters
    ----------
    M : float
        Mach number.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        Flow function, W * sqrt(cp * Tt) / (A * Pt).
    """
    return (
        (gamma / np.sqrt(gamma - 1))
        * M
        * (1 + (gamma - 1) / 2 * M**2) ** (-(gamma + 1) / (2 * (gamma - 1)))
    )


def shaft_speed(area, density, uts):
    """
    Compute the shaft speed at the AN**2 limit of a shroudless turbine.

    Parameters
    ----------
    area : float
        Turbine exit flow area [m^2].
    density : float
        Blade material density [kg/m^3].
    uts : float
        Blade material ultimate tensile strength [Pa].

    Returns
    -------
    float
        Design shaft speed [rpm], with the burst margin applied.
    """
    stress = SHROUDLESS_STRESS_FACTOR * uts
    return np.sqrt(stress * 3600 / (2 * np.pi * density * area)) / BURST_MARGIN


def turbine_efficiency(pi, eta_p):
    """Convert a turbine's polytropic efficiency to isentropic at pressure ratio `pi`."""
    exponent = (GAMMA_T - 1) / GAMMA_T
    return (1 - pi ** (-exponent)) / (1 - pi ** (-exponent / eta_p))


def _expansion_exit_pressure(Pt_in, temperature_ratio, eta_p, guess, tol):
    """Solve for the exit total pressure of a turbine with the given exit-to-inlet Tt ratio."""

    def residual(Pt_out):
        eta = turbine_efficiency(Pt_in / Pt_out, eta_p)
        return Pt_out - Pt_in * (1 - (1 - temperature_ratio) / eta) ** (GAMMA_T / (GAMMA_T - 1))

    return newton(residual, guess, tol=tol, maxiter=50)


def _cooling_fraction(Tt_gas, Tt_cool, T_metal, odtf, clip=False):
    """Compute a cooling flow fraction from the gas, coolant and metal temperatures."""
    phi = (Tt_gas - T_metal) / (Tt_gas - Tt_cool)
    phi_cool = (phi + odtf) / (1 + odtf)
    if clip:
        phi_cool = cs_clip(phi_cool, 1e-6, 1 - 1e-6)
    return COOLING_COEFF * FILM_COOLING_FACTOR * (phi_cool / (1 - phi_cool)) ** COOLING_EXPONENT


def compress_core(
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
    lpc_duct_loss=0.0,
    tip_clearance_passes=1,
):
    """
    Compute the LPC, axial HPC and centrifugal compressor of the core.

    Parameters
    ----------
    W_core : float
        Core mass flow into the LPC [kg/s].
    Tt2, Pt2 : float
        Total temperature [K] and pressure [Pa] into the LPC.
    pi_lpc, pi_hpc, pi_centrif : float
        Pressure ratios; a stage with a pressure ratio of 1 is absent.
    eta_plpc, eta_phpc_init, eta_pcentrif : float
        Polytropic efficiencies, the axial HPC's before the tip-clearance penalty.
    coolIG_frac_W : float
        Fraction of the core flow bled through the LPC to cool an integrated generator.
    lpc_duct_loss : float
        Fractional total pressure loss of the duct between the LPC and HPC.
    tip_clearance_passes : int
        Maximum number of updates of the tip-clearance penalty.

    Returns
    -------
    dict
        Core flow `W` net of the generator cooling, and the compressor states and powers.
    """
    Tt24, eta_lpc = compressor_exit(Tt2, pi_lpc, eta_plpc, GAMMA)
    Pt24 = Pt2 * pi_lpc

    # The generator cooling flow leaves the LPC partway through its work
    W_IG_cool = W_core * coolIG_frac_W
    P_coolIG = W_IG_cool * CP * (Tt24 - Tt2) * COOL_IG_FRAC_WORK
    W = W_core * (1 - coolIG_frac_W)
    P_prim_lpc = W * CP * (Tt24 - Tt2)
    Pt24 = Pt24 * (1 - lpc_duct_loss)

    eta_phpc = eta_phpc_init
    h_blade_old = 0.0
    for _ in range(tip_clearance_passes):
        Tt3, eta_hpc = compressor_exit(Tt24, pi_hpc, eta_phpc, GAMMA)
        Pt3 = Pt24 * pi_hpc

        A3 = W * np.sqrt(CP * Tt3) / (corrected_flow_function(HPC_EXIT_MACH, GAMMA) * Pt3)
        r_tip = np.sqrt(A3 / (np.pi * (1 - HPC_EXIT_HUB_TIP**2))) * 1000  # [mm]
        h_blade = r_tip * (1 - HPC_EXIT_HUB_TIP)  # [mm]

        delta_eta = 0.0
        if np.real(h_blade) < TIP_CLEARANCE_BLADE_HEIGHT:
            delta_eta = TIP_CLEARANCE_COEFF * (
                1.0 - (h_blade / TIP_CLEARANCE_BLADE_HEIGHT) ** TIP_CLEARANCE_EXPONENT
            )
        eta_phpc = eta_phpc_init + delta_eta
        if np.real(abs_cs(h_blade - h_blade_old)) < 0.01:
            break
        h_blade_old = h_blade

    Tt3_centrif, eta_centrif = compressor_exit(Tt3, pi_centrif, eta_pcentrif, GAMMA)

    return {
        'W': W,
        'W_lbs': W * KG_TO_LBM,
        'W_IG_cool': W_IG_cool,
        'P_coolIG': P_coolIG,
        'P_prim_lpc': P_prim_lpc,
        'Wdot_lpc': P_prim_lpc + P_coolIG,
        'Wdot_hpc': W * CP * (Tt3 - Tt24),
        'Wdot_centrif': W * CP * (Tt3_centrif - Tt3),
        'Pt24': Pt24,
        'Tt24': Tt24,
        'Pt3': Pt3,
        'Tt3': Tt3,
        'Pt3_centrif': Pt3 * pi_centrif,
        'Tt3_centrif': Tt3_centrif,
        'eta_lpc': eta_lpc,
        'eta_hpc': eta_hpc,
        'eta_phpc': eta_phpc,
        'eta_centrif': eta_centrif,
        'mass_flow_corrected_lbs': W * (P_SLS / Pt24) * np.sqrt(Tt24 / T_SLS) * KG_TO_LBM,
        'mass_flow_corrected_lbs_centrif': W * (P_SLS / Pt3) * np.sqrt(Tt3 / T_SLS) * KG_TO_LBM,
    }


def burner_and_hpt(W, Tt3, Pt3, Wdot_hpt, Tt4, T_metal, com_dPqP, eta_phpt, cooled=True):
    """
    Size the HPT cooling flows, then close the burner and HPT on the HP shaft power.

    Parameters
    ----------
    W : float
        Core mass flow into the cooling bleeds and burner [kg/s].
    Tt3, Pt3 : float
        Compressor exit total temperature [K] and pressure [Pa], also those of the coolant.
    Wdot_hpt : float
        HP shaft power the HPT must supply [W].
    Tt4 : float
        Turbine inlet total temperature [K].
    T_metal : float
        Maximum allowable metal temperature [K].
    com_dPqP : float
        Combustor fractional total pressure loss.
    eta_phpt : float
        HPT polytropic efficiency.
    cooled : bool
        If False, the turbine is uncooled.

    Returns
    -------
    dict
        Burner and HPT states, cooling flows and HP shaft speed.
    """
    Pt4 = Pt3 * (1.0 - com_dPqP)
    FAR = FAR_EFFICIENCY_FACTOR * (CP_T * Tt4 - CP * Tt3) / (LHV - CP * Tt3)

    def burn(W_3):
        fuel = (CP_T * Tt4 - CP * Tt3) * W_3 / LHV
        return fuel, W_3 + fuel

    def mix(W_gas, Tt_gas, W_cool):
        return (W_cool * CP * Tt3 + W_gas * CP_T * Tt_gas) / ((W_gas + W_cool) * CP_T)

    # A bleed's frac_P is where it enters the turbine: 1 at Pt4 (does no work), 0 at Pt45
    frac_W = dict.fromkeys(COOLING_STREAMS, 0.0)
    frac_P = dict.fromkeys(COOLING_STREAMS, 0.0)
    W_cool = dict.fromkeys(COOLING_STREAMS, 0.0)
    frac_P['NGV'] = 1.0

    frac_W['NGV'] = _cooling_fraction(Tt4 + T_SAFETY, Tt3, T_metal, ODTF_NGV, clip=True)
    if cooled:
        W_cool['NGV'] = cs_max(frac_W['NGV'] * W, 0.0)

    if W_cool['NGV'] == 0.0:
        frac_W['NGV'] = 0.0
        fuel, W_hpt = burn(W)
    else:
        W_rotor_old = 0.0
        for _ in range(COOLING_MAX_ITER):
            fuel, W_hpt = burn(W - W_cool['NGV'] - W_rotor_old)
            Tt41 = mix(W_hpt, Tt4, W_cool['NGV'])
            frac_W['Rotor1'] = _cooling_fraction(
                ROTOR_RELATIVE_TEMPERATURE * Tt41 + T_SAFETY, Tt3, T_metal, ODTF_STAGE
            )
            W_cool['Rotor1'] = cs_max(frac_W['Rotor1'] * W, 0.0)
            if np.real(abs_cs(W_rotor_old - W_cool['Rotor1'])) < COOLING_TOL:
                break
            W_rotor_old = W_cool['Rotor1']

            delta_Tt_hpt = Wdot_hpt / ((W_hpt + W_cool['NGV']) * CP_T)
            if np.real(delta_Tt_hpt) > TWO_STAGE_HPT_DELTA_T:
                frac_P.update(Rotor1=0.5, Stator2=0.5)
                fuel, W_hpt = _size_two_stage_cooling(
                    W, Tt3, Tt4, Pt4, Wdot_hpt, T_metal, eta_phpt, burn, W_cool, frac_W
                )

    # Solve for Pt45, starting from the expansion without the work of the cooling flows
    Tt45_nobleed = Tt4 - Wdot_hpt / (W_hpt * CP_T)
    Pt45_guess = _expansion_exit_pressure(Pt4, Tt45_nobleed / Tt4, eta_phpt, Pt4 * 0.5, 1e-5)
    exponent = (GAMMA - 1.0) / GAMMA

    def coolant_work(Pt45, stream):
        # Specific work of a cooling stream, expanding from where it enters to Pt45, per unit eta
        Pt_cool = Pt45 + frac_P[stream] * (Pt4 - Pt45)
        return CP * (Tt3 - Tt3 * (Pt45 / Pt_cool) ** exponent)

    def power_residual(Pt45):
        eta_t = turbine_efficiency(Pt4 / Pt45, eta_phpt)
        Tt45_s = Tt4 * (Pt45 / Pt4) ** ((GAMMA_T - 1.0) / GAMMA_T)
        P_primary = eta_t * W_hpt * CP_T * (Tt4 - Tt45_s)
        P_bleeds = eta_t * sum(W_cool[s] * coolant_work(Pt45, s) for s in COOLING_STREAMS)
        return P_primary + P_bleeds - Wdot_hpt

    Pt45 = newton(power_residual, Pt45_guess, tol=1e-4, maxiter=50)

    pi_hpt = Pt4 / Pt45
    eta_hpt = turbine_efficiency(pi_hpt, eta_phpt)
    Tt45_s = Tt4 * (Pt45 / Pt4) ** ((GAMMA_T - 1.0) / GAMMA_T)
    Tt45 = Tt4 - eta_hpt * (Tt4 - Tt45_s)

    w_hot = eta_hpt * CP_T * (Tt4 - Tt45_s)
    w_cool = {s: eta_hpt * coolant_work(Pt45, s) for s in COOLING_STREAMS}
    W_cool_total = sum(W_cool.values())
    W_45 = W_hpt + W_cool_total
    w_hpt = (W_hpt * w_hot + sum(W_cool[s] * w_cool[s] for s in COOLING_STREAMS)) / W_45

    # Mixed total temperature into the next turbine
    Tt45_mixed = (W_cool_total * CP * Tt3 + W_hpt * CP_T * Tt45) / (W_45 * CP_T)

    A45 = (
        W_45 * np.sqrt(CP_T * Tt45_mixed) / (corrected_flow_function(HPT_EXIT_MACH, GAMMA_T) * Pt45)
    )

    state = {
        'FAR': FAR,
        'mass_flow_fuel': fuel,
        'W_hpt': W_hpt,
        'W_45': W_45,
        'Wdot_hpt': Wdot_hpt,
        'w_hpt': w_hpt,
        'Pt4': Pt4,
        'Tt4': Tt4,
        'Pt45': Pt45,
        'Tt45': Tt45,
        'Tt45_mixed': Tt45_mixed,
        'pi_hpt': pi_hpt,
        'eta_hpt': eta_hpt,
        'cp_t': CP_T,
        'HP_Nmech_max': shaft_speed(A45, NICKEL_DENSITY, NICKEL_UTS),
        'Pt_cool_NGV': Pt45 + frac_P['NGV'] * (Pt4 - Pt45),
        'Pt_cool_Rotor1': Pt45 + frac_P['Rotor1'] * (Pt4 - Pt45),
    }
    for stream in COOLING_STREAMS:
        state[f'W_cool_{stream}'] = W_cool[stream]
        state[f'w_cool_{stream}'] = w_cool[stream]
        state[f'Wdot_cool_{stream}'] = W_cool[stream] * w_cool[stream]
        state[f'cool{stream}_frac_W'] = frac_W[stream]
        state[f'cool{stream}_frac_P'] = frac_P[stream]
    return state


def _size_two_stage_cooling(W, Tt3, Tt4, Pt4, Wdot_hpt, T_metal, eta_phpt, burn, W_cool, frac_W):
    """
    Size the rotor and second stator cooling flows of a two-stage HPT, updating them in place.

    Returns
    -------
    tuple of float
        Fuel flow and HPT inlet mass flow [kg/s].
    """

    def mix(W_gas, Tt_gas, streams):
        # Mixed total temperature of the gas and the cooling flows injected so far
        W_bled = sum(W_cool[s] for s in streams)
        heat = sum(W_cool[s] * CP * Tt3 for s in streams)
        return (heat + W_gas * CP_T * Tt_gas) / ((W_gas + W_bled) * CP_T)

    W_cool.update(Rotor1=0.0, Rotor2=0.0)
    W_stator_old = 0.0
    for _ in range(COOLING_MAX_ITER):
        W_3 = W - W_cool['NGV'] - W_cool['Rotor1'] - W_stator_old - W_cool['Rotor2']
        fuel, W_hpt = burn(W_3)

        # Stage work is split evenly in pressure ratio
        Tt45_nobleed = Tt4 - Wdot_hpt / ((W_hpt + W_cool['NGV']) * CP_T)
        Pt45 = _expansion_exit_pressure(Pt4, Tt45_nobleed / Tt4, eta_phpt, Pt4 * 0.5, 1e-5)
        pi_hpt = Pt4 / Pt45
        eta_t = turbine_efficiency(pi_hpt, eta_phpt)
        Tt42_s = Tt4 * (1 / pi_hpt**0.5) ** ((GAMMA_T - 1.0) / GAMMA_T)
        Tt42_expanded = Tt4 - eta_t * (Tt4 - Tt42_s)

        Tt41 = mix(W_hpt, Tt4, ('NGV',))
        frac_W['Rotor1'] = _cooling_fraction(
            ROTOR_RELATIVE_TEMPERATURE * Tt41 + T_SAFETY, Tt3, T_metal, ODTF_STAGE
        )
        W_cool['Rotor1'] = cs_max(frac_W['Rotor1'] * W, 0.0)

        Tt42 = mix(W_hpt, Tt42_expanded, ('NGV', 'Rotor1'))
        frac_W['Stator2'] = _cooling_fraction(Tt42 + T_SAFETY, Tt3, T_metal, ODTF_STAGE)
        W_cool['Stator2'] = cs_max(frac_W['Stator2'] * W, 0.0)

        Tt43 = mix(W_hpt, Tt42_expanded, ('NGV', 'Rotor1', 'Stator2'))
        frac_W['Rotor2'] = _cooling_fraction(
            ROTOR_RELATIVE_TEMPERATURE * Tt43 + T_SAFETY, Tt3, T_metal, ODTF_STAGE
        )
        W_cool['Rotor2'] = cs_max(frac_W['Rotor2'] * W, 0.0)

        if np.real(abs_cs(W_stator_old - W_cool['Stator2'])) < COOLING_TOL:
            break
        W_stator_old = W_cool['Stator2']
    return fuel, W_hpt


def low_pressure_turbine(
    W_lpt, Pt45, Tt45_mixed, Wdot_lpt, Pt2, eta_plpt, inlet_duct_loss=0.0, exit_duct_loss=0.0
):
    """
    Close the LPT on its shaft power.

    Parameters
    ----------
    W_lpt : float
        LPT mass flow, with the turbine cooling flows mixed in [kg/s].
    Pt45 : float
        HPT exit total pressure [Pa].
    Tt45_mixed : float
        Mixed total temperature into the LPT [K].
    Wdot_lpt : float
        Shaft power the LPT must supply [W].
    Pt2 : float
        Compressor face total pressure [Pa], which sets the starting guess.
    eta_plpt : float
        LPT polytropic efficiency.
    inlet_duct_loss, exit_duct_loss : float
        Fractional total pressure losses of the ducts before and after the LPT.

    Returns
    -------
    dict
        `Pt45` after the inlet duct, the LPT exit state after the exit duct, its pressure ratio
        and efficiency, and the speed limit of a steel turbine at its exit.
    """
    Tt45_lpt = Tt45_mixed - Wdot_lpt / (W_lpt * CP_T)
    Pt45 = Pt45 * (1 - inlet_duct_loss)
    if np.real(Wdot_lpt) == 0.0:  # no LPC to drive: the LPT is absent
        Pt45_lpt, pi_lpt, eta_lpt = Pt45, 1.0, 1.0
    else:
        guess = Pt45 - 0.5 * (Pt45 - Pt2)
        Pt45_lpt = _expansion_exit_pressure(Pt45, Tt45_lpt / Tt45_mixed, eta_plpt, guess, 1e-6)
        pi_lpt = Pt45 / Pt45_lpt
        eta_lpt = turbine_efficiency(pi_lpt, eta_plpt)
    Pt45_lpt = Pt45_lpt * (1 - exit_duct_loss)

    A45_lpt = (
        W_lpt * np.sqrt(CP * Tt45_lpt) / (corrected_flow_function(LPT_EXIT_MACH, GAMMA) * Pt45_lpt)
    )
    return {
        'W_lpt': W_lpt,
        'Wdot_lpt': Wdot_lpt,
        'Pt45': Pt45,
        'Pt45_lpt': Pt45_lpt,
        'Tt45_lpt': Tt45_lpt,
        'pi_lpt': pi_lpt,
        'eta_lpt': eta_lpt,
        'N_lpt_limit': shaft_speed(A45_lpt, STEEL_DENSITY, STEEL_UTS),
    }
