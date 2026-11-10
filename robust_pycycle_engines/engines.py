"""
Design points and modeling assumptions of the engines studied, keyed by engine name.

The design points were set with back-of-the-envelope cycle calculations to approximate each
engine; they are not manufacturer data and are not validated. Each entry holds the keyword
arguments of `initial_guess.turbofan.initial_guess` or `initial_guess.turboshaft.initial_guess`;
optional ones (efficiencies, losses, the integrated generator `coolIG_frac_W`) appear only where
an engine departs from the default.
"""

from robust_pycycle_engines.initial_guess import turbofan, turboshaft

TURBOFANS = {
    'PW4168': {
        'opr': 32.0,
        'M': 0.8,
        'Tamb': 218.8,
        'Pamb': 23840.0,
        'T_metal': 1300.0,
        'BPR': 5.1,
        'TIT': 1530.0,
        'target_lbf': 20500.0,
        'pwr_offtk_hpt': 450.0,
        'pi_fan': 1.75,
        'pi_lpc': 2.2,
        'eta_phpc_init': 0.903,
        'eta_phpt': 0.873,
        'eta_plpt': 0.882,
    },
    'AE3007': {
        'opr': 28.5,
        'M': 0.78,
        'Tamb': 216.65,
        'Pamb': 21662.71,
        'T_metal': 1300.0,
        'BPR': 5.3,
        'TIT': 1500.0,
        'target_lbf': 2250.0,
        'pwr_offtk_hpt': 80.0,
        'pi_fan': 1.6,
        'pi_lpc': 1.7,
        'eta_phpc_init': 0.903,
        'eta_phpt': 0.873,
        'eta_plpt': 0.882,
        'eta_plpc': 1.0,
    },
    'CF34-8E': {
        'opr': 16.7647058824 * 1.7,
        'M': 0.8,
        'Tamb': 218.8,
        'Pamb': 23840.0,
        'T_metal': 1300.0,
        'BPR': 5.0,
        'TIT': 1600.0,
        'target_lbf': 4350.0,
        'pwr_offtk_hpt': 150.0,
        'pi_fan': 1.7,
        'pi_lpc': 1.0,  # no booster
        'eta_phpc_init': 0.903,
        'eta_phpt': 0.873,
        'eta_plpt': 0.882,
        'eta_plpc': 1.0,
    },
    'GE90': {
        'opr': 43.429815,
        'M': 0.83,
        'Tamb': 218.8,
        'Pamb': 23840.0,
        'T_metal': 1300.0,
        'BPR': 9.0,
        'TIT': 1655.556,
        'target_lbf': 25000.0,
        'pwr_offtk_hpt': 400.0,
        'pi_fan': 1.65,
        'pi_lpc': 1.3,
        'eta_phpc_init': 0.903,
        'eta_phpt': 0.873,
        'eta_plpt': 0.882,
        'W_max': 3000.0,
    },
    'CFM56': {
        'opr': 30.0,
        'M': 0.8,
        'Tamb': 218.8,
        'Pamb': 23840.0,
        'T_metal': 1300.0,
        'BPR': 5.1,
        'TIT': 1587.222,
        'target_lbf': 5900.0,
        'pwr_offtk_hpt': 250.0,
        'pi_fan': 1.685,
        'pi_lpc': 1.935,
        'eta_phpc_init': 0.903,
        'eta_phpt': 0.873,
        'eta_plpt': 0.882,
    },
}

# pyCycle model settings of the turbofans with a cycle model, overriding
# `cycles.turbofan.DEFAULT_SETTINGS`. They are designed at 35,000 ft unless noted.
TURBOFAN_MODELS = {
    'AE3007': {
        'design_alt': 37000.0,
        'W_max_design': 1000.0,
        'W_max_off_design': 5000.0,
    },
    'CFM56': {
        'W_max_design': 1000.0,
        'W_max_off_design': 5000.0,
    },
    'CF34-8E': {
        'W_max_design': 1000.0,
        'W_max_off_design': 5000.0,
        'maxiter': 30,
    },
    'PW4168': {
        'W_max_design': 1000.0,
        'W_max_off_design': 5000.0,
        'maxiter': 6,
    },
    'GE90': {
        'fan_tip_mach': 1.2,
        'fan_face_mach': 0.725,
        'W_max_design': 10000.0,
        'W_max_off_design': 10000.0,
        'BPR_max': 30.0,
        'N_min': 10.0,
        'maxiter': 15,
        'bleed_pc_start': 0.875,
    },
}

# pyCycle model settings of the turboshafts with a cycle model, overriding
# `cycles.turboshaft.DEFAULT_SETTINGS`
TURBOSHAFT_MODELS = {
    'PW127': {'lpc_bleed': True},
    'reference_9100hp': {'design_alt': 33000.0},
    'TPE331-10': {
        'design_alt': 15000.0,
        'cycle_params': {'duct225.dPqP': 0.01, 'icduct.dPqP': 0.01, 'itduct.dPqP': 0.01},
        'map_points': {'hpc_centri': {'RlineMap': 2.15}},  # pyCycle's LPC map design R-line
    },
}

TURBOSHAFTS = {
    # Mach 0.8 reference design of the Mach-altitude sweep in the paper (Fig. 3)
    'reference_9100hp': {
        'opr': 7.421583,
        'M': 0.8,
        'Tamb': 222.7736,
        'Pamb': 26203.0,
        'T_metal': 1300.0,
        'Nozzle_PR': 1.05,
        'TIT': 1587.22,
        'target_hp': 9100.0,
        'pwr_offtk_hpt': 100.0,
        'pi_lpc': 1.9,
        'pi_centrif': 2.0,
        'eta_ppt': 0.887,
    },
    'PW127': {
        'opr': 2.6 * 5.65,  # all-centrifugal core: no axial HPC
        'M': 0.2,
        'Tamb': 288.15,
        'Pamb': 101325.0,
        'T_metal': 1300.0,
        'Nozzle_PR': 1.1,
        'TIT': 1400.0,
        'target_hp': 2750.0,
        'pwr_offtk_hpt': 0.0,
        'pi_lpc': 2.6,
        'pi_centrif': 5.65,
        'eta_ppt': 0.887,
        'disable_cooling': True,
        'eta_plpc': 0.85,
        'eta_phpc_init': 1.0,
        'com_dPqP': 0.0,
    },
    # All-centrifugal gas generator with a free power turbine: no LPC, LPT or IP shaft
    'TPE331-10': {
        'opr': 12.78,
        'M': 0.4,
        'Tamb': 258.4335,
        'Pamb': 57184.0,
        'T_metal': 1300.0,
        'Nozzle_PR': 1.2,
        'TIT': 1300.0,
        'target_hp': 940.0,
        'pwr_offtk_hpt': 0.0,
        'pi_lpc': 1.0,
        'pi_centrif': 12.78,
        'eta_pcentrif': 0.81,
        'eta_phpt': 0.85,
        'eta_ppt': 0.85,
        'disable_cooling': True,
    },
}


def get_engine(engine):
    """
    Describe an engine by its architecture, design point and cycle model settings.

    Parameters
    ----------
    engine : str or dict
        Key of `TURBOFANS` or `TURBOSHAFTS`, or a dictionary describing a new engine with keys

        - ``'architecture'``: ``'turbofan'`` or ``'turboshaft'``.
        - ``'design'``: design point, with the keys of a `TURBOFANS` or `TURBOSHAFTS` entry.
        - ``'model'`` (optional): cycle model settings overriding the architecture's
          ``DEFAULT_SETTINGS`` in `robust_pycycle_engines.cycles`.

    Returns
    -------
    dict
        Keys ``'architecture'``, ``'design'`` and ``'model'``. ``'model'`` is None for a named
        engine without a cycle model.
    """
    if isinstance(engine, dict):
        if engine.get('architecture') not in ('turbofan', 'turboshaft'):
            raise ValueError(
                "An engine dictionary needs 'architecture': 'turbofan' or 'turboshaft'."
            )
        return {
            'architecture': engine['architecture'],
            'design': engine['design'],
            'model': engine.get('model', {}),
        }
    if engine in TURBOFANS:
        return {
            'architecture': 'turbofan',
            'design': TURBOFANS[engine],
            'model': TURBOFAN_MODELS.get(engine),
        }
    if engine in TURBOSHAFTS:
        return {
            'architecture': 'turboshaft',
            'design': TURBOSHAFTS[engine],
            'model': TURBOSHAFT_MODELS.get(engine),
        }
    raise KeyError(f'Unknown engine {engine!r}. Choose from {sorted(TURBOFANS | TURBOSHAFTS)}.')


def initial_guess(engine):
    """
    Compute the on-design initial guess of an engine.

    Parameters
    ----------
    engine : str or dict
        Engine name or description, see `get_engine`.

    Returns
    -------
    dict
        Cycle state at the engine's design point.
    """
    spec = get_engine(engine)
    module = turbofan if spec['architecture'] == 'turbofan' else turboshaft
    return module.initial_guess(**spec['design'])
