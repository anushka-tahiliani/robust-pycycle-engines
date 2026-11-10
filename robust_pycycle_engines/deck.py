"""
Generate engine decks with the pyCycle cycle models.

Each flight condition (Mach number, altitude) gets its own problem: the design point is solved
with the off-design point, which sweeps throttle, a fraction of maximum T4, from full power down.
The first throttle starts from the initial guess; each later one starts from the last converged
throttle, and a throttle that fails is skipped. If no throttle converges, the flight condition is
retried with the mass flow guess scaled by each of `RETRY_FLOW_SCALES`.

On turbofans with an LPC, LPC bleed keeps the LPC at or above the R-line schedule of
`components.BleedMargin`, at every altitude and only as far as needed. Below `TAKEOFF_ALTITUDE`
maximum T4 is raised by `TAKEOFF_T4_FACTOR` for takeoff rating. A turboshaft with the `lpc_bleed`
setting bleeds its LPC the same way.

Run from the command line to generate a deck over the flight conditions of an included one, or
over chosen flight conditions and throttles::

    python -m robust_pycycle_engines.deck PW4168 PW4168.csv --processes 8
    python -m robust_pycycle_engines.deck PW4168 PW4168.csv --point 0.8 35000 --throttles 1 0.95 0.9

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import argparse
import contextlib
import csv
import functools
import io
import multiprocessing

import numpy as np
import openmdao.api as om

from robust_pycycle_engines import engines
from robust_pycycle_engines.cycles import turbofan, turboshaft
from robust_pycycle_engines.decks import DECKS, load_deck
from robust_pycycle_engines.initial_guess.balances import K_TO_R

THROTTLES = tuple(np.round(np.arange(1.0, 0.6249, -0.025), 3))
TAKEOFF_ALTITUDE = 10000.0  # [ft]
TAKEOFF_T4_FACTOR = 1.025
RETRY_FLOW_SCALES = (0.99, 1.01)

# Off-design point and the balances carried from one throttle to the next; a turboshaft's IP speed
# is not seeded and starts from its value at the previous throttle
_TURBOFAN_SEED = {
    'OD': ('balance.W', 'balance.FAR', 'balance.BPR', 'balance.HP_Nmech', 'balance.LP_Nmech')
}
_TURBOSHAFT_SEED = {'OD': ('balance.W', 'balance.FAR', 'balance.HP_Nmech')}


def build_problem(engine, MN, alt, lpc_bleed=True, flow_scale=1.0):
    """
    Set up the design and off-design problem of an engine at one flight condition.

    Parameters
    ----------
    engine : str or dict
        Engine name, or a dictionary describing a new engine; see `engines.get_engine`.
    MN : float
        Off-design Mach number.
    alt : float
        Off-design altitude [ft].
    lpc_bleed : bool
        Turbofans only: if True, off-design LPC bleed holds the LPC R-line schedule; else no LPC
        bleed.
    flow_scale : float
        Factor on the off-design mass flow guess.

    Returns
    -------
    openmdao.api.Problem
        Problem ready to run, with throttle `OD.PC` to set.
    """
    spec = engines.get_engine(engine)
    if spec['model'] is None:
        raise ValueError(
            f'{engine!r} has no cycle model in engines.TURBOFAN_MODELS or TURBOSHAFT_MODELS.'
        )
    module = turbofan if spec['architecture'] == 'turbofan' else turboshaft
    unknown = set(spec['model']) - set(module.DEFAULT_SETTINGS)
    if unknown:
        raise ValueError(f'Unknown model settings {sorted(unknown)}; see DEFAULT_SETTINGS.')
    design = spec['design']
    cycle = engines.initial_guess(spec)
    prob = om.Problem(reports=False)
    if spec['architecture'] == 'turbofan':
        settings = {**turbofan.DEFAULT_SETTINGS, **spec['model']}
        prob.model = turbofan.MPTurbofan(
            engine=design,
            cycle=cycle,
            settings=settings,
            lpc_bleed=lpc_bleed,
            flow_scale=flow_scale,
            flight_condition=(MN, alt),
        )
        prob.setup()
        T4_design = design['TIT'] * K_TO_R
        prob.set_val('DESIGN.Fn_DES', design['target_lbf'], units='lbf')
        T4_factor = TAKEOFF_T4_FACTOR if alt < TAKEOFF_ALTITUDE else 1.0
        prob.set_val('OD.T4_max', T4_design * T4_factor, units='degR')
    else:
        settings = {**turboshaft.DEFAULT_SETTINGS, **spec['model']}
        prob.model = turboshaft.MPTurboshaft(
            engine=design,
            cycle=cycle,
            settings=settings,
            flow_scale=flow_scale,
            flight_condition=(MN, alt),
        )
        prob.setup()
        T4_design = design['TIT'] * K_TO_R
        prob.set_val('DESIGN.balance.rhs:W', design['Nozzle_PR'])
        prob.set_val('DESIGN.nozzle.PR', 1.5)
        prob.set_val('DESIGN.pt.eff', cycle['eta_pt'])
        prob.set_val('OD.T4_max', T4_design, units='degR')
        prob.set_val('OD.nozzle.PR', design['Nozzle_PR'])

    prob.set_val('DESIGN.fc.alt', settings['design_alt'], units='ft')
    prob.set_val('DESIGN.fc.MN', design['M'])
    prob.set_val('DESIGN.balance.rhs:FAR', T4_design, units='degR')
    prob.set_val('DESIGN.hpt.eff', cycle['eta_hpt'])
    if prob.model.DESIGN._get_subsystem('lpt') is not None:
        prob.set_val('DESIGN.lpt.eff', cycle['eta_lpt'])
    prob.set_val('OD.fc.MN', MN)
    prob.set_val('OD.fc.alt', alt, units='ft')
    prob.set_solver_print(level=-1)
    return prob


def sweep_throttle(prob, throttles=THROTTLES):
    """
    Run the off-design points over throttle, warm-starting from the last converged throttle.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Problem from `build_problem`.
    throttles : sequence of float
        Throttles (fractions of maximum T4) to run, in order. The default, `THROTTLES`, is 1 down
        to 0.625 in steps of 0.025.

    Returns
    -------
    list of dict
        Deck row (see `deck_row`) of each converged throttle.
    """
    is_turbofan = isinstance(prob.model, turbofan.MPTurbofan)
    seed_vars = _TURBOFAN_SEED if is_turbofan else _TURBOSHAFT_SEED
    prob.final_setup()  # outside the solve loop, so a model that cannot be built raises
    initial_state = prob.model._outputs.asarray().copy()
    rows = []
    seed = None
    for PC in throttles:
        prob.set_val('OD.PC', PC)
        if seed is None:
            for pt in seed_vars:
                prob.model._get_subsystem(pt).reset_guess()
        else:
            for name, value in seed.items():
                prob.set_val(name, value)
        try:
            prob.run_model()
        except (om.AnalysisError, RuntimeError):  # a failed solve; other errors are real bugs
            if seed is None:  # with no converged point to return to, start the next one clean
                prob.model._outputs.set_val(initial_state)
            continue
        seed = {
            f'{pt}.{var}': prob.get_val(f'{pt}.{var}').copy()
            for pt, variables in seed_vars.items()
            for var in variables
        }
        rows.append(deck_row(prob, PC) if is_turbofan else turboshaft_deck_row(prob, PC))
    return rows


def run_flight_condition(
    engine,
    MN,
    alt,
    throttles=THROTTLES,
    lpc_bleed=True,
    retry_flow_scales=RETRY_FLOW_SCALES,
):
    """
    Compute the deck rows of an engine at one flight condition.

    Parameters
    ----------
    engine : str or dict
        Engine name, or a dictionary describing a new engine; see `engines.get_engine`.
    MN : float
        Mach number.
    alt : float
        Altitude [ft].
    throttles : sequence of float
        Throttles (fractions of maximum T4) to run, in order. The default, `THROTTLES`, is 1 down
        to 0.625 in steps of 0.025.
    lpc_bleed : bool
        Turbofans only: if True, LPC bleed holds the LPC R-line schedule where needed; else no LPC
        bleed.
    retry_flow_scales : sequence of float
        Factors on the mass flow guess to retry with when no throttle converges.

    Returns
    -------
    list of dict
        Deck row of each converged throttle; empty if none converged.
    """
    for flow_scale in (1.0, *retry_flow_scales):
        prob = build_problem(engine, MN, alt, lpc_bleed, flow_scale)
        rows = sweep_throttle(prob, throttles)
        if rows:
            return rows
    return []


def deck_row(prob, PC):
    """
    Collect the deck columns of a converged turbofan off-design point.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Converged problem from `build_problem`.
    PC : float
        Throttle of the point.

    Returns
    -------
    dict
        Column values keyed by the deck headers.
    """

    def get(var, units):
        return float(prob.get_val(f'OD.{var}', units=units)[0])

    def present(element):
        return prob.model.OD._get_subsystem(element) is not None

    last_compressor = 'hpc_centri' if present('hpc_centri') else 'hpc_axi'
    BPR = get('splitter.BPR', None)
    ram_drag = get('inlet.F_ram', 'lbf')
    Fg_core = get('core_nozz.Fg', 'lbf')
    Fg_fan = get('byp_nozz.Fg', 'lbf')
    Fn = get('perf.Fn', 'lbf')
    fuel_flow = get('perf.Wfuel', 'lbm/s') * 3600.0
    fan_power = -get('fan.power', 'hp')

    row = {
        'Mach Number (input)': get('fc.MN', None),
        'Altitude (ft, input)': get('fc.alt', 'ft'),
        'Throttle (input)': PC,
        'Fan Shaft Power (hp, fan output)': fan_power,
        # On a turbofan this column holds the fan shaft power
        'Off-Design Core Shaft Power (hp, core output)': fan_power,
        'Net Core Thrust (lbf, core output)': Fg_core - ram_drag / (BPR + 1.0),
        'Net Fan Thrust (lbf, fan output)': Fg_fan - ram_drag * BPR / (BPR + 1.0),
        'Total Net Thrust (lbf, output)': Fn,
        'Fuel Flow (lb/h, core output)': fuel_flow,
        'Gross Core Exhaust (lbf, output)': Fg_core,
        'Gross Fan Thrust (lbf, output)': Fg_fan,
        'Total Gross Thrust (lbf, output)': get('perf.Fg', 'lbf'),
        'Fan Bypass Mass Flow (lb/s, input)': get('splitter.Fl_O2:stat:W', 'lbm/s'),
        'Bypass Ratio, BPR (output)': BPR,
        'Total Ram Drag (lbf, output)': ram_drag,
        'SFC ((lb/hr)/lbf)': fuel_flow / Fn,
        'Combustor Inlet Stagnation Temperature (R, output)': get(
            f'{last_compressor}.Fl_O:tot:T', 'degR'
        ),
        'Combustor Inlet Stagnation Pressure (psi, output)': get(
            f'{last_compressor}.Fl_O:tot:P', 'psi'
        ),
        'Combustor Inlet Static Temperature (R, output)': get(
            f'{last_compressor}.Fl_O:stat:T', 'degR'
        ),
        'Combustor Inlet Static Pressure (psi, output)': get(
            f'{last_compressor}.Fl_O:stat:P', 'psi'
        ),
        'FAR': get('balance.FAR', None),
        'Combustor Exit Stagnation Temperature (R, output)': get('burner.Fl_O:tot:T', 'degR'),
        'Combustor Exit Stagnation Pressure (psi, output)': get('burner.Fl_O:tot:P', 'psi'),
        'Combustor Exit Static Temperature (R, output)': get('burner.Fl_O:stat:T', 'degR'),
        'Combustor Exit Static Pressure (psi, output)': get('burner.Fl_O:stat:P', 'psi'),
        'Combustor Exit Static Density (kg/m^3, output)': get('burner.Fl_O:stat:rho', 'kg/m**3'),
        'Fan Pressure Ratio (FPR, input)': get('fan.PR', None),
        'Total Fan mass flow rate (kg/s)': get('fan.Fl_O:stat:W', 'kg/s'),
        'Fan Inlet Stagnation Temperature (K)': get('inlet.Fl_O:tot:T', 'K'),
        'Fan Outlet Stagnation Temperature (K)': get('fan.Fl_O:tot:T', 'K'),
        'Fan Rotational Speed (rpm)': get('LP_Nmech', 'rpm'),
        'Fan Area (m^2)': get('inlet.Fl_O:stat:area', 'm**2'),
        'Freestream Static Temperature (K)': get('fc.Fl_O:stat:T', 'K'),
        'Freestream Static Pressure (Pa)': get('fc.Fl_O:stat:P', 'Pa'),
        'Freestream Static Density (kg/m^3)': get('fc.Fl_O:stat:rho', 'kg/m**3'),
        'Core Mass Flow Rate (kg/s)': get('splitter.Fl_O1:stat:W', 'kg/s'),
        'Core Inlet Stagnation Temperature (K)': get('duct4.Fl_O:tot:T', 'K'),
        'Core Inlet Stagnation Pressure (Pa)': get('duct4.Fl_O:tot:P', 'Pa'),
        'Core Inlet Static Density (kg/m^3)': get('duct4.Fl_O:stat:rho', 'kg/m**3'),
        'Core Outlet Stagnation Temperature (K)': get('burner.Fl_O:tot:T', 'K'),
        'Core Outlet Stagnation Pressure (Pa)': get('burner.Fl_O:tot:P', 'Pa'),
        'Core Outlet Static Density (kg/m^3)': get('burner.Fl_O:stat:rho', 'kg/m**3'),
    }
    for nozz, label in (('core_nozz', 'Core'), ('byp_nozz', 'Bypass')):
        row[f'{label} Nozzle Area (m^2)'] = get(f'{nozz}.Fl_O:stat:area', 'm**2')
    for quantity, var, units in (
        ('Exhaust Velocity (m/s)', 'Fl_O:stat:V', 'm/s'),
        ('Exhaust Stagnation Temperature (K)', 'Fl_O:tot:T', 'K'),
        ('Exhaust Stagnation Pressure (Pa)', 'Fl_O:tot:P', 'Pa'),
        ('Exhaust Static Density (kg/m^3)', 'Fl_O:stat:rho', 'kg/m**3'),
        ('Exhaust Mach Number', 'Fl_O:stat:MN', None),
        ('Exhaust Static Temperature (K)', 'Fl_O:stat:T', 'K'),
        ('Exhaust Static Pressure (Pa)', 'Fl_O:stat:P', 'Pa'),
    ):
        for nozz, label in (('core_nozz', 'Core'), ('byp_nozz', 'Bypass')):
            row[f'{label} {quantity}'] = get(f'{nozz}.{var}', units)
    row['HP Shaft Speed (rpm, output)'] = get('HP_Nmech', 'rpm')
    row['LP Shaft Speed (rpm, output)'] = get('LP_Nmech', 'rpm')
    compressors = [
        ('fan', 'Fan'),
        ('lpc', 'LPC'),
        ('hpc_axi', 'HPC'),
        ('hpc_centri', 'Centrifugal'),
    ]
    compressors = [(comp, label) for comp, label in compressors if present(comp)]
    for comp, label in compressors:
        row[f'{label} PR'] = get(f'{comp}.PR', None)
    for comp, label in compressors:
        row[f'{label} Wc'] = get(f'{comp}.Wc', 'lbm/s')
    for turb, label in (('hpt', 'HPT'), ('lpt', 'LPT')):
        row[f'{label} PR'] = get(f'{turb}.PR', None)
    for turb, label in (('hpt', 'HPT'), ('lpt', 'LPT')):
        row[f'{label} Wp'] = get(f'{turb}.Wp', 'lbm/s')
    for comp, label in compressors:
        row[f'{label} Rline'] = get(f'{comp}.map.RlineMap', None)
    row['LPC Bleed Fraction'] = (
        get('bleed.bleed_frac', None) if prob.model.OD.options['lpc_bleed'] else 0.0
    )
    return row


def turboshaft_deck_row(prob, PC):
    """
    Collect the deck columns of a converged turboshaft part-power point.

    Parameters
    ----------
    prob : openmdao.api.Problem
        Converged problem from `build_problem`.
    PC : float
        Throttle of the point.

    Returns
    -------
    dict
        Column values keyed by the deck headers.
    """

    def get(var, units):
        if var.split('.')[0] in ('lpc', 'lpt', 'IP_Nmech') and not has_ip_spool:
            return None
        return float(prob.get_val(f'OD.{var}', units=units)[0])

    has_ip_spool = prob.model.OD._get_subsystem('lpt') is not None
    has_axial = prob.model.OD._get_subsystem('hpc_axi') is not None
    hpc = 'hpc_centri' if prob.model.OD._get_subsystem('hpc_centri') is not None else 'hpc_axi'
    power = get('lp_shaft.pwr_in', 'hp')
    fuel_flow = get('perf.Wfuel', 'lbm/s') * 3600.0
    row = {
        'Mach Number (input)': get('fc.MN', None),
        'Altitude (ft, input)': get('fc.alt', 'ft'),
        'Throttle (input)': PC,
        'Off-Design Core Shaft Power (hp, core output)': power,
        'Net Core Thrust (lbf, core output)': get('perf.Fn', 'lbf'),
        'Fuel Flow (lb/h, core output)': fuel_flow,
        'Gross Core Exhaust (lbf, output)': get('perf.Fg', 'lbf'),
        'Core Mass Flow (lb/s, input)': get('inlet.Fl_O:stat:W', 'lbm/s'),
        'Core Ram Drag (lbf, output)': get('inlet.F_ram', 'lbf'),
        'PSFC ((lb/hr)/hp)': fuel_flow / power,
        'Combustor Inlet Stagnation Temperature (R, output)': get(f'{hpc}.Fl_O:tot:T', 'degR'),
        'Combustor Inlet Stagnation Pressure (psi, output)': get(f'{hpc}.Fl_O:tot:P', 'psi'),
        'Combustor Inlet Static Temperature (R, output)': get(f'{hpc}.Fl_O:stat:T', 'degR'),
        'Combustor Inlet Static Pressure (psi, output)': get(f'{hpc}.Fl_O:stat:P', 'psi'),
        'FAR': get('balance.FAR', None),
        'Combustor Exit Stagnation Temperature (R, output)': get('burner.Fl_O:tot:T', 'degR'),
        'Combustor Exit Stagnation Pressure (psi, output)': get('burner.Fl_O:tot:P', 'psi'),
        'Combustor Exit Static Temperature (R, output)': get('burner.Fl_O:stat:T', 'degR'),
        'Combustor Exit Static Pressure (psi, output)': get('burner.Fl_O:stat:P', 'psi'),
        'Combustor Exit Density (kg/m^3, output)': get('burner.Fl_O:stat:rho', 'kg/m**3'),
        'Freestream Static Temperature (K, output)': get('fc.Fl_O:stat:T', 'K'),
        'Freestream Static Pressure (Pa, output)': get('fc.Fl_O:stat:P', 'Pa'),
        'Freestream Static Density (kg/m^3, output)': get('fc.Fl_O:stat:rho', 'kg/m**3'),
        'Core Inlet Stagnation Temperature (K, output)': get('inlet.Fl_O:tot:T', 'K'),
        'Core Inlet Stagnation Pressure (Pa, output)': get('inlet.Fl_O:tot:P', 'Pa'),
        'Core Inlet Density (kg/m^3, output)': get('inlet.Fl_O:stat:rho', 'kg/m**3'),
        'Core Nozzle Area (m^2, output)': get('nozzle.Fl_O:stat:area', 'm**2'),
        'Core Exhaust Velocity (m/s, output)': get('nozzle.Fl_O:stat:V', 'm/s'),
        'Core Exhaust Stagnation Temperature (K, output)': get('nozzle.Fl_O:tot:T', 'K'),
        'Core Exhaust Stagnation Pressure (Pa, output)': get('nozzle.Fl_O:tot:P', 'Pa'),
        'Core Exhaust Density (kg/m^3, output)': get('nozzle.Fl_O:stat:rho', 'kg/m**3'),
        'Core Exhaust Mach Number (output)': get('nozzle.Fl_O:stat:MN', None),
        'Core Exhaust Static Temperature (K, output)': get('nozzle.Fl_O:stat:T', 'K'),
        'Core Exhaust Static Pressure (Pa, output)': get('nozzle.Fl_O:stat:P', 'Pa'),
        'HP Shaft Speed (rpm, output)': get('HP_Nmech', 'rpm'),
        'IP Shaft Speed (rpm, output)': get('IP_Nmech', 'rpm'),
        'LP Shaft Speed (rpm, output)': get('LP_Nmech', 'rpm'),
        'LPC PR': get('lpc.PR', None),
        'HPC PR': get(f'{hpc}.PR', None),
        'LPC Wc': get('lpc.Wc', 'lbm/s'),
        'HPC Wc': get(f'{hpc}.Wc', 'lbm/s'),
        'HPT PR': get('hpt.PR', None),
        'LPT PR': get('lpt.PR', None),
        'HPT Wp': get('hpt.Wp', 'lbm/s'),
        'LPT Wp': get('lpt.Wp', 'lbm/s'),
        'LPC Rline': get('lpc.map.RlineMap', None),
        'HPC Rline': get(f'{hpc}.map.RlineMap', None),
    }
    if prob.model.OD.options['lpc_bleed']:
        row['LPC Bleed Fraction'] = get('bleed.bleed_frac', None)
    if has_axial and hpc != 'hpc_axi':
        row['Axial HPC PR'] = get('hpc_axi.PR', None)
        row['Axial HPC Wc'] = get('hpc_axi.Wc', 'lbm/s')
        row['Axial HPC Rline'] = get('hpc_axi.map.RlineMap', None)
    return {column: value for column, value in row.items() if value is not None}


def deck_envelope(name):
    """
    List the flight conditions of an included deck.

    Parameters
    ----------
    name : str
        Key of `decks.DECKS`.

    Returns
    -------
    list of tuple
        Sorted (Mach number, altitude [ft]) pairs.
    """
    deck = load_deck(name)
    pairs = zip(deck['Mach Number (input)'], deck['Altitude (ft, input)'])
    return sorted({(round(float(MN), 4), round(float(alt), 1)) for MN, alt in pairs})


def generate_deck(engine, flight_conditions, path=None, processes=None, **kwargs):
    """
    Generate an engine deck over a set of flight conditions, in parallel.

    Parameters
    ----------
    engine : str or dict
        Engine name, or a dictionary describing a new engine; see `engines.get_engine`.
    flight_conditions : sequence of tuple
        (Mach number, altitude [ft]) pairs.
    path : str or path-like, optional
        If given, write the deck to this CSV file.
    processes : int, optional
        Number of worker processes; defaults to the number of CPUs.
    **kwargs
        Passed to `run_flight_condition`, for example `throttles`; the default throttles are 1
        down to 0.625 of maximum T4 in steps of 0.025.

    Returns
    -------
    list of dict
        Deck rows, ordered as `flight_conditions` then by throttle.
    """
    run = functools.partial(_run_pair, engine, kwargs)
    with multiprocessing.Pool(processes) as pool:
        rows = [
            row for pair_rows in pool.map(run, flight_conditions, chunksize=1) for row in pair_rows
        ]
    if path is not None and rows:
        with open(path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def _run_pair(engine, kwargs, pair):
    # Failed solves print and warn at length; a failed point is simply left out of the deck
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return run_flight_condition(engine, *pair, **kwargs)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument(
        'engine', choices=sorted(engines.TURBOFAN_MODELS | engines.TURBOSHAFT_MODELS)
    )
    parser.add_argument('path', help='Output CSV file.')
    parser.add_argument('--processes', type=int, default=None)
    parser.add_argument(
        '--point',
        nargs=2,
        type=float,
        action='append',
        metavar=('MACH', 'ALT_FT'),
        help='Flight condition to run; repeatable. Defaults to those of the included deck.',
    )
    parser.add_argument(
        '--throttles',
        nargs='+',
        type=float,
        default=THROTTLES,
        metavar='PC',
        help='Throttles (fractions of maximum T4) to run at each flight condition, in order; '
        'the default is 1 down to 0.625 in steps of 0.025.',
    )
    args = parser.parse_args(argv)
    if not args.point and args.engine not in DECKS:
        parser.error(
            f'{args.engine} has no included deck to rerun; choose conditions with --point.'
        )
    points = args.point or deck_envelope(args.engine)
    throttles = tuple(args.throttles)
    rows = generate_deck(args.engine, points, args.path, args.processes, throttles=throttles)
    if not rows:
        raise SystemExit(f'No point converged, so {args.path} was not written.')
    requested = len(points) * len(throttles)
    print(
        f'Wrote {len(rows)} of {requested} points over {len(points)} flight conditions to {args.path}'
    )


if __name__ == '__main__':
    main()
