"""
Example: model a CFM56-class turbofan and generate a small engine deck.

The script defines the engine, computes its initial guess, runs the pyCycle model at cruise, and
writes a deck over a few flight conditions to `CFM56_example.csv` in the current directory. To model
another turbofan, change the values in `ENGINE`; to change where it runs, edit the flight
conditions and throttles below it. Run it with::

    python examples/cfm56_turbofan.py

The design point approximates the engine with back-of-the-envelope cycle calculations; it is not
manufacturer data.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

from robust_pycycle_engines import deck, engines

ENGINE = {
    'architecture': 'turbofan',
    'design': {
        # Design flight condition: cruise at the model's design altitude of 35,000 ft
        'M': 0.8,
        'Tamb': 218.8,  # [K] ambient static temperature
        'Pamb': 23840.0,  # [Pa] ambient static pressure
        # Cycle: the axial HPC supplies the pressure ratio the fan and LPC do not
        'target_lbf': 5900.0,  # [lbf] design thrust
        'opr': 30.0,  # overall pressure ratio
        'pi_fan': 1.685,
        'pi_lpc': 1.935,  # set to 1 for an engine with no booster
        'BPR': 5.1,
        'TIT': 1587.222,  # [K] turbine inlet temperature
        'T_metal': 1300.0,  # [K] turbine metal temperature of the initial guess's cooling model
        'pwr_offtk_hpt': 250.0,  # [hp] power off-take from the HP shaft
        # Polytropic efficiencies; those left out default to 0.9. The HPC value is used by the guess
        'eta_phpc_init': 0.903,
        'eta_phpt': 0.873,
        'eta_plpt': 0.882,
    },
    # Overrides of `robust_pycycle_engines.cycles.turbofan.DEFAULT_SETTINGS`
    'model': {
        'W_max_design': 1000.0,  # [lbm/s] upper bound of the design mass flow
        'W_max_off_design': 5000.0,  # [lbm/s] upper bound of the off-design mass flow
        # Design point of each compressor on its map; these values repeat the defaults
        'map_points': {
            'fan': {'RlineMap': 1.8},
            'lpc': {'RlineMap': 1.6},
            'hpc_axi': {'RlineMap': 2.05},
        },
    },
}

# Where to run the engine. Throttle is the fraction of maximum T4. Each throttle starts from the
# last converged one, so list them from high to low in small steps; large steps fail more often.
CRUISE = (0.8, 35000.0)  # (Mach number, altitude [ft]) of the cruise printout
CRUISE_THROTTLES = (1.0, 0.95, 0.9)
# (Mach number, altitude [ft]) of each deck flight condition; add pairs to cover more of the envelope
DECK_FLIGHT_CONDITIONS = [(0.3, 10000.0), (0.6, 25000.0), (0.8, 35000.0)]
# Deck throttles: the default, deck.THROTTLES, is 1 down to 0.625 in steps of 0.025. To change it,
# list your own, for example DECK_THROTTLES = (1.0, 0.95, 0.9, 0.85, 0.8)
DECK_THROTTLES = deck.THROTTLES
DECK_PATH = 'CFM56_example.csv'


def print_initial_guess():
    """Compute the initial guess that seeds the pyCycle model and print part of it."""
    cycle = engines.initial_guess(ENGINE)
    print('Initial guess at the design point')
    print(f'  total mass flow  {cycle["W_total"]:8.1f} kg/s')
    print(f'  fuel-air ratio   {cycle["FAR"]:8.4f}')
    print(f'  HPC pressure ratio {cycle["pi_hpc"]:6.2f}')
    print(f'  thrust           {cycle["Thrust_lbf"]:8.0f} lbf')


def cruise_performance(flight_condition=CRUISE, throttles=CRUISE_THROTTLES):
    """
    Run the pyCycle model at one flight condition over a few throttles.

    Parameters
    ----------
    flight_condition : tuple of float
        (Mach number, altitude [ft]).
    throttles : sequence of float
        Throttles to run, in order.

    Returns
    -------
    list of dict
        Deck row of each converged throttle; a throttle that fails is left out.
    """
    return deck.run_flight_condition(ENGINE, *flight_condition, throttles=tuple(throttles))


def print_cruise_performance(rows, flight_condition=CRUISE):
    """Print the throttle sweep from `cruise_performance`."""
    MN, alt = flight_condition
    print(f'\nMach {MN}, {alt:,.0f} ft (throttle is the fraction of maximum T4)')
    print('  throttle  thrust [lbf]  fuel [lb/h]  TSFC [(lb/h)/lbf]  LPC bleed')
    for row in rows:
        print(
            f'  {row["Throttle (input)"]:8.3f}'
            f'  {row["Total Net Thrust (lbf, output)"]:12.0f}'
            f'  {row["Fuel Flow (lb/h, core output)"]:11.0f}'
            f'  {row["SFC ((lb/hr)/lbf)"]:17.3f}'
            f'  {row["LPC Bleed Fraction"]:9.3f}'
        )


def write_deck(flight_conditions=DECK_FLIGHT_CONDITIONS, throttles=DECK_THROTTLES, path=DECK_PATH):
    """
    Generate a small deck and write it to a CSV file.

    Parameters
    ----------
    flight_conditions : sequence of tuple
        (Mach number, altitude [ft]) pairs.
    throttles : sequence of float
        Throttles to run at each flight condition, in order.
    path : str
        Output CSV file.

    Returns
    -------
    list of dict
        Deck rows.
    """
    rows = deck.generate_deck(ENGINE, flight_conditions, path=path, throttles=tuple(throttles))
    requested = len(flight_conditions) * len(throttles)
    print(f'\nWrote {len(rows)} of {requested} points to {path}; failed points are left out')
    return rows


if __name__ == '__main__':  # the deck's flight conditions run in parallel processes
    print_initial_guess()
    print_cruise_performance(cruise_performance())
    write_deck()
