# Robust pyCycle engines

Physics-based initial guesses for [pyCycle](https://github.com/OpenMDAO/pyCycle) engine models, the
pyCycle cycle models they seed, and tabulated engine decks generated with them.

pyCycle's Newton solver depends strongly on its starting point, and that sensitivity grows as an
optimizer moves through a wide design space. The initial guesses here produce an analytically
consistent starting state for mass flow, fuel-air ratio, turbine pressure ratios, cooling flows, and
shaft speeds before the solver runs, so the design point needs no warm start or tabulated guess.

The method is described in:

> A. Tahiliani, H. Hajdik, A. Lamkin, M. Folk, and A. Uranga, "Robust Approach to Engine Cycle
> Modeling to Enable Multidisciplinary Optimization of Advanced Aircraft," *AIAA AVIATION 2026
> Forum*, 2026. [doi:10.2514/6.2026-4499](https://doi.org/10.2514/6.2026-4499)

```bibtex
@inproceedings{Tahiliani2026,
  author    = {Tahiliani, Anushka and Hajdik, Hannah and Lamkin, Andrew and Folk, Masha and Uranga, Alejandra},
  title     = {Robust Approach to Engine Cycle Modeling to Enable Multidisciplinary Optimization of Advanced Aircraft},
  booktitle = {AIAA AVIATION 2026 Forum},
  publisher = {American Institute of Aeronautics and Astronautics},
  year      = {2026},
  doi       = {10.2514/6.2026-4499},
}
```

## Installation

Requires Python 3.10 or later.

```bash
git clone https://github.com/anushka-tahiliani/robust-pycycle-engines.git
cd robust-pycycle-engines
pip install -e .
```

This installs [pyCycle](https://github.com/OpenMDAO/pyCycle) 4.4 or later from PyPI. The initial
guesses themselves need only NumPy, SciPy, and OpenMDAO.

## What is included

| Engine model | Initial guess | How it closes |
| --- | --- | --- |
| Turbofan, direct-drive or geared | [`turbofan.py`](robust_pycycle_engines/initial_guess/turbofan.py) | Root find in total mass flow to match thrust |
| Turboshaft with free power turbine | [`turboshaft.py`](robust_pycycle_engines/initial_guess/turboshaft.py) | Root find in core mass flow to match shaft power |
| Electrically driven fan | [`fan.py`](robust_pycycle_engines/initial_guess/fan.py) | Closed form at design, nozzle and power balance off-design |

[`balances.py`](robust_pycycle_engines/initial_guess/balances.py) maps those states onto pyCycle balance variables on
and off design, including the geared variables (LP shaft speed, fan speed, and gearbox torque).
[`engines.py`](robust_pycycle_engines/engines.py) lists the design point of each studied engine, and
[`robust_pycycle_engines/decks`](robust_pycycle_engines/decks/README.md) holds the engine decks.

The design points (pressure ratios, bypass ratio, turbine inlet temperature, thrust or power,
off-takes) were set with back-of-the-envelope cycle calculations to approximate each engine. They
are not manufacturer data and the models are not validated against it. No accuracy is claimed, so
use them for relative cycle studies.

The cycle models in [`robust_pycycle_engines/cycles`](robust_pycycle_engines/cycles) are a direct-drive turbofan
(PW4168, GE90, CF34-8E, AE3007, CFM56) and a turboshaft with a free power turbine and an
optional IP spool (PW127, `reference_9100hp`, TPE331-10). Their layouts are notional: they
approximate each engine's performance, not its hardware. The electric fan and geared turbofan have
initial guesses only. pyCycle has no centrifugal compressor map, so centrifugal stages use its LPC
map, and [`cooling.py`](robust_pycycle_engines/cycles/cooling.py) is a version of pyCycle's turbine
cooling model in which each rotor does its work before its coolant mixes in. [`deck.py`](robust_pycycle_engines/deck.py) generates decks with them
(see [Make a deck](#make-a-deck)):

- Each flight condition sweeps throttle from 1 down to 0.625. The first throttle starts from the
  initial guess and each later one from the last converged throttle; failed throttles are skipped.
  If no throttle converges, the flight condition is retried with the mass flow guess scaled by 0.99
  and then 1.01.
- Turbofans throttle on T4. At every altitude, LPC bleed keeps the LPC at or above a target R-line
  (its operating line on the compressor map, where 1 is the stall line), linear in throttle, and is
  zero when the LPC is already above that line. Pass `lpc_bleed=False` to `run_flight_condition`
  to turn it off. Below 10,000 ft maximum T4 is raised 2.5% for takeoff.
- Turboshafts also throttle on T4, as a fraction of maximum T4. With the `lpc_bleed` setting (PW127),
  LPC bleed keeps the LPC at or above that R-line schedule in the same way.

### Architectures you can represent

The overall pressure ratio `opr` is an input, and the axial HPC supplies whatever the other
compressors do not, so the compressor layout follows from the pressure ratios you set. The details
are in the [initial guess notes](robust_pycycle_engines/initial_guess/README.md).

| Architecture | How to set it | Example |
| --- | --- | --- |
| Turbofan: fan, LPC, axial HPC | `pi_fan`, `pi_lpc`, `opr` | `PW4168`, `GE90` |
| Turbofan without a booster | `pi_lpc = 1` | `CF34-8E` |
| Turbofan with a centrifugal stage after the HPC | `pi_centrif > 1` | `PW4168` with `pi_centrif=1.5` |
| Turboshaft: LPC, axial HPC, centrifugal stage | `pi_lpc`, `pi_centrif`, `opr` | `reference_9100hp` |
| Turboshaft with an all-centrifugal core | `opr = pi_lpc * pi_centrif` | `PW127` |
| Turboshaft: centrifugal core, no LPC, free power turbine | `pi_lpc = 1`, `pi_centrif = opr` | `TPE331-10` |

Each of these works in both the initial guess and the pyCycle cycle model. A pressure ratio of 1
only tells the initial guess that a stage does no work: pyCycle models a stage only when its
pressure ratio is above 1.

```python
from robust_pycycle_engines import deck, engines
from robust_pycycle_engines.initial_guess import turbofan, turboshaft

# Turboshaft guess, axial HPC then centrifugal stage: pi_hpc = 7.42 / (1.9 * 2.0) = 1.95
cycle = turboshaft.initial_guess(**engines.TURBOSHAFTS['reference_9100hp'])
print(f"pi_hpc = {cycle['pi_hpc']:.2f}, centrifugal PR = {cycle['Pt3_centrif'] / cycle['Pt3']:.2f}")

# Turbofan guess with a centrifugal stage added: pi_hpc = 32 / (1.75 * 2.2 * 1.5) = 5.54
cycle = turbofan.initial_guess(**{**engines.TURBOFANS['PW4168'], 'pi_centrif': 1.5})
print(f"pi_hpc = {cycle['pi_hpc']:.2f}, centrifugal PR = {cycle['Pt3_centrif'] / cycle['Pt3']:.2f}")

# Cycle model with an all-centrifugal core: pi_lpc = 2.6, pi_centrif = 5.65, no axial HPC
rows = deck.run_flight_condition('PW127', 0.2, 0.0, throttles=(1.0, 0.9))
print([round(row['Off-Design Core Shaft Power (hp, core output)']) for row in rows])
```

### Integrated generator (optional)

The initial guesses and cycle models can include a generator housed in the LPC hub, as in the
integrated compressor-generator concept of MIT's megawatt-class motor drive work:

- Z. S. Spakovszky et al., "A Megawatt-Class Electrical Machine Technology Demonstrator for
  Turbo-Electric Propulsion," *AIAA AVIATION 2023 Forum*, 2023.
  [hdl:1721.1/150870](https://hdl.handle.net/1721.1/150870)
- Y. Chen, "Technology Demonstration of a Megawatt-Class Integrated Motor Drive for Aircraft
  Propulsion," Ph.D. thesis, Massachusetts Institute of Technology, 2023.
  [hdl:1721.1/154196](https://hdl.handle.net/1721.1/154196)

Set `coolIG_frac_W`, the fraction of the core flow bled through the LPC to cool the generator, in
an engine's design point. The cooling flow leaves the LPC halfway through its work (LPC port
`cool_IG`). It defaults to 0, meaning no generator, which is how every engine in `engines.py` is set up. Only
the cooling flow is modeled; generator power is not computed.

## Usage

Compute the design state of a studied engine:

```python
from robust_pycycle_engines import engines

cycle = engines.initial_guess('PW4168')
print(f"W = {cycle['W_total']:.1f} kg/s, FAR = {cycle['FAR']:.4f}, pi_hpc = {cycle['pi_hpc']:.3f}")
```

Load one of the included decks:

```python
from robust_pycycle_engines.decks import load_deck

thrust = load_deck('PW4168')['Total Net Thrust (lbf, output)']
```

### Make a deck

[`examples/cfm56_turbofan.py`](examples/cfm56_turbofan.py) is a complete script to start from: it
defines a CFM56-class turbofan, runs it at cruise, and writes a small deck. Change its `ENGINE`
values to model another turbofan, and `CRUISE`, `CRUISE_THROTTLES`, `DECK_FLIGHT_CONDITIONS` and
`DECK_THROTTLES` to choose where it runs.

```bash
python examples/cfm56_turbofan.py
```

The steps on their own: pick an engine, the flight conditions to run, as (Mach number, altitude
[ft]) pairs, and optionally the throttles. The engines with a cycle model are `PW4168`, `GE90`, `CF34-8E`, `AE3007`, `CFM56`,
`PW127`, `TPE331-10` and `reference_9100hp` (the Mach 0.8 reference design of the paper); to run a
different one, see [Your own engine](#your-own-engine).

```python
import itertools

from robust_pycycle_engines import deck

if __name__ == '__main__':  # the flight conditions run in parallel processes
    machs = [0.2, 0.4, 0.6, 0.8]
    altitudes = [0.0, 10000.0, 20000.0, 30000.0]
    flight_conditions = list(itertools.product(machs, altitudes))
    # Throttle is the fraction of maximum T4. The default, deck.THROTTLES, is 1 down to 0.625 in
    # steps of 0.025; to change it, list your own, for example throttles = (1.0, 0.95, 0.9)
    throttles = deck.THROTTLES
    rows = deck.generate_deck(
        'PW4168', flight_conditions, path='PW4168.csv', processes=8, throttles=throttles
    )
    print(f'{len(rows)} rows')
```

For one flight condition, `deck.run_flight_condition('PW4168', 0.8, 35000.0, throttles=(1.0, 0.9))`
returns the rows without writing a file.

Or from the command line, with `--point` repeated for each flight condition and the throttles
listed after `--throttles`:

```bash
python -m robust_pycycle_engines.deck PW4168 PW4168.csv --processes 8 --point 0.8 35000 --point 0.3 5000 --throttles 1 0.95 0.9
```

Without `--throttles`, each flight condition runs the default throttles: 1 down to 0.625 of
maximum T4 in steps of 0.025. To change them, list your own after `--throttles`, as above.
Leaving out `--point` reruns every flight
condition of the included deck of that engine, which takes a while: a flight condition takes about
half a minute on one core, and longer if throttles fail.

A generated CSV has one row per converged point, and most column headers give their units:

- `Mach Number (input)`, `Altitude (ft, input)` and `Throttle (input)` identify the point. Throttle
  is the fraction of maximum T4. The default is 1 down to 0.625 in steps of 0.025. To change it,
  pass `throttles=` or `--throttles`, listed from high to low in small steps: each throttle starts
  from the last converged one, and large steps fail more often. On turbofans below 10,000 ft,
  throttle 1 is 1.025 times the design T4 (takeoff rating), so decks step at that altitude.
- Thrust is `Total Net Thrust (lbf, output)` for a turbofan, and shaft power is
  `Off-Design Core Shaft Power (hp, core output)` for a turboshaft. Fuel flow is
  `Fuel Flow (lb/h, core output)` for both. On a turbofan, `Off-Design Core Shaft Power` holds the
  fan shaft power. `HPC PR` is the axial HPC on a turbofan, and the last compressor (the centrifugal
  stage, if there is one) on a turboshaft.
- A point that does not converge is left out, not flagged. Compare the row count with the
  conditions you asked for: a full sweep has 16 rows per flight condition.

### Your own engine

Describe an engine with a dictionary and pass it anywhere an engine name is accepted:

```python
from robust_pycycle_engines import deck, engines

my_turbofan = {
    'architecture': 'turbofan',  # or 'turboshaft'
    # Design point, with the keys of an engines.TURBOFANS (or TURBOSHAFTS) entry
    'design': {**engines.TURBOFANS['PW4168'], 'BPR': 6.0, 'target_lbf': 22000.0},
    # Optional overrides of robust_pycycle_engines.cycles.turbofan.DEFAULT_SETTINGS
    'model': {},
}

if __name__ == '__main__':
    flight_conditions = [(0.8, 35000.0), (0.3, 5000.0)]  # (Mach number, altitude [ft])
    # Fractions of maximum T4; the default is 1 down to 0.625 in steps of 0.025. To change it,
    # list your own, for example throttles = (1.0, 0.95, 0.9)
    throttles = deck.THROTTLES
    deck.generate_deck(my_turbofan, flight_conditions, path='my_turbofan.csv', throttles=throttles)
```

A turboshaft looks the same, with `Nozzle_PR` and `target_hp` in place of `BPR`, `pi_fan` and
`target_lbf`:

```python
my_turboshaft = {
    'architecture': 'turboshaft',
    'design': {**engines.TURBOSHAFTS['TPE331-10'], 'target_hp': 1000.0},
    'model': engines.TURBOSHAFT_MODELS['TPE331-10'],  # its design altitude and duct losses
}
```

- The required `design` keys are the arguments without defaults of `initial_guess` in
  [`initial_guess/turbofan.py`](robust_pycycle_engines/initial_guess/turbofan.py) and
  [`turboshaft.py`](robust_pycycle_engines/initial_guess/turboshaft.py).
- The design point is at the model's `design_alt` (35,000 ft for turbofans, sea level for
  turboshafts). To move it, set `design_alt` in `model` and `M`, `Tamb` and `Pamb` in `design` to
  the same flight condition. An unknown `model` key raises an error.
- `map_points` in `model` sets where each compressor's design point sits on its map:
  `'model': {'map_points': {'lpc': {'RlineMap': 1.8, 'NcMap': 0.95}}}`. The compressors are `lpc`,
  `hpc_axi` and `hpc_centri`, plus `fan` on a turbofan. The defaults are in [`maps.py`](robust_pycycle_engines/cycles/maps.py): R-line 1.8 on the fan,
  1.6 on the LPC and centrifugal stage and 2.05 on the HPC. R-line 1 is the stall line, so a low
  design R-line leaves little stall margin at part throttle; raise it if a compressor stalls there.
- The guess and the model share the design point, pressure ratios, turbine efficiencies and the
  compressor polytropic efficiencies `eta_pfan`, `eta_plpc` and `eta_pcentrif`.
- These `design` keys are used by the initial guess only: `T_metal` (the model's turbine cooling
  uses 2460 degR, about 1367 K), `in_dP`, `com_dPqP`, `disable_cooling` (the model always cools the
  HPT) and, on turbofans, `eta_phpc_init` (the model uses 0.9). Set model losses with `cycle_params`
  on a turboshaft.
- If points fail, try raising `maxiter`, the `W_max_design` and `W_max_off_design` bounds in `model`
  (or `W_max` in `design` if the guess finds no root), finer `throttles`, or more
  `retry_flow_scales`. To debug one point, build it with
  `prob = deck.build_problem(my_turbofan, 0.8, 35000.0)`, set the throttle with
  `prob.set_val('OD.PC', 1.0)`, call `prob.set_solver_print(level=2)` and then `prob.run_model()`;
  this solves the design and off-design points together.

### Seeding your own pyCycle model

[`balances.py`](robust_pycycle_engines/initial_guess/balances.py) converts the cycle state to pyCycle balance
values. Assign them in the model's `guess_nonlinear`:

```python
from robust_pycycle_engines.initial_guess import balances

def guess_nonlinear(self, inputs, outputs, residuals):
    cycle = self.options['cycle']  # the state returned by initial_guess
    for name, value in balances.turbofan_design_guess(cycle).items():
        outputs[name] = value
```

The dictionary is keyed by the balance names in
[`cycles/turbofan.py`](robust_pycycle_engines/cycles/turbofan.py) (for example `balance.W` and
`balance.cool_NGV_frac_W`), so name your balances to match or remap the keys.

`balances.turbofan_off_design_guess` scales the design state to an off-design flight condition and
throttle (Eqs. 7-11 of the paper). The deck generator does not use it: it starts each flight condition's off-design point from the design state, with the
mass flow and inlet statics scaled to the flight condition at fixed corrected flow. Seeding with
`turbofan_off_design_guess` instead did not converge at the PW4168 and GE90 points tried.

## Units

The initial-guess cycle state is in SI units, except for target thrust [lbf], shaft power [hp], and
shaft speeds [rpm]. `balances.py` converts to the imperial units pyCycle uses. The docstring of
[`fan.py`](robust_pycycle_engines/initial_guess/fan.py) gives its own units, which include lbm/s.

## Development

```bash
pip install -e .[dev]
ruff format --check .
ruff check .
testflo tests
```

The regression baselines in `tests/data` come from stand-alone models of these engines and from the
[FINCH](https://github.com/ideas-um/FINCH) implementation of the same method.

## License

Licensed under the [Apache License 2.0](LICENSE).

## Acknowledgments

This work was partly supported by the National Aeronautics and Space Administration under contract
number 80GRC024CA036, and carried out under Electra's leadership.
