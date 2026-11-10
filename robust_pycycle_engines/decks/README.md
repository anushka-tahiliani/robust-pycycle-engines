# Engine decks

Tabulated engine performance from the pyCycle models. Inputs are Mach number, altitude, and throttle,
where throttle is the fraction of the local maximum turbine inlet temperature. Each column header
states its units and whether it is an input or output.

| Key | File | Type | Rows | Converged | Mach | Altitude [ft] | Throttle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `AE3007` | `AE_3007_deck.csv` | Turbofan | 13,330 | 99.9% | 0.05-0.9 | 0-40,000 | 0.625-1 |
| `CF34-8E` | `CF34_8E_deck.csv` | Turbofan | 14,217 | 99.4% | 0.05-0.9 | 0-40,000 | 0.625-1 |
| `CFM56` | `CFM56_UPS_deck.csv` | Turbofan | 17,253 | 94.8% | 0.05-0.9 | 0-40,000 | 0.625-1 |
| `GE90` | `GE90_deck.csv` | Turbofan | 12,882 | 95.7% | 0.05-0.9 | 0-40,000 | 0.625-1 |
| `PW4168` | `PW_4168_deck.csv` | Turbofan | 8,037 | 99.7% | 0.05-0.9 | 0-40,500 | 0.625-1 |
| `PW127` | `PW_127_deck.csv` | Turboshaft | 4,960 | 100% | 0.05-0.8 | 0-24,000 | 0.625-1 |
| `TPE331-10` | `TPE331-10_deck.csv` | Turboshaft | 1,856 | 100% | 0.1-0.8 | 0-27,500 | 0.625-1 |

Each flight condition sweeps throttle from 1 down to 0.625 in steps of 0.025; "Converged" is the
share of those points in the deck. Missing points are left out rather than filled in. Most are at
the low-throttle end above about 22,000 ft on `CFM56` and `GE90`.

The columns differ between decks, so list them with `load_deck(key).keys()`. All values are pyCycle
model outputs, not manufacturer data. The design points were set with back-of-the-envelope cycle
calculations and are not validated, so treat the decks as performance estimates.

All the decks were generated with `robust_pycycle_engines.deck` (see the top-level README), so
`Throttle (input)` is a fraction of maximum T4 throughout. Turbofan decks include an
`LPC Bleed Fraction` column, which is zero for `CF34-8E` (no LPC). Turboshaft decks give shaft power
under `Off-Design Core Shaft Power (hp, core output)`; `PW127` also has `LPC Bleed Fraction`.

The decks were run on a Linux cluster. A few flight conditions that failed there were rerun on their
own, so regenerating a deck can give a slightly different set of converged points.
