import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import openmdao.api as om
import pycycle.api as pyc
from openmdao.utils.assert_utils import assert_near_equal
from pycycle.elements.ambient import Ambient

from robust_pycycle_engines import engines
from robust_pycycle_engines.initial_guess import fan, turbofan, turboshaft

BASELINES = json.loads(
    (Path(__file__).parent / 'data' / 'initial_guess_baselines.json').read_text()
)


class TestTurbofanInitialGuess(unittest.TestCase):
    def test_matches_baselines(self):
        # The baseline inputs set eta_pcentrif = 1, a lossless centrifugal stage
        for name, case in BASELINES['turbofan'].items():
            with self.subTest(engine=name):
                cycle = turbofan.initial_guess(**case['inputs'], W_max=3000.0)
                for key, expected in case['expected'].items():
                    assert_near_equal(cycle[key], expected, tolerance=1e-6)

    def test_engines_reach_target_thrust(self):
        for name, inputs in engines.TURBOFANS.items():
            with self.subTest(engine=name):
                cycle = engines.initial_guess(name)
                assert_near_equal(cycle['Thrust_lbf'], inputs['target_lbf'], 1e-6)

    def test_engines_near_per_engine_models(self):
        # Paper-consistent shaft speed and cooling changes move mass flow only slightly
        for name in BASELINES['turbofan']:
            with self.subTest(engine=name):
                cycle = engines.initial_guess(name)
                expected = BASELINES['turbofan'][name]['per_engine_W_total']
                assert_near_equal(cycle['W_total'], expected, tolerance=0.02)

    def test_pressure_ratio_split(self):
        inputs = engines.TURBOFANS['PW4168']
        cycle = turbofan.initial_guess(**inputs)
        assert_near_equal(cycle['pi_hpc'], inputs['opr'] / (inputs['pi_fan'] * inputs['pi_lpc']))
        self.assertEqual(cycle['pi_centrif'], 1.0)
        self.assertEqual(cycle['Wdot_centrif'], 0.0)

        with_centrif = turbofan.initial_guess(**inputs, pi_centrif=2.0)
        assert_near_equal(with_centrif['pi_hpc'], cycle['pi_hpc'] / 2.0, 1e-12)
        self.assertGreater(with_centrif['Wdot_centrif'], 0.0)

    def test_generator_cooling_is_a_fraction_of_core_flow(self):
        cycle = turbofan.initial_guess(**engines.TURBOFANS['PW4168'], coolIG_frac_W=0.02)
        assert_near_equal(cycle['W_IG_cool'], 0.02 * cycle['W_core'], 1e-12)

    def test_opr_below_other_stages_raises(self):
        inputs = dict(engines.TURBOFANS['PW4168'], opr=3.0)
        with self.assertRaisesRegex(ValueError, 'axial HPC pressure ratio below 1'):
            turbofan.initial_guess(**inputs)

    def test_no_bracket_raises(self):
        inputs = dict(engines.TURBOFANS['AE3007'], target_lbf=1.0e7)
        with self.assertRaises(ValueError):
            turbofan.initial_guess(**inputs)


class TestStagesAtUnity(unittest.TestCase):
    # A pressure ratio or efficiency of exactly 1 must not divide by zero
    def assert_finite(self, cycle):
        for key, value in cycle.items():
            self.assertTrue(np.all(np.isfinite(np.real(value))), key)

    def test_turbofan(self):
        design = engines.TURBOFANS['PW4168']
        for changes in (
            {'pi_lpc': 1.0},
            {'pi_centrif': 1.0, 'eta_pcentrif': 1.0},
            {'pi_centrif': 1.5, 'eta_pcentrif': 1.0},
            {'pi_lpc': 1.0, 'eta_plpc': 1.0},
        ):
            with self.subTest(changes=changes):
                self.assert_finite(turbofan.initial_guess(**{**design, **changes}))

    def test_turboshaft(self):
        design = engines.TURBOSHAFTS['reference_9100hp']
        for changes in (
            {'pi_lpc': 1.0},
            {'pi_lpc': 1.0, 'eta_plpt': 1.0},
            {'pi_centrif': 1.0, 'eta_pcentrif': 1.0},
            {'pi_centrif': 2.0, 'eta_pcentrif': 1.0},
        ):
            with self.subTest(changes=changes):
                self.assert_finite(turboshaft.initial_guess(**{**design, **changes}))


class TestTurboshaftInitialGuess(unittest.TestCase):
    def test_matches_baselines(self):
        for name, case in BASELINES['turboshaft'].items():
            with self.subTest(engine=name):
                cycle = turboshaft.initial_guess(**case['inputs'])
                for key, expected in case['expected'].items():
                    assert_near_equal(cycle[key], expected, tolerance=1e-6)

    def test_engines_reach_target_power(self):
        for name, inputs in engines.TURBOSHAFTS.items():
            with self.subTest(engine=name):
                cycle = engines.initial_guess(name)
                assert_near_equal(cycle['hp_out'], inputs['target_hp'], 1e-6)

    def test_all_centrifugal_core_without_axial_pressure_rise(self):
        inputs = engines.TURBOSHAFTS['PW127']
        cycle = turboshaft.initial_guess(**inputs)
        self.assertEqual(cycle['pi_hpc'], 1.0)
        self.assertEqual(cycle['Wdot_hpc'], 0.0)
        assert_near_equal(cycle['eta_hpc'], 1.0, 1e-12)
        assert_near_equal(cycle['hp_out'], inputs['target_hp'], 1e-6)

    def test_uncooled_turbine(self):
        cycle = engines.initial_guess('PW127')
        self.assertEqual(cycle['W_cool_NGV'], 0.0)

    def test_engines_near_per_engine_models(self):
        for name in BASELINES['turboshaft']:
            with self.subTest(engine=name):
                # The per-engine mass flows were captured at the baseline nozzle pressure ratio
                nozzle_pr = BASELINES['turboshaft'][name]['inputs']['Nozzle_PR']
                design = {**engines.TURBOSHAFTS[name], 'Nozzle_PR': nozzle_pr}
                cycle = engines.initial_guess({'architecture': 'turboshaft', 'design': design})
                expected = BASELINES['turboshaft'][name]['per_engine_W']
                assert_near_equal(cycle['W'], expected, tolerance=0.04)

    def test_unknown_engine(self):
        with self.assertRaises(KeyError):
            engines.initial_guess('not_an_engine')


class _ElectricFan(pyc.Cycle):
    # Inlet, fan and nozzle, as in pyCycle's electric_propulsor example. On design the mass flow
    # matches the shaft power; off design it matches the design nozzle throat area, and the fan
    # speed matches the shaft power.
    def setup(self):
        design = self.options['design']
        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC
        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem(
            'fan', pyc.Compressor(map_data=pyc.FanMap, map_extrap=True, bleed_names=[])
        )
        self.add_subsystem('nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=0))
        balance = om.BalanceComp()
        balance.add_balance(
            'W',
            units='lbm/s',
            eq_units='hp' if design else 'inch**2',
            val=300.0,
            lower=1.0,
            upper=5000.0,
        )
        if design:
            self.add_subsystem('shaft', om.IndepVarComp('Nmech', 3000.0, units='rpm'))
            self.connect('shaft.Nmech', 'fan.Nmech')
            self.add_subsystem('balance', balance, promotes_inputs=[('rhs:W', 'pwr_target')])
            self.connect('fan.power', 'balance.lhs:W')
        else:
            balance.add_balance(
                'Nmech', val=3000.0, units='rpm', lower=10.0, upper=20000.0, eq_units='hp'
            )
            self.add_subsystem('balance', balance, promotes_inputs=[('rhs:Nmech', 'pwr_target')])
            self.connect('nozz.Throat:stat:area', 'balance.lhs:W')
            self.connect('balance.Nmech', 'fan.Nmech')
            self.connect('fan.power', 'balance.lhs:Nmech')
        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'nozz.Fl_I')
        self.connect('fc.Fl_O:stat:P', 'nozz.Ps_exhaust')
        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect('fan.Fl_O:tot:P', 'perf.Pt3')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozz.Fg', 'perf.Fg_0')
        self.connect('balance.W', 'fc.W')
        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-8
        newton.options['rtol'] = 1e-8
        newton.options['maxiter'] = 30
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['err_on_non_converge'] = True
        newton.options['iprint'] = -1
        newton.linesearch = om.BoundsEnforceLS()
        self.linear_solver = om.DirectSolver()
        super().setup()


class _MPElectricFan(pyc.MPCycle):
    def setup(self):
        self.pyc_add_pnt('DESIGN', _ElectricFan(design=True))
        self.pyc_add_pnt('OD', _ElectricFan(design=False))
        self.pyc_use_default_des_od_conns()
        self.pyc_connect_des_od('nozz.Throat:stat:area', 'balance.rhs:W')
        super().setup()


def _ambient(alt):
    prob = om.Problem(reports=False)
    prob.model.add_subsystem('atm', Ambient())
    prob.setup()
    prob.set_val('atm.alt', alt, units='ft')
    prob.run_model()
    return prob.get_val('atm.Ts', units='K')[0], prob.get_val('atm.Ps', units='Pa')[0]


class TestFanInitialGuess(unittest.TestCase):
    # Fan designs at cruise: FINCH's electric fan, and pyCycle's electric_propulsor example
    DESIGNS = {
        'FPR 1.685': {'power': 9100.0, 'FPR': 1.685, 'M': 0.8, 'alt': 35000.0},
        'FPR 1.2': {'power': 3486.657, 'FPR': 1.2, 'M': 0.8, 'alt': 32808.0},
    }
    # Off-design (Mach number, altitude [ft], fraction of design power)
    OFF_DESIGN = ((0.7, 30000.0, 1.0), (0.8, 35000.0, 0.7), (0.75, 30000.0, 0.4))
    ETA_PFAN = 0.9

    def _solve_design(self, d):
        g = 1.4
        eff = (d['FPR'] ** ((g - 1) / g) - 1) / (d['FPR'] ** ((g - 1) / (self.ETA_PFAN * g)) - 1)
        prob = om.Problem(reports=False)
        prob.model = _MPElectricFan()
        prob.setup()
        prob.set_val('DESIGN.fc.alt', d['alt'], units='ft')
        prob.set_val('DESIGN.fc.MN', d['M'])
        prob.set_val('DESIGN.inlet.MN', 0.6)
        prob.set_val('DESIGN.fan.PR', d['FPR'])
        prob.set_val('DESIGN.fan.eff', eff)
        prob.set_val('DESIGN.pwr_target', -d['power'], units='hp')
        prob.set_val('OD.fc.alt', d['alt'], units='ft')
        prob.set_val('OD.fc.MN', d['M'])
        prob.set_val('OD.pwr_target', -d['power'], units='hp')
        Tamb = _ambient(d['alt'])[0]
        W_guess = fan.design_mass_flow(d['power'], d['FPR'], self.ETA_PFAN, Tamb, d['M'])
        prob.set_val('DESIGN.balance.W', W_guess, units='lbm/s')
        prob.set_val('OD.balance.W', W_guess, units='lbm/s')
        with contextlib.redirect_stdout(io.StringIO()):
            prob.run_model()
        return prob, W_guess

    def test_design_mass_flow(self):
        for name, d in self.DESIGNS.items():
            with self.subTest(fan=name):
                prob, W_guess = self._solve_design(d)
                W = prob.get_val('DESIGN.balance.W', units='lbm/s')[0]
                assert_near_equal(W_guess, W, tolerance=0.02)

    def test_off_design_mass_flow_and_pressure_ratio(self):
        for name, d in self.DESIGNS.items():
            prob, _ = self._solve_design(d)
            solved = prob.model._outputs.asarray().copy()
            area = prob.get_val('DESIGN.nozz.Throat:stat:area', units='inch**2')[0]
            for M, alt, frac in self.OFF_DESIGN:
                with self.subTest(fan=name, M=M, alt=alt, power=frac):
                    # Start from the solved design point, with the guessed mass flow
                    prob.model._outputs.set_val(solved)
                    prob.set_val('OD.fc.MN', M)
                    prob.set_val('OD.fc.alt', alt, units='ft')
                    prob.set_val('OD.pwr_target', -d['power'] * frac, units='hp')
                    Tamb, Pamb = _ambient(alt)
                    W_guess, FPR_guess = fan.off_design_mass_flow(
                        d['power'] * frac, area, Tamb, Pamb, M, self.ETA_PFAN
                    )
                    prob.set_val('OD.balance.W', W_guess, units='lbm/s')
                    with contextlib.redirect_stdout(io.StringIO()):
                        prob.run_model()
                    assert_near_equal(W_guess, prob.get_val('OD.balance.W', units='lbm/s')[0], 0.03)
                    assert_near_equal(FPR_guess, prob.get_val('OD.fan.PR')[0], 0.03)

    def test_off_design_at_design_condition_recovers_design(self):
        d = self.DESIGNS['FPR 1.685']
        prob, _ = self._solve_design(d)
        area = prob.get_val('DESIGN.nozz.Throat:stat:area', units='inch**2')[0]
        Tamb, Pamb = _ambient(d['alt'])
        W_guess, FPR_guess = fan.off_design_mass_flow(d['power'], area, Tamb, Pamb, d['M'])
        assert_near_equal(W_guess, prob.get_val('DESIGN.balance.W', units='lbm/s')[0], 0.02)
        assert_near_equal(FPR_guess, d['FPR'], 0.01)


if __name__ == '__main__':
    unittest.main()
