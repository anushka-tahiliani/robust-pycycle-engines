import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from robust_pycycle_engines import deck, engines
from robust_pycycle_engines.cycles import components


class TestTurbofanCycle(unittest.TestCase):
    def test_PW4168_cruise_matches_deck(self):
        # Full-throttle and part-throttle cruise points of the PW4168 deck
        rows = deck.run_flight_condition('PW4168', 0.8, 35000.0, throttles=(1.0, 0.975))
        self.assertEqual([row['Throttle (input)'] for row in rows], [1.0, 0.975])
        thrust = [row['Total Net Thrust (lbf, output)'] for row in rows]
        fuel_flow = [row['Fuel Flow (lb/h, core output)'] for row in rows]
        assert_near_equal(thrust, [20500.0, 18412.5], tolerance=1e-3)
        assert_near_equal(fuel_flow, [12765.9, 11367.9], tolerance=1e-3)
        self.assertLess(rows[0]['LPC Bleed Fraction'], 1e-3)  # on the schedule at the design point
        self.assertGreater(rows[1]['LPC Bleed Fraction'], 1e-3)

    def test_PW4168_bleeds_at_low_altitude(self):
        # Without bleed the LPC reaches its stall line before the lowest throttle
        rows = deck.run_flight_condition('PW4168', 0.6, 5000.0, throttles=(1.0, 0.9))
        self.assertEqual(len(rows), 2)
        self.assertGreater(rows[1]['LPC Bleed Fraction'], 1e-3)
        assert_near_equal(
            rows[1]['LPC Rline'], 1.6 + (1.1 - 1.6) * (1.0 - 0.9) / 0.375, tolerance=1e-3
        )

    def test_turning_off_bleed(self):
        rows = deck.run_flight_condition(
            'PW4168', 0.3, 5000.0, throttles=(1.0, 0.9), lpc_bleed=False
        )
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row['LPC Bleed Fraction'], 0.0)

    def test_PW4168_needs_no_bleed_above_the_schedule(self):
        # Holding the schedule here would take negative bleed, which has no solution
        rows = deck.run_flight_condition('PW4168', 0.5, 32000.0, throttles=(1.0,))
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]['LPC Rline'], 1.6)
        self.assertLess(rows[0]['LPC Bleed Fraction'], 1e-3)


class TestTurboshaftCycle(unittest.TestCase):
    def test_PW127_design_condition_throttles_on_T4(self):
        rows = deck.run_flight_condition('PW127', 0.2, 0.0, throttles=(1.0, 0.975))
        power = [row['Off-Design Core Shaft Power (hp, core output)'] for row in rows]
        assert_near_equal(power[0], 2750.0, tolerance=1e-4)
        self.assertLess(power[1], power[0])

    def test_PW127_bleeds_to_stay_off_the_LPC_stall_line(self):
        rows = deck.run_flight_condition('PW127', 0.2, 0.0, throttles=(1.0, 0.8))
        self.assertEqual(len(rows), 2)
        self.assertLess(rows[0]['LPC Bleed Fraction'], 1e-3)
        self.assertGreater(rows[1]['LPC Bleed Fraction'], 0.05)
        assert_near_equal(
            rows[1]['LPC Rline'], 1.6 + (1.1 - 1.6) * (1.0 - 0.8) / 0.375, tolerance=1e-3
        )

    def test_PW127_needs_no_bleed_above_the_schedule(self):
        rows = deck.run_flight_condition('PW127', 0.8, 24000.0, throttles=(1.0,))
        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]['LPC Rline'], 1.6)
        self.assertLess(rows[0]['LPC Bleed Fraction'], 1e-3)

    def test_first_throttle_failure_does_not_spoil_the_next(self):
        # At full T4 the LPC runs off the top of its map; the next throttle must still converge
        rows = deck.run_flight_condition('PW127', 0.2, 24000.0, throttles=(1.0, 0.95))
        self.assertEqual([row['Throttle (input)'] for row in rows], [0.95])


class TestOneSidedBleed(unittest.TestCase):
    def _bleed(self, rline, target):
        prob = om.Problem(reports=False)
        prob.model.add_subsystem('bleed', components.OneSidedBleed())
        prob.setup(force_alloc_complex=True)
        prob.set_val('bleed.RlineMap', rline)
        prob.set_val('bleed.RlineTarget', target)
        return prob

    def test_partials(self):
        for rline, bleed in ((1.3, 0.05), (1.7, 0.0), (1.5, 0.1)):
            prob = self._bleed(rline, 1.5)
            prob.set_val('bleed.bleed_frac', bleed)
            prob.run_model()
            assert_check_partials(prob.check_partials(out_stream=None, method='cs'))

    def test_residual_vanishes_only_on_the_complementary_branches(self):
        def residual(rline, bleed):
            prob = self._bleed(rline, 1.5)
            prob.set_val('bleed.bleed_frac', bleed)
            prob.final_setup()
            prob.model.run_apply_nonlinear()
            return float(prob.model.bleed._residuals['bleed_frac'][0])

        self.assertLess(abs(residual(1.7, 0.0)), 1e-6)  # above the schedule, no bleed
        self.assertLess(abs(residual(1.5, 0.1)), 1e-6)  # on the schedule, bleeding
        self.assertGreater(abs(residual(1.3, 0.05)), 1e-2)  # below the schedule is not a solution


class TestArchitectures(unittest.TestCase):
    def test_turbofan_without_booster_has_no_lpc(self):
        rows = deck.run_flight_condition('CF34-8E', 0.8, 35000.0, throttles=(1.0, 0.975))
        self.assertEqual([row['Throttle (input)'] for row in rows], [1.0, 0.975])
        assert_near_equal(rows[0]['Total Net Thrust (lbf, output)'], 4350.0, tolerance=1e-3)
        self.assertNotIn('LPC PR', rows[0])
        self.assertEqual(rows[0]['LPC Bleed Fraction'], 0.0)

    def test_turbofan_with_centrifugal_stage(self):
        engine = {
            'architecture': 'turbofan',
            'design': {**engines.TURBOFANS['PW4168'], 'pi_centrif': 1.5},
            'model': {**engines.TURBOFAN_MODELS['PW4168'], 'maxiter': 15},
        }
        rows = deck.run_flight_condition(engine, 0.8, 35000.0, throttles=(1.0,))
        self.assertEqual(len(rows), 1)
        assert_near_equal(rows[0]['Total Net Thrust (lbf, output)'], 20500.0, tolerance=1e-3)
        assert_near_equal(rows[0]['Centrifugal PR'], 1.5, tolerance=1e-3)

    def test_turboshaft_with_axial_and_centrifugal_stages(self):
        rows = deck.run_flight_condition('reference_9100hp', 0.8, 33000.0, throttles=(1.0,))
        self.assertEqual(len(rows), 1)
        power = rows[0]['Off-Design Core Shaft Power (hp, core output)']
        assert_near_equal(power, 9100.0, tolerance=1e-3)
        assert_near_equal(rows[0]['Axial HPC PR'], 7.421583 / (1.9 * 2.0), tolerance=1e-3)

    def test_turboshaft_with_only_a_centrifugal_compressor(self):
        rows = deck.run_flight_condition('TPE331-10', 0.4, 15000.0, throttles=(1.0, 0.975))
        self.assertEqual(len(rows), 2)
        power = rows[0]['Off-Design Core Shaft Power (hp, core output)']
        assert_near_equal(power, 940.0, tolerance=1e-3)
        assert_near_equal(rows[0]['HPC PR'], 12.78, tolerance=1e-3)
        self.assertNotIn('LPC PR', rows[0])
        self.assertNotIn('Axial HPC PR', rows[0])

    def test_map_points_set_the_design_point_on_the_map(self):
        cases = {
            'TPE331-10': ('DESIGN.hpc_centri.map', 'hpc_centri', 'turboshaft', 15000.0),
            'CF34-8E': ('DESIGN.fan.map', 'fan', 'turbofan', 35000.0),
        }
        for name, (path, element, architecture, alt) in cases.items():
            with self.subTest(name):
                engine = engines.get_engine(name)
                point = {element: {'RlineMap': 2.3, 'NcMap': 0.97}}
                engine = {**engine, 'model': {**engine['model'], 'map_points': point}}
                self.assertEqual(engine['architecture'], architecture)
                prob = deck.build_problem(engine, 0.4, alt)
                assert_near_equal(prob.get_val(f'{path}.RlineMap'), 2.3)
                assert_near_equal(prob.get_val(f'{path}.NcMap'), 0.97)

    def test_unknown_map_point_raises(self):
        engine = engines.get_engine('TPE331-10')
        for point, message in (
            ({'hpc': {'RlineMap': 2.0}}, 'hpc'),
            ({'hpc_centri': {'Rline': 2.0}}, 'Rline'),
        ):
            bad = {**engine, 'model': {**engine['model'], 'map_points': point}}
            with self.assertRaisesRegex(ValueError, message):
                deck.build_problem(bad, 0.4, 15000.0)

    def test_unknown_model_setting_raises(self):
        engine = {
            'architecture': 'turboshaft',
            'design': engines.TURBOSHAFTS['TPE331-10'],
            'model': {'design_altt': 15000.0},
        }
        with self.assertRaisesRegex(ValueError, 'design_altt'):
            deck.build_problem(engine, 0.4, 15000.0)

    def test_turbofan_model_needs_an_axial_hpc(self):
        design = engines.TURBOFANS['PW4168']
        opr = design['pi_fan'] * design['pi_lpc']
        engine = {'architecture': 'turbofan', 'design': {**design, 'opr': opr}}
        with self.assertRaisesRegex(ValueError, 'needs an axial HPC'):
            deck.build_problem(engine, 0.8, 35000.0)


if __name__ == '__main__':
    unittest.main()
