import unittest

import numpy as np
from openmdao.utils.assert_utils import assert_near_equal

from robust_pycycle_engines import engines
from robust_pycycle_engines.initial_guess import balances


class TestTurbofanBalances(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cycle = engines.initial_guess('PW4168')
        cls.flight = dict(M=0.8, Tamb=218.8, Pamb=23840.0)

    def test_design_guess_units(self):
        guess = balances.turbofan_design_guess(self.cycle)
        assert_near_equal(guess['balance.W'], self.cycle['W_total'] * 2.20462, 1e-12)
        self.assertNotIn('balance.gb_trq', guess)

    def test_gearbox_torque_from_fan_power(self):
        guess = balances.turbofan_design_guess(self.cycle, geared=True)
        omega = 2 * np.pi * self.cycle['N_fan'] / 60
        power = guess['balance.gb_trq'] / 0.7375621 * omega
        assert_near_equal(power, self.cycle['Wdot_fan'], 1e-12)

    def test_gear_ratio_matches_pycycle_convention(self):
        # pyCycle's Gearbox defines gear_ratio = N_out / N_in, so the reducing fan drive is below 1
        ratio = balances.gear_ratio(self.cycle)
        assert_near_equal(ratio, self.cycle['N_fan'] / self.cycle['LP_Nmech_max'], 1e-12)
        self.assertLess(ratio, 1.0)

    def test_full_throttle_at_design_condition_recovers_design(self):
        T4 = self.cycle['Tt4']
        guess = balances.turbofan_off_design_guess(
            self.cycle, T4_max=T4, fan_area=1.0, geared=True, **self.flight
        )
        assert_near_equal(guess['balance.FAR'], self.cycle['FAR'], 1e-12)
        assert_near_equal(guess['balance.BPR'], self.cycle['BPR'], 1e-12)
        assert_near_equal(guess['balance.HP_Nmech'], self.cycle['HP_Nmech_max'], 1e-12)
        assert_near_equal(guess['balance.LP_Nmech'], self.cycle['LP_Nmech_max'], 1e-12)
        assert_near_equal(guess['balance.FAN_Nmech'], self.cycle['N_fan'], 1e-12)

    def test_part_throttle_follows_paper_scaling(self):
        tau = 0.8
        guess = balances.turbofan_off_design_guess(
            self.cycle, T4_max=tau * self.cycle['Tt4'], fan_area=1.0, **self.flight
        )
        assert_near_equal(guess['balance.FAR'], self.cycle['FAR'] * tau, 1e-12)  # Eq. (10)
        expected_BPR = self.cycle['BPR'] * (1 + 0.5 * (1 - tau))  # Eq. (11)
        assert_near_equal(guess['balance.BPR'], expected_BPR, 1e-12)
        assert_near_equal(guess['balance.HP_Nmech'], self.cycle['HP_Nmech_max'] * tau, 1e-12)
        # Direct drive: the LP shaft runs at the fan speed and there is no separate fan balance
        assert_near_equal(guess['balance.LP_Nmech'], self.cycle['N_fan'] * tau, 1e-12)
        self.assertNotIn('balance.FAN_Nmech', guess)

    def test_converged_design_fan_speed_overrides_guess(self):
        guess = balances.turbofan_off_design_guess(
            self.cycle, T4_max=self.cycle['Tt4'], fan_area=1.0, N_fan_des=2000.0, **self.flight
        )
        assert_near_equal(guess['balance.LP_Nmech'], 2000.0, 1e-12)

    def test_off_design_flow_scales_with_area_and_pressure(self):
        kwargs = dict(T4_max=self.cycle['Tt4'], M=0.8, Tamb=218.8)
        base = balances.turbofan_off_design_guess(self.cycle, fan_area=1.0, Pamb=23840.0, **kwargs)
        double = balances.turbofan_off_design_guess(
            self.cycle, fan_area=2.0, Pamb=47680.0, **kwargs
        )
        assert_near_equal(double['balance.W'], 4 * base['balance.W'], 1e-12)  # Eq. (7)


class TestTurboshaftBalances(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cycle = engines.initial_guess('reference_9100hp')

    def test_design_guess(self):
        guess = balances.turboshaft_design_guess(self.cycle)
        assert_near_equal(guess['balance.pt_PR'], self.cycle['pi_pt'], 1e-12)

    def test_flow_bounded_by_design_at_altitude(self):
        guess = balances.turboshaft_off_design_guess(
            self.cycle, M=0.8, Tamb=216.65, Pamb=18000.0, T4_target=self.cycle['Tt4']
        )
        assert_near_equal(guess['balance.W'], self.cycle['W_lbs'], 1e-12)

    def test_flow_scales_at_sea_level(self):
        guess = balances.turboshaft_off_design_guess(
            self.cycle, M=0.2, Tamb=288.15, Pamb=101325.0, T4_target=self.cycle['Tt4']
        )
        self.assertGreater(guess['balance.W'], self.cycle['W_lbs'])


if __name__ == '__main__':
    unittest.main()
