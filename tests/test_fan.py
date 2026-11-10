import unittest

from openmdao.utils.assert_utils import assert_near_equal

from robust_pycycle_engines.initial_guess import fan


class TestFanInitialGuess(unittest.TestCase):
    def test_design_flow_closes_power_balance(self):
        W_lbm = fan.design_mass_flow(fan_power=5000.0, FPR=1.4, eta_pfan=0.9, Tamb=220.0, M=0.8)
        Tt0 = 220.0 * (1 + 0.2 * 0.8**2)
        Tt2 = Tt0 * 1.4 ** (0.4 / (0.9 * 1.4))
        power_hp = W_lbm / fan.KG_TO_LBM * fan.CP * (Tt2 - Tt0) / fan.HP_TO_W
        assert_near_equal(power_hp, 5000.0, 1e-12)

    def test_off_design_flow_is_self_consistent(self):
        W_lbm, FPR = fan.off_design_mass_flow(
            fan_power=5000.0, nozzle_area=1500.0, Tamb=220.0, Pamb=25000.0, M=0.8
        )
        self.assertGreater(W_lbm, 0.0)
        self.assertGreaterEqual(FPR, 1.05)

    def test_off_design_more_power_raises_pressure_ratio(self):
        kwargs = dict(nozzle_area=1500.0, Tamb=220.0, Pamb=25000.0, M=0.8)
        _, FPR_low = fan.off_design_mass_flow(fan_power=3000.0, **kwargs)
        _, FPR_high = fan.off_design_mass_flow(fan_power=8000.0, **kwargs)
        self.assertGreater(FPR_high, FPR_low)


if __name__ == '__main__':
    unittest.main()
