import unittest

import numpy as np

from robust_pycycle_engines.decks import DECKS, load_deck


class TestDecks(unittest.TestCase):
    def test_decks_load_with_flight_condition_inputs(self):
        for name in DECKS:
            with self.subTest(deck=name):
                deck = load_deck(name)
                mach = deck['Mach Number (input)']
                self.assertGreater(mach.size, 0)
                self.assertTrue(np.all(mach >= 0.0))
                self.assertTrue(np.all(deck['Altitude (ft, input)'] >= 0.0))
                throttle = deck['Throttle (input)']
                self.assertTrue(np.all((throttle > 0.0) & (throttle <= 1.0)))

    def test_no_missing_values(self):
        for name in DECKS:
            with self.subTest(deck=name):
                for values in load_deck(name).values():
                    self.assertFalse(np.any(np.isnan(values)))


if __name__ == '__main__':
    unittest.main()
