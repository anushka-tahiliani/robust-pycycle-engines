import contextlib
import importlib.util
import io
import unittest
from pathlib import Path
from unittest import mock

from openmdao.utils.assert_utils import assert_near_equal

from robust_pycycle_engines import deck, engines
from robust_pycycle_engines.cycles.turbofan import COMPRESSOR_MAPS

EXAMPLE = Path(__file__).parent.parent / 'examples' / 'cfm56_turbofan.py'


def load_example():
    spec = importlib.util.spec_from_file_location('cfm56_turbofan', EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCFM56Example(unittest.TestCase):
    def test_matches_the_engine_table(self):
        example = load_example()
        self.assertEqual(example.ENGINE['design'], engines.TURBOFANS['CFM56'])
        model = dict(example.ENGINE['model'])
        for element, point in model.pop('map_points').items():
            self.assertEqual(point['RlineMap'], COMPRESSOR_MAPS[element].defaults['RlineMap'])
        self.assertEqual(model, engines.TURBOFAN_MODELS['CFM56'])

    def test_cruise_performance(self):
        example = load_example()
        rows = example.cruise_performance()
        self.assertEqual([row['Throttle (input)'] for row in rows], list(example.CRUISE_THROTTLES))
        assert_near_equal(rows[0]['Total Net Thrust (lbf, output)'], 5900.0, tolerance=1e-3)

    def test_chosen_flight_condition_and_throttles(self):
        rows = load_example().cruise_performance((0.7, 30000.0), (1.0, 0.975))
        self.assertEqual([row['Throttle (input)'] for row in rows], [1.0, 0.975])
        self.assertEqual(rows[0]['Mach Number (input)'], 0.7)
        assert_near_equal(rows[0]['Altitude (ft, input)'], 30000.0, tolerance=1e-9)

    def test_deck_uses_chosen_flight_conditions_and_throttles(self):
        example = load_example()
        with mock.patch.object(deck, 'generate_deck', return_value=[]) as generate:
            with contextlib.redirect_stdout(io.StringIO()):
                example.write_deck()
                example.write_deck([(0.5, 20000.0)], (1.0, 0.9), path='other.csv')
        first, second = generate.call_args_list
        self.assertEqual(first.args[1], example.DECK_FLIGHT_CONDITIONS)
        self.assertEqual(first.kwargs['throttles'], tuple(example.DECK_THROTTLES))
        self.assertEqual(first.kwargs['path'], example.DECK_PATH)
        self.assertEqual(second.args[1], [(0.5, 20000.0)])
        self.assertEqual(second.kwargs['throttles'], (1.0, 0.9))
        self.assertEqual(second.kwargs['path'], 'other.csv')


class TestCommandLine(unittest.TestCase):
    def run_main(self, *argv):
        with mock.patch.object(deck, 'generate_deck', return_value=[{}]) as generate:
            with contextlib.redirect_stdout(io.StringIO()):
                deck.main(list(argv))
        return generate.call_args

    def test_chosen_points_and_throttles(self):
        call = self.run_main(
            'PW4168', 'out.csv', '--point', '0.8', '35000', '--point', '0.3', '5000',
            '--throttles', '1', '0.95', '0.9',
        )  # fmt: skip
        self.assertEqual(call.args[:3], ('PW4168', [[0.8, 35000.0], [0.3, 5000.0]], 'out.csv'))
        self.assertEqual(call.kwargs['throttles'], (1.0, 0.95, 0.9))

    def test_default_throttles(self):
        call = self.run_main('PW127', 'out.csv', '--point', '0.2', '0')
        self.assertEqual(call.kwargs['throttles'], tuple(deck.THROTTLES))
