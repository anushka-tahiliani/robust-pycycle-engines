"""Tabulated engine decks and a loader for them."""

import csv
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / 'data'

DECKS = {
    'AE3007': 'AE_3007_deck.csv',
    'CF34-8E': 'CF34_8E_deck.csv',
    'CFM56': 'CFM56_UPS_deck.csv',
    'GE90': 'GE90_deck.csv',
    'PW4168': 'PW_4168_deck.csv',
    'PW127': 'PW_127_deck.csv',
    'TPE331-10': 'TPE331-10_deck.csv',
}


def load_deck(name):
    """
    Load an engine deck.

    Parameters
    ----------
    name : str
        Key of `DECKS`.

    Returns
    -------
    dict
        Column values as float arrays, keyed by column header.
    """
    with open(DATA_DIR / DECKS[name], newline='') as file:
        reader = csv.reader(file)
        headers = [header.strip() for header in next(reader)]
        rows = np.array(list(reader), dtype=float)
    return {header: rows[:, i] for i, header in enumerate(headers)}
