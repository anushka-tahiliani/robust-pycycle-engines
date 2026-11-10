"""
Compressor maps with the design points used to build the engine decks.

pyCycle places each compressor's design point on its map at `MapData.defaults`. The decks were
built with the fan and LPC design points moved from pyCycle's, so these copies override those
defaults and leave pyCycle itself untouched; an engine's `map_points` setting moves them further.
pyCycle has no centrifugal map, so centrifugal stages use the LPC map.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import copy

import pycycle.api as pyc


def with_design_point(map_data, **defaults):
    """
    Copy a pyCycle compressor map with a different design point.

    Parameters
    ----------
    map_data : pycycle.maps.map_data.MapData
        Map to copy.
    **defaults : float
        Design point values to override, e.g. `RlineMap` or `NcMap`.

    Returns
    -------
    pycycle.maps.map_data.MapData
        Copy of `map_data` sharing its tables.
    """
    new = copy.copy(map_data)
    new.defaults = {**map_data.defaults, **defaults}
    return new


def maps_at_design_points(default_maps, map_points):
    """
    Copy a cycle's compressor maps with an engine's design points on them.

    Parameters
    ----------
    default_maps : dict
        Default map of each compressor, keyed by element name.
    map_points : dict
        `RlineMap` and `NcMap` values to override, keyed by element name, e.g.
        `{'hpc_centri': {'RlineMap': 2.15}}`.

    Returns
    -------
    dict
        Map of each compressor, keyed by element name.
    """
    unknown = set(map_points) - set(default_maps)
    if unknown:
        raise ValueError(f'Unknown compressors {sorted(unknown)}; use {sorted(default_maps)}.')
    for element, point in map_points.items():
        unknown = set(point) - {'RlineMap', 'NcMap'}
        if unknown:
            raise ValueError(f'{element}: {sorted(unknown)} not settable; use RlineMap or NcMap.')
    return {
        element: with_design_point(map_data, **map_points.get(element, {}))
        for element, map_data in default_maps.items()
    }


FAN_MAP = with_design_point(pyc.FanMap, RlineMap=1.8)
LPC_MAP = with_design_point(pyc.LPCMap, PR=1.8, RlineMap=1.6)
HPC_MAP = pyc.HPCMap
