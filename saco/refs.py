"""
Reference flow (typically EFI) calculation via WRGIS method.

"""
from copy import deepcopy

import pandas as pd

from .dataset import Dataset


def derive_reference_flows(ds: Dataset) -> pd.DataFrame:
    """
    Calculate reference flows based on (relative) deviation from natural flows.

    Deviations are calculated using abstraction sensitivity bands (ASBs). The fractional
    deviations associated with each flow percentile for each ASB are given in the
    ASBPercentages table. ASBs per waterbody are given in AbsSensBands_NBB. This is how
    the EFI is calculated in WRGIS.

    The calculation logic is now part of the Dataset class. This function is retained
    for (overall) backwards compatibility.

    Args:
        ds: Input Dataset.

    Returns:
        Dataframe containing reference flows for requested percentiles. Columns are
        named following ``f'REFSQ{percentile}'``.

    """
    ds = deepcopy(ds)
    ds.set_reference_flows()
    return ds.refs.data.copy()
