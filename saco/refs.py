"""
Reference flow (typically EFI) calculation via WRGIS method.

"""
from typing import List

import pandas as pd

from .config import Constants
from .dataset import Dataset


def derive_reference_flows(
        ds: Dataset, percentiles: List[int] = None, constants: Constants = None,
) -> pd.DataFrame:
    """
    Calculate reference flows based on (relative) deviation from natural flows.

    Deviations are calculated using abstraction sensitivity bands (ASBs). The fractional
    deviations associated with each flow percentile for each ASB are given in the
    ASBPercentages table. ASBs per waterbody are given in AbsSensBands_NBB. This is how
    the EFI is calculated in WRGIS.

    Args:
        ds: Input Dataset.
        percentiles: Flow percentiles (natural).
        constants: Global constants defined by default in config.Constants.

    Returns:
        Dataframe containing reference flows for requested percentiles. Columns are
        named following ``f'REFSQ{percentile}'``.

    """
    if constants is None:
        constants = Constants()
    if percentiles is None:
        percentiles = constants.valid_percentiles

    asb_col = ds.asbs.asb_column
    perc_col = ds.asb_percs.percent_column

    qnat_cols = [
        ds.qnat.get_value_column(p, constants.ups_abb) for p in percentiles
    ]
    df = pd.concat([ds.qnat.data[qnat_cols], ds.asbs.data], axis=1)

    dfs = []
    ref_cols = []
    for p in percentiles:
        p_label = ds.asb_percs.percentile_label(p)
        percs = ds.asb_percs.data.loc[
            ds.asb_percs.data.index.get_level_values(0) == p_label
        ].reset_index()
        percs = percs.drop(columns=ds.asb_percs.index_name[0])
        percs = percs.rename(columns={ds.asb_percs.index_name[1]: asb_col})

        qnat_col = ds.qnat.get_value_column(p, constants.ups_abb)
        ref_col = ds.refs.get_value_column(p)

        df1 = df[[qnat_col, asb_col]].reset_index().merge(percs, how='left', on=asb_col)
        df1 = df1.set_index(constants.waterbody_id_column)
        df1.loc[df1[perc_col].isna(), perc_col] = 1.0

        df1[ref_col] = df1[qnat_col] - (df1[qnat_col] * df1[perc_col])

        dfs.append(df1[[ref_col]])
        ref_cols.append(ref_col)

    df2 = pd.concat(dfs, axis=1)
    df2 = df2[ref_cols]

    return df2
