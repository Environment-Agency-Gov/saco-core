import os
from typing import Union, List, Dict
from pathlib import Path

import numpy as np
import pandas as pd


def check_if_output_path_exists(output_path: Union[Path, str], overwrite: bool):
    if (not overwrite) and os.path.exists(output_path):
        raise ValueError(
            f'Cannot overwrite existing file if overwrite arg is False: {output_path}'
        )


def check_column_uniqueness(
        df: pd.DataFrame, column_names: Union[str, List[str]],
) -> bool:
    """
    Check whether a column or group of columns is unique.

    For assessing whether a column(s) can form an appropriate dataframe index.

    Args:
        df: Dataframe to check.
        column_names: Primary key/index column(s) as str or list of str.

    Returns:
        Indication of whether index columns are unique.

    """
    if isinstance(column_names, str):
        if df[column_names].shape[0] == df[column_names].unique().shape[0]:
            is_unique = True
        else:
            is_unique = False
    elif isinstance(column_names, list):  # and (len(index_names) == 2)
        df1 = df[column_names].value_counts().reset_index(name='id_group_count')
        if df1['id_group_count'].max() == 1:
            is_unique = True
        else:
            is_unique = False
    else:
        raise ValueError(
            'Unable to parse column_names: must be either str or list of str.'
        )

    return is_unique


def check_indexes_match(dfs: Dict[str, pd.DataFrame], ref_table: str) -> bool:
    """
    Check whether indexes match for a set of dataframes.

    Args:
        dfs: Keys are table names (str) and values are pd.DataFrame.
        ref_table: Table to use reference to compare other tables against.

    Returns:
        Indicates whether indexes are OK (i.e. all match ref_table).

    """
    ref_index = dfs[ref_table].index
    tables_to_check = [item for item in dfs.keys() if item != ref_table]

    indexes_match = True
    for table_name in tables_to_check:
        df = dfs[table_name]

        if np.all(df.index == ref_index):
            pass
        else:
            indexes_match = False
            break

    return indexes_match


def infill_cols(df: pd.DataFrame, dc: Dict) -> pd.DataFrame:
    """
    Infill dataframe columns where they contain nan/inf.

    Args:
        df: Dataframe with columns to infill.
        dc: Dictionary with infill values as keys and lists of columns to infill as
            values.

    Returns:
        Infilled dataframe.

    """
    df = df.copy()
    for infill_value, columns in dc.items():
        for column in columns:
            if column in df.columns:
                df.loc[df[column].isna(), column] = infill_value
    return df
