import itertools

import numpy as np
import pandas as pd
import pytest

from saco import Dataset, Optimiser


# Capping refers (here) to whether or not impacts are capped to their feasible
# components in cases where flow targets are zero. (If uncapped, an impact may be only
# partially realisable.)
CASE_IDS = ['01', '02']
CAPPING_VARIANTS = ['uncapped', 'capped']
TEST_TYPES = {
    '01': {'uncapped': 'basic', 'capped': 'full'},
    '02': {'uncapped': 'basic', 'capped': 'basic'},
}



@pytest.mark.parametrize('case_id', CASE_IDS)
@pytest.mark.parametrize('capping_variant', CAPPING_VARIANTS)
@pytest.mark.filterwarnings('ignore:Some flow targets')
def test_optimiser_run(case_id, capping_variant, test_types=None):
    if test_types is None:
        test_types = TEST_TYPES

    df_ref = pd.read_csv('./tests/data/optimiser_reference.csv', dtype={'Case_ID': str})

    ds = Dataset(data_folder=f'./tests/data/{case_id}/base')
    ds.load_data()

    ds.set_flow_targets()
    ds.set_optimise_flag(exclude_deregulated=False, exclude_below=None)

    if capping_variant == 'uncapped':
        uncapped_zero_targets = True
    else:
        uncapped_zero_targets = False

    optimiser = Optimiser(ds, uncapped_zero_targets=uncapped_zero_targets)
    ds_test = optimiser.run()

    # Basic checks on objectives
    for scenario, percentile in itertools.product(ds.scenarios, ds.percentiles):
        metrics = optimiser.models[(scenario, percentile, 'max-point-equality')].metrics
        test_abstraction = metrics.domain_total_abstraction
        test_mad = metrics.domain_point_mad

        ref_abstraction = df_ref.loc[
            (df_ref['Scenario'] == scenario) & (df_ref['Percentile'] == percentile)
            & (df_ref['Case_ID'] == case_id)
            & (df_ref['Capping_Variant'] == capping_variant),
            'Domain_Total_Abstraction'
        ].to_numpy()[0]
        ref_mad = df_ref.loc[
            (df_ref['Scenario'] == scenario) & (df_ref['Percentile'] == percentile)
            & (df_ref['Case_ID'] == case_id)
            & (df_ref['Capping_Variant'] == capping_variant),
            'Domain_Point_MAD'
        ].to_numpy()[0]
        ref_count = df_ref.loc[
            (df_ref['Scenario'] == scenario) & (df_ref['Percentile'] == percentile)
            & (df_ref['Case_ID'] == case_id)
            & (df_ref['Capping_Variant'] == capping_variant),
            'Non_Compliance_Count'
        ].to_numpy()[0]

        assert test_abstraction >= (ref_abstraction - 1e-02)
        assert test_mad <= (ref_mad + 1e-02)

        test_count = np.sum(ds_test.mt.data[f'COMP{scenario}Q{percentile}'] > 0)
        assert test_count == ref_count

    # Row- and column-wise checks on full outputs
    if test_types[case_id][capping_variant] == 'full':
        df_ref_master = pd.read_parquet(f'./tests/data/{case_id}/optimiser/Master.parquet')
        pd.testing.assert_frame_equal(ds_test.mt.data, df_ref_master, check_like=True)

        df_ref_gwabs = pd.read_parquet(f'./tests/data/{case_id}/optimiser/GWABs_NBB.parquet')
        pd.testing.assert_frame_equal(ds_test.gwabs.data, df_ref_gwabs, check_like=True)

        df_ref_swabs = pd.read_parquet(f'./tests/data/{case_id}/optimiser/SWABS_NBB.parquet')
        pd.testing.assert_frame_equal(ds_test.swabs.data, df_ref_swabs, check_like=True)
