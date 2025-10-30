import pandas as pd
import pytest

from saco import Dataset, Optimiser


CASE_IDS = ['01', '02']


@pytest.mark.parametrize('case_id', CASE_IDS)
@pytest.mark.filterwarnings('ignore:Some flow targets')
def test_optimiser_run(case_id):
    ds = Dataset(data_folder=f'./tests/data/{case_id}/base')
    ds.load_data()

    ds.set_flow_targets()
    ds.set_optimise_flag(exclude_deregulated=False, exclude_below=None)

    optimiser = Optimiser(ds)
    ds_test = optimiser.run()

    df_ref_master = pd.read_parquet(f'./tests/data/{case_id}/optimiser/Master.parquet')
    pd.testing.assert_frame_equal(ds_test.mt.data, df_ref_master, check_like=True)

    df_ref_gwabs = pd.read_parquet(f'./tests/data/{case_id}/optimiser/GWABs_NBB.parquet')
    pd.testing.assert_frame_equal(ds_test.gwabs.data, df_ref_gwabs, check_like=True)

    df_ref_swabs = pd.read_parquet(f'./tests/data/{case_id}/optimiser/SWABS_NBB.parquet')
    pd.testing.assert_frame_equal(ds_test.swabs.data, df_ref_swabs, check_like=True)
