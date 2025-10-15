import pandas as pd
import pytest

from saco import Dataset, Calculator


CASE_IDS = ['01', '02']
CAPPING_VARIANTS = ['uncapped', 'capped_simple', 'capped_net']
CAPPING_ARGS = {
    'uncapped': None, 'capped_simple': 'simple', 'capped_net': 'cap-net-impacts',
}


def get_io(data_folder, ref_path):
    ds = Dataset(data_folder)
    ds.load_data()
    df_ref = pd.read_parquet(ref_path)
    return ds, df_ref


@pytest.mark.parametrize('case_id', CASE_IDS)
@pytest.mark.parametrize('capping_variant', CAPPING_VARIANTS)
def test_calculator_run(case_id, capping_variant, capping_args=None):
    if capping_args is None:
        capping_args = CAPPING_ARGS

    ds, df_ref = get_io(
        data_folder=f'./tests/data/{case_id}/base',
        ref_path=f'./tests/data/{case_id}/calculator/Master__{capping_variant}.parquet',
    )

    calculator = Calculator(ds, capping_method=capping_args[capping_variant])
    ds_test = calculator.run()

    pd.testing.assert_frame_equal(ds_test.mt.data, df_ref, check_like=True)
