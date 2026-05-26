import pandas as pd

from saco import Dataset


DATA_FOLDER = './tests/data/01/base'


def get_dataset(data_folder=DATA_FOLDER):
    ds = Dataset(data_folder=data_folder)
    ds.load_data()
    return ds


def get_gwabs_data():
    ds = Dataset()

    lta_col = ds.gwabs.get_lta_column('FL')
    percentile_col = ds.gwabs.get_value_column('FL', 95)

    dc = {
        lta_col: [10.0], percentile_col: [11.4], ds.gwabs.impfac_column: 2.0,
        ds.gwabs.consumptiveness_column: 0.6,
    }
    df = pd.DataFrame(dc, index=['GW001'])

    return df, {'scenario': 'FL', 'percentile': 95}


def test_gwabs_infer_mean_abstraction():
    df, dc = get_gwabs_data()
    ds = get_dataset()

    ds.set_tables({'gwabs': df})

    ds.gwabs.infer_mean_abstraction(dc['scenario'], dc['percentile'])

    lta_col = ds.gwabs.get_lta_column(dc['scenario'])
    assert ds.gwabs.data[lta_col].to_numpy()[0] == df[lta_col].to_numpy()[0]


def test_gwabs_infer_percentile_impact():
    df, dc = get_gwabs_data()
    ds = get_dataset()

    ds.set_tables({'gwabs': df})

    ds.gwabs.infer_percentile_impact(dc['scenario'], dc['percentile'])

    value_col = ds.gwabs.get_value_column(dc['scenario'], dc['percentile'])
    assert ds.gwabs.data[value_col].to_numpy()[0] == df[value_col].to_numpy()[0]


def get_swabs_data():
    ds = Dataset()

    lta_col = ds.swabs.get_lta_column('FL')
    percentile_col = ds.swabs.get_value_column('FL', 95)

    dc = {
        lta_col: [10.0], percentile_col: [1.0], ds.swabs.start_month_column: 1,
        ds.swabs.end_month_column: 12, ds.swabs.consumptiveness_column: 0.4,
        ds.swabs.hof_value_column: 0.0,

    }
    df = pd.DataFrame(dc, index=['SW001'])

    return df, {'scenario': 'FL', 'percentile': 95, 'sfac': 0.25}


def test_swabs_infer_mean_abstraction():
    df, dc = get_swabs_data()
    ds = get_dataset()

    ds.sfac.data.loc[ds.sfac.data.index == '1&12', f'SFAC{dc['percentile']}'] = (
        dc['sfac']
    )

    ds.set_tables({'swabs': df})

    ds.swabs.infer_mean_abstraction(dc['scenario'], dc['percentile'], ds.sfac)

    lta_col = ds.swabs.get_lta_column(dc['scenario'])
    assert ds.swabs.data[lta_col].to_numpy()[0] == df[lta_col].to_numpy()[0]


def test_swabs_infer_percentile_impact():
    df, dc = get_swabs_data()
    ds = get_dataset()

    ds.sfac.data.loc[ds.sfac.data.index == '1&12', f'SFAC{dc['percentile']}'] = (
        dc['sfac']
    )

    ds.set_tables({'swabs': df})

    ds.swabs.infer_percentile_impact(dc['scenario'], dc['percentile'], ds.sfac)

    lta_col = ds.swabs.get_lta_column(dc['scenario'])
    assert ds.swabs.data[lta_col].to_numpy()[0] == df[lta_col].to_numpy()[0]
