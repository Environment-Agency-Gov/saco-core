import numpy as np

from saco import Dataset


DATA_FOLDER = './tests/data/01/base'


def test_infer_percentile_impact(data_folder=DATA_FOLDER):
    ds = Dataset(data_folder)
    ds.load_data()

    ds.gwabs.data['GWQ95FLWR__ORIGINAL'] = ds.gwabs.data['GWQ95FLWR']
    ds.swabs.data['SWQ95FLWR__ORIGINAL'] = ds.swabs.data['SWQ95FLWR']

    ds.infer_percentile_impact(scenarios=['FL'], percentiles=[95])

    assert np.all(np.abs(ds.gwabs.data['GWQ95FLWR'] - ds.gwabs.data['GWQ95FLWR__ORIGINAL']) < 1e-02)

    mask = ds.swabs.data['HOFMLD'] > 0.0
    df = ds.swabs.data.loc[~mask]
    df = df.loc[np.abs(df['SWQ95FLWR'] - df['SWQ95FLWR__ORIGINAL']) >= 1e-02]
    assert df.shape[0] == 0
