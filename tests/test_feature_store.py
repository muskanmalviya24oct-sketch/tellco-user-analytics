import pandas as pd
from telco_analytics.feature_store import FeatureStore


def test_save_and_load_roundtrip(tmp_path):
    fs = FeatureStore(str(tmp_path))
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
    fs.save('sample', df, description='a test table')
    loaded = fs.load('sample')
    pd.testing.assert_frame_equal(df, loaded)


def test_list_shows_saved_table(tmp_path):
    fs = FeatureStore(str(tmp_path))
    df = pd.DataFrame({'a': [1, 2]})
    fs.save('sample', df, description='desc')
    listing = fs.list()
    assert 'sample' in listing['name'].values
    assert listing.loc[listing['name'] == 'sample', 'rows'].iloc[0] == 2


def test_exists_and_delete(tmp_path):
    fs = FeatureStore(str(tmp_path))
    df = pd.DataFrame({'a': [1]})
    fs.save('sample', df)
    assert fs.exists('sample')
    fs.delete('sample')
    assert not fs.exists('sample')


def test_load_missing_table_raises(tmp_path):
    fs = FeatureStore(str(tmp_path))
    try:
        fs.load('does_not_exist')
        assert False, 'expected KeyError'
    except KeyError:
        pass


def test_save_preserves_index_with_name(tmp_path):
    fs = FeatureStore(str(tmp_path))
    df = pd.DataFrame({'score': [1.5, 2.5]}, index=pd.Index([10, 20], name='MSISDN'))
    fs.save('indexed', df)
    loaded = fs.load('indexed')
    assert loaded.index.name == 'MSISDN'
    assert list(loaded.index) == [10, 20]
