import pandas as pd
import pytest
from src import DataIngestion

def test_load_data_success(test_csv_path):
    loader = DataIngestion(path=test_csv_path)
    df = loader.load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert set(["query", "response", "intent", "domain"]).issubset(df.columns)

def test_load_data_missing_file(tmp_path):
    fake_path = tmp_path / "missing.csv"
    loader = DataIngestion(path=str(fake_path))
    with pytest.raises(FileNotFoundError):
        loader.load_data()
