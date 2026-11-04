
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_processed_files_exist():
    train_path = PROJECT_ROOT / "data" / "processed" / "train.csv"
    test_path = PROJECT_ROOT / "data" / "processed" / "test.csv"

    assert train_path.exists()
    assert test_path.exists()


def test_processed_data_is_not_empty():
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

    assert len(train) > 0
    assert len(test) > 0


def test_target_column_exists():
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

    assert "Churn" in train.columns
    assert "Churn" in test.columns


def test_processed_data_has_no_missing_values():
    train = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "train.csv")
    test = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "test.csv")

    assert not train.isnull().any().any()
    assert not test.isnull().any().any() 
