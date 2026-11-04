
import logging
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "dataset.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "model"

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

NUMERICAL_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

TEST_SIZE = 0.20
RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_data():
    """Load the original customer churn dataset."""

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {RAW_DATA_PATH}"
        )

    df = pd.read_csv(RAW_DATA_PATH)

    logger.info("Dataset loaded: %s", df.shape)

    return df


def clean_data(df):
    """Clean data and prepare the target variable."""

    df = df.copy()

    if TARGET_COLUMN not in df.columns:
        raise ValueError("Churn target column is missing.")

    if ID_COLUMN not in df.columns:
        raise ValueError("customerID column is missing.")

    df = df.drop_duplicates()

    df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype(str).str.strip(),
    errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df = df.drop(columns=[ID_COLUMN])

    df[TARGET_COLUMN] = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .map({"No": 0, "Yes": 1})
    )

    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Invalid or missing churn target values.")

    logger.info("Data cleaning completed: %s", df.shape)

    return df


def split_data(df):
    """Split features and target into training and testing sets."""

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    logger.info("Training records: %s", len(X_train))
    logger.info("Testing records: %s", len(X_test))

    return X_train, X_test, y_train, y_test


def build_preprocessor():
    """Create numerical and categorical preprocessing pipelines."""

    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_features = [
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ]

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                numerical_pipeline,
                NUMERICAL_FEATURES
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )

    return preprocessor


def save_processed_data(
    X_train,
    X_test,
    y_train,
    y_test
):
    """Save cleaned training and testing datasets."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df[TARGET_COLUMN] = y_train

    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = y_test

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    logger.info("Processed datasets saved successfully.")


def main():
    """Execute the complete preprocessing workflow."""

    df = load_data()

    df = clean_data(df)

    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor()

    save_processed_data(
        X_train,
        X_test,
        y_train,
        y_test
    )

    logger.info("Preprocessing pipeline created successfully.")

    logger.info("Preprocessing completed successfully.")


if __name__ == "__main__":
    main()
