
import logging
from pathlib import Path 

import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data.preprocess import (
    build_preprocessor,
    TARGET_COLUMN
)

from src.models.evaluate import evaluate_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.csv"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.csv"

MODEL_DIR = PROJECT_ROOT / "artifacts" / "model"
MODEL_PATH = MODEL_DIR / "model.joblib"
METRICS_DIR = PROJECT_ROOT / "metrics"
METRICS_PATH = METRICS_DIR / "evaluation.json"

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_TRACKING_URI = f"sqlite:///{MLFLOW_DB_PATH.as_posix()}"

RANDOM_STATE = 42


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def load_datasets():
    """Load the processed training and testing datasets."""

    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError(
            "Processed train.csv or test.csv not found. "
            "Run preprocessing first."
        )

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN].astype(int)

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN].astype(int)

    logger.info("Training data loaded: %s", X_train.shape)
    logger.info("Testing data loaded: %s", X_test.shape)

    return X_train, X_test, y_train, y_test


def get_models():
    """Return the three classification algorithms."""

    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=6,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


def train_and_compare(X_train, y_train):
    """Train and compare models using validation F1-score."""

    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train
    )

    models = get_models()
    results = []

    best_score = -1
    best_name = None
    best_estimator = None

    for name, estimator in models.items():

        pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", estimator)
            ]
        )

        with mlflow.start_run(run_name=name):

            pipeline.fit(X_fit, y_fit)

            predictions = pipeline.predict(X_val)
            probabilities = pipeline.predict_proba(X_val)[:, 1]

            from sklearn.metrics import (
                accuracy_score,
                precision_score,
                recall_score,
                f1_score,
                roc_auc_score
            )

            metrics = {
                "accuracy": accuracy_score(y_val, predictions),
                "precision": precision_score(
                    y_val, predictions, zero_division=0
                ),
                "recall": recall_score(
                    y_val, predictions, zero_division=0
                ),
                "f1_score": f1_score(
                    y_val, predictions, zero_division=0
                ),
                "roc_auc": roc_auc_score(
                    y_val, probabilities
                )
            }

            mlflow.log_param("model_name", name)
            mlflow.log_param("validation_size", 0.20)
            mlflow.log_param("random_state", RANDOM_STATE)

            mlflow.log_params(estimator.get_params())
            mlflow.log_metrics(metrics)

            logger.info("%s: %s", name, metrics)

            results.append({
                "Model": name,
                **metrics
            })

            if metrics["f1_score"] > best_score:
                best_score = metrics["f1_score"]
                best_name = name
                best_estimator = estimator

    results_df = pd.DataFrame(results)

    print("\nModel Comparison:")
    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    logger.info("Best model: %s", best_name)
    logger.info("Best validation F1-score: %.4f", best_score)

    return best_name, best_estimator, results_df


def train_best_model(
    best_name,
    best_estimator,
    X_train,
    y_train,
    X_test,
    y_test
):
    """Refit the best complete pipeline and evaluate on test data."""

    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", best_estimator)
        ]
    )

    with mlflow.start_run(run_name=f"Best_{best_name}"):

        best_pipeline.fit(X_train, y_train)

        test_metrics = evaluate_model(
            best_pipeline,
            X_test,
            y_test
        )
        METRICS_DIR.mkdir(parents=True, exist_ok=True)

        with open(METRICS_PATH, "w") as f:
            json.dump(test_metrics, f, indent=4)

        logging.info("Evaluation metrics saved: %s", METRICS_PATH)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(best_pipeline, MODEL_PATH)

        mlflow.log_param("best_model", best_name)
        mlflow.log_param("selection_metric", "validation_f1_score")
        mlflow.log_metrics({
            f"test_{key}": value
            for key, value in test_metrics.items()
        })

        mlflow.sklearn.log_model(
        sk_model=best_pipeline,
        name="customer-churn-model",
        skops_trusted_types=["numpy.dtype"]
    )

    return best_pipeline, best_name


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Customer_Churn_Prediction")

    X_train, X_test, y_train, y_test = load_datasets()

    best_pipeline, best_name, comparison_df = train_and_compare(
    X_train, y_train
   )

    train_best_model(
        best_pipeline,
        best_name,
        X_train,
        y_train,
        X_test,
        y_test
    )


if __name__ == "__main__":
    main() 