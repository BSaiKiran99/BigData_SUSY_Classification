from pathlib import Path
import time
import json

import pandas as pd
from pyspark.sql import SparkSession, functions as F

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
)

def make_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("SUSY_Sklearn_Baseline_8GB")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )

def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed" / "susy_parquet"
    out_dir = root / "reports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    spark = make_spark()

    df = spark.read.parquet(str(data_path)).withColumn("label", F.col("label").cast("int"))
    feature_cols = [c for c in df.columns if c != "label"]

    N = 300_000

    sample_df = df.orderBy(F.rand(seed=42)).limit(N)

    pdf = sample_df.toPandas()
    spark.stop()

    X = pdf[feature_cols].values
    y = pdf["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    results = []
    run_id = time.strftime("%Y%m%d_%H%M%S")

    lr_pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1))
    ])

    t0 = time.time()
    lr_pipe.fit(X_train, y_train)
    train_time = time.time() - t0

    probs = lr_pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    lr_row = {
        "run_id": run_id,
        "model": "sklearn_logreg",
        "n_rows": int(N),
        "train_seconds": float(train_time),
        "auc_roc": float(roc_auc_score(y_test, probs)),
        "auc_pr": float(average_precision_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds)),
        "confusion_matrix": json.dumps(confusion_matrix(y_test, preds).tolist()),
    }
    results.append(lr_row)

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )

    t0 = time.time()
    rf.fit(X_train, y_train)
    train_time = time.time() - t0

    probs = rf.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    rf_row = {
        "run_id": run_id,
        "model": "sklearn_random_forest",
        "n_rows": int(N),
        "train_seconds": float(train_time),
        "auc_roc": float(roc_auc_score(y_test, probs)),
        "auc_pr": float(average_precision_score(y_test, probs)),
        "f1": float(f1_score(y_test, preds)),
        "confusion_matrix": json.dumps(confusion_matrix(y_test, preds).tolist()),
    }
    results.append(rf_row)

    out_csv = out_dir / f"sklearn_baseline_{run_id}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f" Saved scikit-learn baseline results to: {out_csv}")

    print(pd.DataFrame(results))

if __name__ == "__main__":
    main()
