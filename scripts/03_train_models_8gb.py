from pathlib import Path
import json
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, LinearSVC
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


def make_spark() -> SparkSession:

    return (
        SparkSession.builder
        .appName("SUSY_Train_Models_8GB")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024))  # 64MB
        .getOrCreate()
    )


def evaluate_predictions(pred_df, label_col="label", prob_col="probability", raw_col="rawPrediction"):

    roc_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_col, metricName="areaUnderROC")
    pr_eval = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol=raw_col, metricName="areaUnderPR")

    metrics = {
        "auc_roc": float(roc_eval.evaluate(pred_df)),
        "auc_pr": float(pr_eval.evaluate(pred_df)),
    }

    f1_eval = MulticlassClassificationEvaluator(labelCol=label_col, predictionCol="prediction", metricName="f1")
    metrics["f1"] = float(f1_eval.evaluate(pred_df))

    cm = (
        pred_df.select(
            F.col(label_col).cast("int").alias("y"),
            F.col("prediction").cast("int").alias("yhat"),
        )
        .groupBy("y", "yhat").count()
        .toPandas()
    )
    metrics["confusion_matrix_counts"] = cm.to_dict(orient="records")
    return metrics


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed" / "susy_parquet"
    model_dir = root / "models"
    report_dir = root / "reports" / "metrics"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    spark = make_spark()

    df = spark.read.parquet(str(data_path))

    df = df.withColumn("label", F.col("label").cast("double"))

    feature_cols = [c for c in df.columns if c != "label"]

    SAMPLE_FRACTION = 0.30 
    df = df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=42)

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    train = train.repartition(64).cache()
    test = test.repartition(64).cache()
    train.count() 
    test.count()

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")

    scaler = StandardScaler(inputCol="features_vec", outputCol="features", withStd=True, withMean=False)


    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=30, regParam=0.01, elasticNetParam=0.0)
    rf = RandomForestClassifier(featuresCol="features_vec", labelCol="label", numTrees=80, maxDepth=8, featureSubsetStrategy="auto")
    gbt = GBTClassifier(featuresCol="features_vec", labelCol="label", maxIter=50, maxDepth=5)
    svm = LinearSVC(featuresCol="features", labelCol="label", maxIter=30, regParam=0.1)

    lr_pipe = Pipeline(stages=[assembler, scaler, lr])
    svm_pipe = Pipeline(stages=[assembler, scaler, svm])

    rf_pipe = Pipeline(stages=[assembler, rf])
    gbt_pipe = Pipeline(stages=[assembler, gbt])

    experiments = [
        ("logreg", lr_pipe, True),
        ("random_forest", rf_pipe, False),
        ("gbt", gbt_pipe, False),
        ("linear_svm", svm_pipe, True),
    ]

    all_results = []
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    for name, pipe, uses_scaled in experiments:
        print(f"\n==============================")
        print(f"Training: {name} | scaled={uses_scaled}")
        print(f"==============================")

        model = pipe.fit(train)
        preds = model.transform(test).select("label", "prediction", "rawPrediction", *([ "probability" ] if "probability" in model.transform(test).columns else []))

        metrics = evaluate_predictions(preds)


        out_model_path = model_dir / f"{name}_{run_id}"
        model.write().overwrite().save(str(out_model_path))
        print(f" Saved model to: {out_model_path}")

        result_row = {
            "run_id": run_id,
            "model": name,
            "sample_fraction": SAMPLE_FRACTION,
            **metrics,
        }
        all_results.append(result_row)

        with open(report_dir / f"{name}_{run_id}.json", "w") as f:
            json.dump(result_row, f, indent=2)
        print(f" Saved metrics JSON to: {report_dir}/{name}_{run_id}.json")


    results_df = spark.createDataFrame(all_results)


results_df = results_df.withColumn(
    "confusion_matrix_counts_json",
    F.to_json(F.col("confusion_matrix_counts"))
).drop("confusion_matrix_counts")

results_csv_path = report_dir / f"summary_{run_id}"
results_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(results_csv_path))
print(f"\nSaved CSV summary to folder: {results_csv_path}")    print(f"\n✅ Saved CSV summary to folder: {results_csv_path}")


    train.unpersist()
    test.unpersist()
    spark.stop()
    print("\nDONE ")


if __name__ == "__main__":
    main()
