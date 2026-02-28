from pathlib import Path
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def make_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("SUSY_GBT_CrossValidator_8GB")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed" / "susy_parquet"
    models_dir = root / "models"
    reports_dir = root / "reports" / "metrics"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    spark = make_spark()

    df = spark.read.parquet(str(data_path)).withColumn("label", F.col("label").cast("double"))
    feature_cols = [c for c in df.columns if c != "label"]

    SAMPLE_FRACTION = 0.30
    df = df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=42)

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    train = train.repartition(64).cache()
    test = test.repartition(64).cache()
    train.count()
    test.count()

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        seed=42
    )

    pipeline = Pipeline(stages=[assembler, gbt])

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [4, 5])
        .addGrid(gbt.maxIter, [30, 50])
        .addGrid(gbt.stepSize, [0.05, 0.1])
        .build()
    )

    evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3,
        parallelism=2,  
        seed=42
    )

    print("🚀 Starting CrossValidator training (GBT)...")
    cv_model = cv.fit(train)

    preds = cv_model.transform(test)
    test_auc = evaluator.evaluate(preds)

    best_model = cv_model.bestModel
    best_gbt = best_model.stages[-1]

    print("\nBest GBT Params:")
    print("  maxDepth:", best_gbt.getMaxDepth())
    print("  maxIter:", best_gbt.getMaxIter())
    print("  stepSize:", best_gbt.getStepSize())
    print("Test ROC-AUC:", test_auc)

    best_path = models_dir / f"gbt_cv_{run_id}"
    best_model.write().overwrite().save(str(best_path))
    print(f" Saved best model to: {best_path}")

    rows = []
    for params, metric in zip(param_grid, cv_model.avgMetrics):
        rows.append({
            "run_id": run_id,
            "sample_fraction": SAMPLE_FRACTION,
            "num_folds": 3,
            "metric_auc_roc_mean": float(metric),
            "maxDepth": int(params[gbt.maxDepth]),
            "maxIter": int(params[gbt.maxIter]),
            "stepSize": float(params[gbt.stepSize]),
        })

    results_df = spark.createDataFrame(rows).orderBy(F.desc("metric_auc_roc_mean"))

    out_dir = reports_dir / f"gbt_cv_results_{run_id}"
    results_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(out_dir))
    print(f" Saved CV results CSV to: {out_dir}")

    train.unpersist()
    test.unpersist()
    spark.stop()
    print("\nDONE ")


if __name__ == "__main__":
    main()
