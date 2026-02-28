from pathlib import Path
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder


def make_spark(checkpoint_dir: str) -> SparkSession:
    spark = (
        SparkSession.builder
        .appName("SUSY_GBT_CV_8GB_LIGHT")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.default.parallelism", "32")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )
    spark.sparkContext.setCheckpointDir(checkpoint_dir)
    return spark


def main():
    root = Path(__file__).resolve().parents[1]
    data_path = root / "data" / "processed" / "susy_parquet"
    models_dir = root / "models"
    reports_dir = root / "reports" / "metrics"
    checkpoint_dir = str(root / "logs" / "spark_checkpoints")

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    spark = make_spark(checkpoint_dir)

    df = spark.read.parquet(str(data_path)).withColumn("label", F.col("label").cast("double"))
    feature_cols = [c for c in df.columns if c != "label"]

    SAMPLE_FRACTION = 0.10
    df = df.sample(withReplacement=False, fraction=SAMPLE_FRACTION, seed=42)

    train, test = df.randomSplit([0.8, 0.2], seed=42)

    train = train.repartition(32).cache()
    test = test.repartition(32).cache()
    train.count()
    test.count()

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        seed=42,
        checkpointInterval=10,
        maxBins=32
    )

    pipeline = Pipeline(stages=[assembler, gbt])

    param_grid = (
        ParamGridBuilder()
        .addGrid(gbt.maxDepth, [4, 5])
        .addGrid(gbt.maxIter, [20, 30])
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
        numFolds=2,       
        parallelism=1,   
        seed=42
    )

    print("🚀 Starting LIGHT CrossValidator training (GBT)...")
    cv_model = cv.fit(train)

    preds = cv_model.transform(test)
    test_auc = evaluator.evaluate(preds)

    best_model = cv_model.bestModel
    best_gbt = best_model.stages[-1]

    print("\n Best GBT Params:")
    print("  maxDepth:", best_gbt.getMaxDepth())
    print("  maxIter:", best_gbt.getMaxIter())
    print(" Test ROC-AUC:", test_auc)

    best_path = models_dir / f"gbt_cv_light_{run_id}"
    best_model.write().overwrite().save(str(best_path))
    print(f" Saved best model to: {best_path}")

    rows = []
    for params, metric in zip(param_grid, cv_model.avgMetrics):
        rows.append({
            "run_id": run_id,
            "sample_fraction": SAMPLE_FRACTION,
            "num_folds": 2,
            "metric_auc_roc_mean": float(metric),
            "maxDepth": int(params[gbt.maxDepth]),
            "maxIter": int(params[gbt.maxIter]),
        })

    results_df = spark.createDataFrame(rows).orderBy(F.desc("metric_auc_roc_mean"))
    out_dir = reports_dir / f"gbt_cv_light_results_{run_id}"
    results_df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(out_dir))
    print(f" Saved CV results CSV to: {out_dir}")

    train.unpersist()
    test.unpersist()
    spark.stop()
    print("\nDONE ")


if __name__ == "__main__":
    main()
