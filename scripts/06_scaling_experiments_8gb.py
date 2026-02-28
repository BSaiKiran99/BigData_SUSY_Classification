from pathlib import Path
import time
from datetime import datetime

from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator


def make_spark(master: str, shuffle_parts: int) -> SparkSession:
    return (
        SparkSession.builder
        .appName(f"SUSY_Scaling_{master}_sp{shuffle_parts}")
        .master(master)
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.shuffle.partitions", str(shuffle_parts))
        .config("spark.default.parallelism", str(shuffle_parts))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def run_one(root: Path, master: str, shuffle_parts: int, sample_fraction: float, run_id: str) -> dict:
    spark = make_spark(master, shuffle_parts)

    data_path = root / "data" / "processed" / "susy_parquet"
    df = spark.read.parquet(str(data_path)).withColumn("label", F.col("label").cast("double"))

    feature_cols = [c for c in df.columns if c != "label"]

    df = df.sample(withReplacement=False, fraction=sample_fraction, seed=42)
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    train = train.repartition(shuffle_parts).cache()
    test = test.repartition(shuffle_parts).cache()
    train.count()
    test.count()

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

    gbt = GBTClassifier(
        labelCol="label",
        featuresCol="features",
        seed=42,
        maxDepth=5,
        maxIter=30,
        checkpointInterval=10,
        maxBins=32
    )

    pipeline = Pipeline(stages=[assembler, gbt])
    evaluator = BinaryClassificationEvaluator(labelCol="label", metricName="areaUnderROC")

    start = time.time()
    model = pipeline.fit(train)
    fit_seconds = time.time() - start

    start = time.time()
    preds = model.transform(test)
    auc = evaluator.evaluate(preds)
    eval_seconds = time.time() - start

    train.unpersist()
    test.unpersist()
    spark.stop()

    return {
        "run_id": run_id,
        "master": master,
        "shuffle_partitions": shuffle_parts,
        "sample_fraction": sample_fraction,
        "fit_seconds": float(fit_seconds),
        "eval_seconds": float(eval_seconds),
        "auc_roc": float(auc),
    }


def main():
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "reports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")


    masters = ["local[2]", "local[4]", "local[6]"]

    sample_fracs = [0.05, 0.10, 0.20]

    shuffle_options = [16, 32, 64]

    results = []

    fixed_sample = 0.10
    for m in masters:
        for sp in shuffle_options:
            print(f"Running strong scaling: {m}, sp={sp}, sample={fixed_sample}")
            results.append(run_one(root, m, sp, fixed_sample, run_id))

    weak_pairs = [("local[2]", 0.05), ("local[4]", 0.10), ("local[6]", 0.20)]
    for m, sf in weak_pairs:
        sp = 32
        print(f"Running weak scaling: {m}, sp={sp}, sample={sf}")
        results.append(run_one(root, m, sp, sf, run_id))


    import pandas as pd
    out_csv = out_dir / f"scaling_results_{run_id}.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False)
    print(f"\n Saved scaling results to: {out_csv}")


if __name__ == "__main__":
    main()
