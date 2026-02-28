import json
from pathlib import Path
from pyspark.sql import SparkSession

def main():
    root = Path(__file__).resolve().parents[1]
    metrics_dir = root / "reports" / "metrics"
    output_dir = metrics_dir / "summary_from_json"

    spark = (
        SparkSession.builder
        .appName("Generate_Summary_From_JSON")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

    records = []

    for json_file in metrics_dir.glob("*.json"):
        with open(json_file, "r") as f:
            data = json.load(f)

        if "confusion_matrix_counts" in data:
            data["confusion_matrix_counts"] = json.dumps(data["confusion_matrix_counts"])

        records.append(data)

    df = spark.createDataFrame(records)

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(str(output_dir))

    print(f" Summary CSV created at: {output_dir}")

    spark.stop()

if __name__ == "__main__":
    main()
