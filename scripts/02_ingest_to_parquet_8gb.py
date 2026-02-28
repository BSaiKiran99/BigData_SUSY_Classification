from pathlib import Path
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, DoubleType

COLS = [
    "label",
    "lepton_1_pT", "lepton_1_eta", "lepton_1_phi",
    "lepton_2_pT", "lepton_2_eta", "lepton_2_phi",
    "missing_energy_magnitude", "missing_energy_phi",
    "MET_rel", "axial_MET", "M_R", "M_TR_2", "R", "MT2",
    "S_R", "M_Delta_R", "dPhi_r_b", "cos_theta_r1"
]

def build_schema() -> StructType:
    return StructType([StructField(c, DoubleType(), True) for c in COLS])

def make_spark() -> SparkSession:

    return (
        SparkSession.builder
        .appName("SUSY_Ingest_8GB")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.driver.maxResultSize", "1g")
        .config("spark.sql.shuffle.partitions", "64")
        .config("spark.default.parallelism", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024)) 
        .getOrCreate()
    )

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    in_path = root / "data" / "raw" / "SUSY.csv.gz"
    out_dir = root / "data" / "processed" / "susy_parquet"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    spark = make_spark()

    df = (
        spark.read.format("csv")
        .option("header", "false")
        .schema(build_schema())
        .load(str(in_path))
    )

    print("Schema:")
    df.printSchema()

    print("Label distribution (this triggers a job):")
    df.groupBy("label").count().show()

    df = df.repartition(64, "label")

    (
        df.write.mode("overwrite")
        .option("compression", "snappy")
        .partitionBy("label")
        .parquet(str(out_dir))
    )

    print(f" Parquet saved at: {out_dir}")
    spark.stop()

if __name__ == "__main__":
    main()
