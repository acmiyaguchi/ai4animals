from pyspark.sql import SparkSession, functions as F, Window
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from ticc.TICC_solver import TICC
from pathlib import Path
import typer

app = typer.Typer()


def get_spark():
    return SparkSession.builder.config("spark.local.dir", "/tmp").getOrCreate()


def update_behaviour(label):
    if label in ["drinking_milk", "lying", "running", "standing"]:
        return label
    else:
        return "other"


@app.command()
def main(
    input_path="~/scratch/ai4animals/sensor_eda/AcTBeCalf.csv",
    output_path="~/scratch/ai4animals/sensor_eda/ticc",
):
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    spark = get_spark()
    df = spark.read.csv(input_path.as_posix(), header=True, inferSchema=True).cache()

    k = 20
    # take the first few sequences just for testing
    test = (
        df.where("calfId = 1306")
        .withColumn("behaviour", F.udf(update_behaviour, "string")("behaviour"))
        .where(F.col("behaviour") != "other")
        .orderBy("dateTime")
    )
    # keep the first few
    seg = (
        test.select("calfId", "segId")
        .distinct()
        .withColumn(
            "rank", F.row_number().over(Window.partitionBy("calfId").orderBy("segId"))
        )
    )
    test = test.join(
        seg.where(f"rank <= {k}").select("calfId", "segId"), on=["calfId", "segId"]
    ).orderBy("dateTime")

    # distribution of behaviors
    test.select("segId", "behaviour").distinct().groupBy("behaviour").count().show()

    # now get the accelerometer data
    pdf = test.select("dateTime", "accX", "accY", "accZ", "behaviour").toPandas()
    pdf.plot(x="dateTime", y=["accX", "accY", "accZ"])
    plt.title(f"Accelerometer Data for Calf 1306 (First {k} Segments)")
    plt.savefig((output_path / "accelerometer_data.png").as_posix())
    plt.close()

    X = pdf[["accX", "accY", "accZ"]].to_numpy()
    infile = (output_path / "test_ticc.txt").as_posix()
    np.savetxt(infile, X, delimiter=",")

    print("Running TICC with 3 clusters and window size of 100")
    # 25 hz, so window size of 100 is 4 seconds
    start_time = time.time()
    ticc = TICC(
        window_size=100,
        number_of_clusters=3,
        lambda_parameter=11e-2,
        beta=600,
        maxIters=100,
        threshold=2e-5,
        write_out_file=False,
        prefix_string=output_path.as_posix() + "/",
        num_proc=4,
    )
    (cluster_assignment, cluster_MRFs) = ticc.fit(input_file=infile)
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(
        f"TICC training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)"
    )

    # just pickle the data
    with open((output_path / "ticc_output.pkl").as_posix(), "wb") as f:
        pickle.dump((cluster_assignment, cluster_MRFs), f)
    print("TICC done")


if __name__ == "__main__":
    main()
