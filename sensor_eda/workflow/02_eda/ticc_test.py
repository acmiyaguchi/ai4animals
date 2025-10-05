import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import typer
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from ticc.TICC_solver import TICC

app = typer.Typer()


def get_spark():
    return SparkSession.builder.config("spark.local.dir", "/tmp").getOrCreate()


behaviors = [
    "drinking_milk",
    "lying",
    "running",
    "standing",
]


def update_behaviour(label):
    if label in behaviors:
        return label
    else:
        return "other"


def map_behavior_to_int(label):
    mapping = {b: i for i, b in enumerate(behaviors)}
    return mapping.get(label, -1)


@app.command()
def fit(
    input_path="~/scratch/ai4animals/sensor_eda/AcTBeCalf.csv",
    output_path="~/scratch/ai4animals/sensor_eda/ticc/v2/fit",
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
        .withColumn("behaviour_id", F.udf(map_behavior_to_int, "int")("behaviour"))
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
    test.select("behaviour_id", "behaviour").distinct().orderBy("behaviour_id").show()

    # now get the accelerometer data
    pdf = test.select("dateTime", "accX", "accY", "accZ", "behaviour").toPandas()
    pdf.plot(x="dateTime", y=["accX", "accY", "accZ"])
    plt.title(f"Accelerometer Data for Calf 1306 (First {k} Segments)")
    plt.savefig((output_path / "accelerometer_data.png").as_posix())
    plt.close()
    plt.clf()

    pdf.plot(y=["accX", "accY", "accZ"])
    plt.title(f"Accelerometer Data for Calf 1306 (First {k} Segments)")
    plt.savefig((output_path / "accelerometer_data_no_time.png").as_posix())
    plt.close()
    plt.clf()

    # also plot the behaviours
    pdf.plot(y="behaviour_id")
    plt.title(f"Behaviours for Calf 1306 (First {k} Segments)")
    plt.savefig((output_path / "behaviours.png").as_posix())
    plt.close()
    plt.clf()

    X = pdf[["accX", "accY", "accZ"]].to_numpy()
    infile = (output_path / "test_ticc.txt").as_posix()
    np.savetxt(infile, X, delimiter=",")

    # 25 hz, so window size of 500 is 20 seconds
    # number_of_clusters = 4  # lying, standing, drinking, walking/running
    start_time = time.time()
    ticc = TICC(
        window_size=500,
        number_of_clusters=4,
        lambda_parameter=11e-2,
        # switching penalty
        beta=200,
        maxIters=100,
        threshold=2e-5,
        write_out_file=True,
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


@app.command()
def validate(
    input_path="~/scratch/ai4animals/sensor_eda/ticc/v2/fit",
    output_path="~/scratch/ai4animals/sensor_eda/ticc/v2/validate",
):
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    with (input_path / "ticc_output.pkl").open("rb") as f:
        cluster, mrfs = pickle.load(f)

    print(f"Cluster assignments shape: {np.array(cluster).shape}")
    print(f"Number of clusters: {len(mrfs)}")

    # plot the clusters to see if they make sense
    plt.plot(cluster)
    plt.title("TICC Cluster Assignments")
    plt.xlabel("Window Index")
    plt.ylabel("Cluster")
    plt.savefig((output_path / "cluster_assignments.png").as_posix())
    plt.close()
    plt.clf()


if __name__ == "__main__":
    app()
