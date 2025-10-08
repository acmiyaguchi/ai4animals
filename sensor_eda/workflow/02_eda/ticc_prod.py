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
    "oral_manipulation",
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
    output_path="~/scratch/ai4animals/sensor_eda/ticc/v5/fit",
):
    input_path = Path(input_path).expanduser()
    output_path = Path(output_path).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)

    spark = get_spark()
    df = spark.read.csv(input_path.as_posix(), header=True, inferSchema=True).cache()
    seed = 42

    # we have 4 subsequences, and we want 5 of each to see if we can distinguish between
    # the clusters.
    test = (
        df.withColumn("behaviour", F.udf(update_behaviour, "string")("behaviour"))
        .where(F.col("behaviour") != "other")
        .withColumn("behaviourId", F.udf(map_behavior_to_int, "int")("behaviour"))
    ).cache()
    # for each behavior, let's randomly select k of the segments
    # keep the first few
    seg = (
        test.select("calfId", "segId", "behaviourId", "behaviour")
        .distinct()
        .withColumn(
            "rank",
            F.row_number().over(Window.partitionBy("behaviourId").orderBy(F.rand())),
        )
        .where(F.col("rank") <= 5)
        .drop("rank")
        .withColumn("sortKey", F.rand(seed))
        .orderBy("sortKey")
    )
    seg.show(100, truncate=False)
    # randomize segment id orders, but make segments within a segment ordered by time
    test = (
        test.join(seg.select("calfId", "segId", "sortKey"), on=["calfId", "segId"])
        .withColumn("uid", F.row_number().over(Window.orderBy("sortKey", "dateTime")))
        .drop("sortKey")
        .orderBy("uid")
    ).cache()

    # distribution of behaviors
    test.select("calfId", "segId", "behaviour").distinct().groupBy(
        "behaviour"
    ).count().show()
    test.select("behaviourId", "behaviour").distinct().orderBy("behaviourId").show()
    test.describe().show()

    pdf = test.select(
        "uid", "dateTime", "accX", "accY", "accZ", "behaviourId", "behaviour"
    ).toPandas()

    # now get the accelerometer data
    pdf.plot(x="uid", y=["accX", "accY", "accZ"])
    plt.title("Accelerometer Data for Selected Segments")
    plt.savefig((output_path / "accelerometer_data_no_time.png").as_posix())
    plt.close()
    plt.clf()

    # also plot the behaviours
    pdf.plot(x="uid", y="behaviourId")
    plt.title("Behaviours for Selected Segments")
    plt.savefig((output_path / "behaviours.png").as_posix())
    plt.close()
    plt.clf()

    X = pdf[["accX", "accY", "accZ"]].to_numpy()
    infile = (output_path / "test_ticc.txt").as_posix()
    np.savetxt(infile, X, delimiter=",")

    # also we need to save the original labels for validation later
    pdf[["uid", "dateTime", "behaviourId", "behaviour"]].to_csv(
        (output_path / "original_labels.csv").as_posix(), index=False
    )

    # 25 hz, so for a window of 125 we have 5 seconds
    # number_of_clusters = 4  # lying, standing, drinking, walking/running
    start_time = time.time()
    ticc = TICC(
        window_size=25,
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
    input_path="~/scratch/ai4animals/sensor_eda/ticc/v5/fit",
    output_path="~/scratch/ai4animals/sensor_eda/ticc/v5/validate",
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
