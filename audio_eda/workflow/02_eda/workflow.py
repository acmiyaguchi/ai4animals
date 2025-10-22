from pyspark.sql import SparkSession, functions as F
import luigi
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("luigi-interface")


def get_spark():
    return SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()


class SchemaAndCountsTask(luigi.Task):
    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/schema_and_counts.txt")

    def run(self):
        spark = get_spark()
        embed = spark.read.parquet(f"{self.input_path}/embed.parquet")
        predict = spark.read.parquet(f"{self.input_path}/predict.parquet")

        # predictions are large, so make them an array column
        # we were running into issues with this in pandas before
        predict = predict.select(
            *predict.columns[:3],
            F.array(*[F.col(c) for c in predict.columns[3:]]).alias("predictions"),
        )

        # do some stuff to get these schemas and counts writte out to file instead stdout
        with open(self.output().path, "w") as f:
            f.write("Embed Schema:\n")
            f.write(embed._jdf.schema().treeString())
            f.write("\n\nPredict Schema:\n")
            f.write(predict._jdf.schema().treeString())

            f.write("\n\nExample Embed Rows:\n")
            f.write(embed._jdf.showString(20, 20, False))
            f.write("\n\nExample Predict Rows:\n")
            f.write(predict._jdf.showString(20, 20, False))

            f.write("\n\nEmbed Counts:\n")
            f.write(f"Total rows: {embed.count()}\n")
            f.write(
                f"Distinct audio files: {embed.select('file').distinct().count()}\n"
            )


class Workflow(luigi.Task):
    def run(self):
        input_root = Path(
            "~/scratch/ai4animals/audio_eda/pacific_sounds/processed"
        ).expanduser()
        output_root = Path(__file__).parent / "results"
        output_root.mkdir(parents=True, exist_ok=True)

        yield [
            SchemaAndCountsTask(
                input_path=str(input_root),
                output_path=str(output_root),
            )
        ]


def main():
    luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    main()
