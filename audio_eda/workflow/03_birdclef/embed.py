import json
import logging
from pathlib import Path

import bioacoustics_model_zoo as bmz
import luigi
from contexttimer import Timer
from pyspark.sql import SparkSession, functions as F
import random

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("luigi-interface")


class OptionsMixin:
    input_path = luigi.Parameter(
        description="Path to the input audio file or directory containing audio files",
    )
    output_root = luigi.Parameter(
        description="Directory to save the output files",
    )
    clip_step = luigi.FloatParameter(
        default=5.0,
        description="The increment in seconds between starts of consecutive clips",
    )


class EmbedAudio(luigi.Task, OptionsMixin):
    def output(self):
        output_dir = Path(self.output_root).expanduser() / "parts"
        name = Path(self.input_path).expanduser().stem
        return {
            "embed": luigi.LocalTarget(output_dir / f"embed/{name}.parquet"),
            "predict": luigi.LocalTarget(output_dir / f"predict/{name}.parquet"),
            "timing": luigi.LocalTarget(output_dir / f"timing/{name}.json"),
        }

    def run(self):
        """Process a single partition of audio files."""
        input_path = Path(self.input_path).expanduser()
        logger.info(f"Starting task for {input_path.name}")

        model = bmz.BirdSetEfficientNetB1()

        for key in ["embed", "predict", "timing"]:
            Path(self.output()[key].path).parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating embeddings for {input_path.name}...")
        with Timer() as t:
            results = model.embed(
                [str(input_path)], return_preds=True, clip_step=self.clip_step
            )

        embed_df, predict_df = results
        embed_df.reset_index().to_parquet(self.output()["embed"].path, index=False)
        predict_df.reset_index().to_parquet(self.output()["predict"].path, index=False)

        timing_info = {
            "part_name": input_path.stem,
            "elapsed": t.elapsed,
        }
        with self.output()["timing"].open("w") as f:
            json.dump(timing_info, f)

        logger.info(
            f"Finished task for {input_path.name}. "
            f"Time elapsed: {t.elapsed:.2f} seconds."
        )


class AggregateEmbeddings(luigi.Task):
    input_root = luigi.Parameter()
    output_root = luigi.Parameter()

    def output(self):
        return {
            "embed": luigi.LocalTarget(
                Path(self.output_root).expanduser() / "embed.parquet"
            ),
            "predict": luigi.LocalTarget(
                Path(self.output_root).expanduser() / "predict.parquet"
            ),
        }

    def run(self):
        spark = SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()
        embed = spark.read.parquet(f"{self.input_root}/parts/embed")
        predict = spark.read.parquet(f"{self.input_root}/parts/predict")

        # these dataframes need to be vectorized
        stem_udf = F.udf(lambda path: Path(path).stem)

        embed = (
            embed.select(
                *embed.columns[:3],
                F.array(*[F.col(c) for c in embed.columns[3:]]).alias("embedding"),
            )
            .withColumn("file", stem_udf(F.col("file")))
            .orderBy("file", "start_time")
        )

        # don't attempt to vectorize this, there are nearly 10k columns
        # and this is bound to be sparse because we're under the ocean
        predict = predict.withColumn("file", stem_udf(F.col("file"))).orderBy(
            "file", "start_time"
        )

        embed.toPandas().to_parquet(self.output()["embed"].path, index=False)
        predict.toPandas().to_parquet(self.output()["predict"].path, index=False)


class Workflow(luigi.Task):
    def run(self):
        input_root = Path(
            "~/scratch/ai4animals/audio_eda/birdclef/raw/train_soundscapes"
        ).expanduser()
        output_root = Path(
            "~/scratch/ai4animals/audio_eda/birdclef/processed"
        ).expanduser()

        wav_files = list(input_root.glob("*.wav"))
        # randomly sample 10 files (with a seed)
        wav_files = random.sample(wav_files, 10)

        logger.info(f"Workflow found {len(wav_files)} .wav files to process.")

        yield [
            EmbedAudio(
                input_path=str(p),
                output_root=str(output_root),
                clip_step=0.5,
            )
            for p in wav_files
        ]

        yield AggregateEmbeddings(
            input_root=str(output_root),
            output_root=str(output_root),
        )


def main():
    luigi.build([Workflow()], local_scheduler=True, workers=5)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
