"""
Tasks for creating visual representations of audio embeddings.

This module contains tasks for exploring and visualizing the structure of audio embeddings
through various dimensionality reduction and statistical analysis techniques.
"""

import logging
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from matplotlib.colors import SymLogNorm
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class SchemaAndCountsTask(luigi.Task):
    """Examine and document the schema and basic statistics of embeddings."""

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(f"{self.output_path}/schema_and_counts.txt")

    def run(self):
        from pyspark.sql import SparkSession
        from pyspark.sql import functions as F

        spark = SparkSession.builder.config("spark.driver.memory", "8g").getOrCreate()
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


class PlotUMAPScatter(luigi.Task):
    """Use UMAP to plot a 2D scatter of the embeddings for the first file."""

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()

    def output(self):
        return luigi.LocalTarget(Path(self.output_path) / "umap_scatter.png")

    def run(self):
        # Reading with pandas is more convenient for ML tasks on single files
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        # choose first file
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")

        if df_filtered.empty:
            raise ValueError(f"No data found for file: {file_to_plot}")

        logger.info(f"Generating UMAP plot for {file_to_plot}")
        embeddings = np.stack(df_filtered["embedding"].values)

        reducer = umap.UMAP(n_components=2, random_state=42)
        embedding_2d = reducer.fit_transform(embeddings)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            embedding_2d[:, 0],
            embedding_2d[:, 1],
            c=df_filtered["start_time"],
            cmap="viridis",
            s=5,
        )
        plt.title(f"UMAP 2D Visualization for {Path(file_to_plot).name}")
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.colorbar(scatter, label="start_time (seconds)")
        plt.grid(True)
        plt.tight_layout()

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output().path, dpi=300)
        plt.close()


class PlotScreePlot(luigi.Task):
    """Plot the scree plot of the SVD components for the first file."""

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()

    def output(self):
        return {
            "plot": luigi.LocalTarget(Path(self.output_path) / "scree_plot.png"),
            "data": luigi.LocalTarget(Path(self.output_path) / "scree_plot_data.txt"),
        }

    def run(self):
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        # choose first file
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot]

        if df_filtered.empty:
            raise ValueError(f"No data found for file: {file_to_plot}")

        logger.info(f"Generating Scree plot for {file_to_plot}")
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA()
        pca.fit(embeddings)

        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

        plt.figure(figsize=(10, 6))
        plt.plot(cumulative_variance, marker=".", linestyle="-")
        plt.title(f"PCA Cumulative Explained Variance for {Path(file_to_plot).name}")
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.grid(True)
        plt.axhline(y=0.8, color="r", linestyle="--", label="80% Cutoff")
        plt.legend()
        plt.tight_layout()

        Path(self.output()["plot"].path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output()["plot"].path, dpi=300)
        plt.close()

        # Save the cumulative variance data to the text file
        np.savetxt(self.output()["data"].path, cumulative_variance, fmt="%.8f")


class PlotSVDHeatmap(luigi.Task):
    """Create a spectrogram-like heatmap of top SVD components for the first file."""

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=20)

    def output(self):
        return luigi.LocalTarget(Path(self.output_path) / "svd_heatmap.png")

    def run(self):
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        # choose first file
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")

        if df_filtered.empty:
            raise ValueError(f"No data found for file: {file_to_plot}")

        logger.info(f"Generating SVD Heatmap for {file_to_plot}")
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(embeddings)

        # Use SymLogNorm for log scaling with positive and negative values
        # Set a linear threshold around zero.
        linthresh = np.abs(components).mean() * 0.1  # Small threshold
        if linthresh == 0:
            linthresh = 1e-5  # fallback for zero data
        norm = SymLogNorm(
            linthresh=linthresh, vmin=np.min(components), vmax=np.max(components)
        )

        plt.figure(figsize=(15, 5))
        plt.imshow(components.T, aspect="auto", cmap="viridis", norm=norm)
        plt.title(
            f"Top {self.n_components} SVD Components Over Time for {Path(file_to_plot).name}"
        )
        plt.xlabel("Time (clip index)")
        plt.ylabel("SVD Component Index")
        plt.colorbar(label="Component Value")

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output().path)
        plt.close()
