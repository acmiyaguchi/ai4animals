import logging
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
import umap
from matplotlib.colors import SymLogNorm
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from sklearn.decomposition import PCA

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


class PlotMatrixProfile(luigi.Task):
    """
    Calculates and plots the multi-dimensional matrix profile for the SVD components.
    This version plots the full N-dimensional profile, which is more standard.
    """

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)  # 10 second window (20 * 0.5s)

    def output(self):
        return luigi.LocalTarget(Path(self.output_path) / "matrix_profile.png")

    def run(self):
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        # choose first file
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")

        if df_filtered.empty:
            raise ValueError(f"No data found for file: {file_to_plot}")

        logger.info(f"Generating Matrix Profile plot for {file_to_plot}")
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(embeddings)  # Shape (n_samples, n_components)

        # Transpose for stumpy: (n_dimensions, n_samples)
        svd_time_series = components.T

        # Ensure data is C-contiguous (mstump optimization)
        svd_time_series = np.ascontiguousarray(svd_time_series)

        logger.info(f"Running stumpy.mstump with m={self.window_size}...")
        m = self.window_size
        mp, mpi = stumpy.mstump(svd_time_series, m=m)

        # Get the full N-dimensional matrix profile (the last row)
        full_mp = mp[-1, :]

        # Find the top motif (lowest value in the full-N-dim profile)
        motif_idx = np.argmin(full_mp)
        neighbor_idx = mpi[-1, motif_idx]

        logger.info(f"Motif found at index: {motif_idx}, Neighbor at: {neighbor_idx}")

        # Create a 2-panel plot
        fig, axs = plt.subplots(
            2, 1, figsize=(15, 10), sharex=True, gridspec_kw={"height_ratios": [1, 1]}
        )

        # 1. Plot the SVD heatmap on top
        linthresh = np.abs(components).mean() * 0.1
        if linthresh == 0:
            linthresh = 1e-5
        norm = SymLogNorm(
            linthresh=linthresh, vmin=np.min(components), vmax=np.max(components)
        )
        axs[0].imshow(components.T, aspect="auto", cmap="viridis", norm=norm)
        axs[0].set_title(
            f"Top {self.n_components} SVD Components for {Path(file_to_plot).name}"
        )
        axs[0].set_ylabel("SVD Component Index")

        # 2. Plot the Full N-Dimensional Matrix Profile on the bottom
        axs[1].plot(full_mp)
        axs[1].set_title(f"Full {self.n_components}-Dimensional Matrix Profile")
        axs[1].set_xlabel("Time (clip index)")
        axs[1].set_ylabel("Z-Normalized Distance")

        # Add vertical lines for the motif and neighbor to both plots
        for ax in axs:
            ax.axvline(motif_idx, color="r", linestyle="--", label="Motif")
            ax.axvline(neighbor_idx, color="g", linestyle="--", label="Neighbor")
        axs[0].legend()
        axs[1].legend()

        plt.tight_layout()
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output().path)
        plt.close()


class PlotMotifDetail(luigi.Task):
    """
    Plots each SVD component as a separate line plot, highlighting the
    location of the top motif and its neighbor.
    """

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)  # 10 second window (20 * 0.5s)

    def output(self):
        return luigi.LocalTarget(Path(self.output_path) / "motif_detail.png")

    def run(self):
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        # choose first file
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")

        if df_filtered.empty:
            raise ValueError(f"No data found for file: {file_to_plot}")

        logger.info(f"Generating Motif Detail plot for {file_to_plot}")
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(embeddings)  # Shape (n_samples, n_components)
        svd_time_series = components.T  # Shape (n_components, n_samples)
        svd_time_series_c = np.ascontiguousarray(svd_time_series)

        logger.info(f"Running stumpy.mstump with m={self.window_size}...")
        m = self.window_size
        mp, mpi = stumpy.mstump(svd_time_series_c, m=m)

        # Get the full N-dimensional matrix profile (the last row)
        full_mp = mp[-1, :]
        motif_idx = np.argmin(full_mp)
        neighbor_idx = mpi[-1, motif_idx]

        logger.info(f"Plotting motif at {motif_idx} and neighbor at {neighbor_idx}")

        # Create a multi-panel plot, one for each component
        fig, axs = plt.subplots(
            self.n_components, 1, figsize=(15, 2 * self.n_components), sharex=True
        )
        fig.suptitle(
            f"Motif Detail: SVD Components for {Path(file_to_plot).name}",
            fontsize=16,
            y=1.02,
        )

        for i in range(self.n_components):
            axs[i].plot(svd_time_series[i, :])
            axs[i].set_ylabel(f"SVD {i}")

            # Add shaded regions for motif and neighbor
            axs[i].axvspan(
                motif_idx, motif_idx + m, color="r", alpha=0.3, label="Motif"
            )
            axs[i].axvspan(
                neighbor_idx, neighbor_idx + m, color="g", alpha=0.3, label="Neighbor"
            )

        axs[0].legend(loc="upper right")
        axs[-1].set_xlabel("Time (clip index)")

        plt.tight_layout()
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output().path)
        plt.close()


class PlotNaturalDimensionality(luigi.Task):
    """
    Plots the "Elbow Plot" (Fig. 6 in the mSTAMP paper) to help
    find the "natural" dimensionality of the motifs.
    """

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)

    def output(self):
        return luigi.LocalTarget(
            Path(self.output_path) / "natural_dimensionality_plot.png"
        )

    def _get_data(self):
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(embeddings)
        svd_time_series = np.ascontiguousarray(components.T)
        return svd_time_series, Path(file_to_plot).name

    def run(self):
        svd_time_series, file_name = self._get_data()
        m = self.window_size

        logger.info("Running stumpy.mstump for elbow plot...")
        mp, mpi = stumpy.mstump(svd_time_series, m=m)

        # For each k, find the minimum value of its k-dimensional profile
        min_distances = [mp[k, :].min() for k in range(self.n_components)]

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, self.n_components + 1), min_distances, marker="o", linestyle="--"
        )
        plt.title(f"Natural Dimensionality Elbow Plot for {file_name}")
        plt.xlabel("Number of Dimensions (k)")
        plt.ylabel("Minimum Matrix Profile Value (Distance)")
        plt.xticks(range(1, self.n_components + 1))
        plt.grid(True)

        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output().path)
        plt.close()


class PlotTopDiscords(luigi.Task):
    """Finds and plots the top K discords (anomalies)."""

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)
    top_k: int = luigi.IntParameter(default=3)

    def output(self):
        return luigi.LocalTarget(Path(self.output_path) / "top_discords.png")

    def _get_data(self):
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(embeddings)
        return components, Path(file_to_plot).name

    def run(self):
        components, file_name = self._get_data()
        svd_time_series = np.ascontiguousarray(components.T)
        m = self.window_size

        logger.info("Running stumpy.mstump for discords...")
        mp, mpi = stumpy.mstump(svd_time_series, m=m)
        full_mp = mp[-1, :]

        # Find top k discords (highest values)
        # We need to sort and then pick top_k, ignoring neighbors
        exclusion_zone = m // 2
        sorted_indices = np.argsort(full_mp)[::-1]  # Sort high to low
        discord_indices = []
        for idx in sorted_indices:
            is_neighbor = False
            for d_idx in discord_indices:
                if abs(idx - d_idx) < exclusion_zone:
                    is_neighbor = True
                    break
            if not is_neighbor:
                discord_indices.append(idx)
            if len(discord_indices) == self.top_k:
                break

        logger.info(f"Top {self.top_k} discords found at: {discord_indices}")

        plt.figure(figsize=(15, 5))
        linthresh = np.abs(components).mean() * 0.1
        if linthresh == 0:
            linthresh = 1e-5
        norm = SymLogNorm(
            linthresh=linthresh, vmin=np.min(components), vmax=np.max(components)
        )
        plt.imshow(components.T, aspect="auto", cmap="viridis", norm=norm)

        colors = plt.cm.autumn(np.linspace(0, 1, self.top_k))
        for i, idx in enumerate(discord_indices):
            plt.axvline(
                idx,
                color=colors[i],
                linestyle="--",
                label=f"Discord {i + 1} (idx {idx})",
            )

        plt.title(f"Top {self.top_k} Discords for {file_name}")
        plt.xlabel("Time (clip index)")
        plt.ylabel("SVD Component Index")
        plt.legend()
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output().path)
        plt.close()


class PlotTopMotifs(luigi.Task):
    """Finds and plots the top K motifs (repeating patterns)."""

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)
    top_k: int = luigi.IntParameter(default=3)

    def output(self):
        return luigi.LocalTarget(Path(self.output_path) / "top_motifs.png")

    def _get_data(self):
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(embeddings)
        return components, Path(file_to_plot).name

    def run(self):
        components, file_name = self._get_data()
        svd_time_series = np.ascontiguousarray(components.T)
        m = self.window_size

        logger.info(f"Running stumpy.mstump for motifs...")
        mp, mpi = stumpy.mstump(svd_time_series, m=m)
        full_mp = mp[-1, :]
        full_mpi = mpi[-1, :]

        # Manually find top k motifs (lowest values), respecting exclusion zones
        exclusion_zone = m // 2
        sorted_indices = np.argsort(full_mp)  # Sort low to high

        motif_groups = []  # Will store tuples of (distance, [idx, neighbor_idx])

        # Keep track of indices that have been used
        used_indices = np.zeros_like(full_mp, dtype=bool)

        for idx in sorted_indices:
            if used_indices[idx]:
                continue

            neighbor_idx = full_mpi[idx]

            # Check if neighbor is also unused
            if used_indices[neighbor_idx]:
                continue

            # Add this motif pair (and its neighbors) to the used list
            motif_dist = full_mp[idx]
            motif_groups.append((motif_dist, [idx, neighbor_idx]))

            # Apply exclusion zone around both motif and neighbor
            used_indices[
                max(0, idx - exclusion_zone) : min(
                    len(full_mp), idx + exclusion_zone + 1
                )
            ] = True
            used_indices[
                max(0, neighbor_idx - exclusion_zone) : min(
                    len(full_mp), neighbor_idx + exclusion_zone + 1
                )
            ] = True

            if len(motif_groups) == self.top_k:
                break

        logger.info(f"Top {len(motif_groups)} motif groups found.")

        plt.figure(figsize=(15, 5))
        linthresh = np.abs(components).mean() * 0.1
        if linthresh == 0:
            linthresh = 1e-5
        norm = SymLogNorm(
            linthresh=linthresh, vmin=np.min(components), vmax=np.max(components)
        )
        plt.imshow(components.T, aspect="auto", cmap="viridis", norm=norm)

        colors = plt.cm.cool(np.linspace(0, 1, len(motif_groups)))
        # Loop through the found motif groups
        for i, (distance, indices) in enumerate(motif_groups):
            label = f"Motif {i + 1} (dist: {distance:.2f})"
            for j, idx in enumerate(indices):
                plt.axvspan(
                    idx,
                    idx + m,
                    color=colors[i],
                    alpha=0.4,
                    label=label if j == 0 else None,  # Only label first one
                )
    
        plt.title(f"Top {len(motif_groups)} Motifs for {file_name}")
        plt.xlabel("Time (clip index)")
        plt.ylabel("SVD Component Index")
        plt.legend()
        Path(self.output().path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output().path)
        plt.close()


class Workflow(luigi.Task):
    def run(self):
        input_root = Path(
            "~/scratch/ai4animals/audio_eda/pacific_sounds/processed"
        ).expanduser()
        output_root = Path(__file__).parent / "results"
        output_root.mkdir(parents=True, exist_ok=True)

        yield [
            SchemaAndCountsTask(
                input_path=str(input_root), output_path=str(output_root)
            ),
            PlotUMAPScatter(input_path=str(input_root), output_path=str(output_root)),
            PlotScreePlot(input_path=str(input_root), output_path=str(output_root)),
            PlotSVDHeatmap(
                input_path=str(input_root),
                output_path=str(output_root),
                n_components=16,
            ),
            PlotMatrixProfile(
                input_path=str(input_root),
                output_path=str(output_root),
                n_components=16,
                window_size=20,
            ),
            PlotMotifDetail(
                input_path=str(input_root),
                output_path=str(output_root),
                n_components=16,
                window_size=20,
            ),
            PlotNaturalDimensionality(
                input_path=str(input_root),
                output_path=str(output_root),
                n_components=16,
                window_size=20,
            ),
            PlotTopDiscords(
                input_path=str(input_root),
                output_path=str(output_root),
                n_components=16,
                window_size=20,
                top_k=3,
            ),
            PlotTopMotifs(
                input_path=str(input_root),
                output_path=str(output_root),
                n_components=16,
                window_size=20,
                top_k=3,
            ),
        ]


def main():
    luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    main()
