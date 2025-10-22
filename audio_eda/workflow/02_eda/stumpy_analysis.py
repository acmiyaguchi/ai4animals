"""
Tasks for time series pattern analysis using STUMPY matrix profiling.

This module contains tasks for discovering motifs (repeating patterns) and discords (anomalies)
in multivariate time series data using the STUMPY library.
"""

import json
import logging
from pathlib import Path

import luigi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import stumpy
from matplotlib.colors import SymLogNorm
from sklearn.decomposition import PCA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class PlotNaturalDimensionality(luigi.Task):
    """
    Plots the "Elbow Plot" (Fig. 6 in the mSTAMP paper) to help
    find the "natural" dimensionality of the motifs.
    Also saves the determined 'natural_k'.
    """

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)

    def output(self):
        return {
            "plot": luigi.LocalTarget(
                Path(self.output_path) / "natural_dimensionality_plot.png"
            ),
            "json": luigi.LocalTarget(Path(self.output_path) / "natural_k.json"),
        }

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

        # Find the elbow (natural_k)
        points = np.array([range(1, self.n_components + 1), min_distances]).T
        line_vec = points[-1] - points[0]
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        vec_from_first = points - points[0]
        scalar_proj = vec_from_first.dot(line_vec_norm)
        vec_proj = scalar_proj[:, np.newaxis] * line_vec_norm
        dist_from_line = np.linalg.norm(vec_from_first - vec_proj, axis=1)
        natural_k = np.argmax(dist_from_line) + 1  # +1 for 1-based index

        logger.info(f"Natural dimensionality (k) found: {natural_k}")

        # Save the natural_k to json
        with open(self.output()["json"].path, "w") as f:
            json.dump({"natural_k": int(natural_k)}, f)

        # Plot the elbow plot
        plt.figure(figsize=(10, 6))
        plt.plot(
            range(1, self.n_components + 1), min_distances, marker="o", linestyle="--"
        )
        # Highlight the chosen elbow
        plt.axvline(
            natural_k,
            color="r",
            linestyle="--",
            label=f"Natural k = {natural_k}",
        )
        plt.title(f"Natural Dimensionality Elbow Plot for {file_name}")
        plt.xlabel("Number of Dimensions (k)")
        plt.ylabel("Minimum Matrix Profile Value (Distance)")
        plt.xticks(range(1, self.n_components + 1))
        plt.legend()
        plt.grid(True)

        Path(self.output()["plot"].path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output()["plot"].path)
        plt.close()


class PlotMatrixProfile(luigi.Task):
    """
    Calculates and plots the k-dimensional matrix profile, where 'k'
    is the natural dimensionality found in the previous step.
    """

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)  # 10 second window (20 * 0.5s)

    def requires(self):
        return PlotNaturalDimensionality(
            input_path=self.input_path,
            output_path=self.output_path,
            n_components=self.n_components,
            window_size=self.window_size,
        )

    def output(self):
        return luigi.LocalTarget(Path(self.output_path) / "matrix_profile.png")

    def run(self):
        # Load the natural_k
        with open(self.input()["json"].path, "r") as f:
            natural_k = json.load(f)["natural_k"]

        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")

        if df_filtered.empty:
            raise ValueError(f"No data found for file: {file_to_plot}")

        logger.info(
            f"Generating {natural_k}-Dimensional Matrix Profile plot for {file_to_plot}"
        )
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(embeddings)  # Shape (n_samples, n_components)
        svd_time_series = np.ascontiguousarray(components.T)

        logger.info(f"Running stumpy.mstump with m={self.window_size}...")
        m = self.window_size
        mp, mpi = stumpy.mstump(svd_time_series, m=m)

        # Get the k-dimensional matrix profile (k-1 index)
        k_dim_mp = mp[natural_k - 1, :]
        k_dim_mpi = mpi[natural_k - 1, :]

        # Find the top motif (lowest value in the k-dim profile)
        motif_idx = np.argmin(k_dim_mp)
        neighbor_idx = k_dim_mpi[motif_idx]

        logger.info(f"Motif found at index: {motif_idx}, Neighbor at: {neighbor_idx}")

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

        # 2. Plot the k-Dimensional Matrix Profile on the bottom
        axs[1].plot(k_dim_mp)
        axs[1].set_title(f"Natural {natural_k}-Dimensional Matrix Profile")
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
    location of the top motif and its neighbor from the k-dim profile.
    """

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)  # 10 second window (20 * 0.5s)

    def requires(self):
        return PlotNaturalDimensionality(
            input_path=self.input_path,
            output_path=self.output_path,
            n_components=self.n_components,
            window_size=self.window_size,
        )

    def output(self):
        return luigi.LocalTarget(Path(self.output_path) / "motif_detail.png")

    def run(self):
        with open(self.input()["json"].path, "r") as f:
            natural_k = json.load(f)["natural_k"]

        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
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

        # Get the k-dimensional matrix profile (k-1 index)
        k_dim_mp = mp[natural_k - 1, :]
        motif_idx = np.argmin(k_dim_mp)
        neighbor_idx = mpi[natural_k - 1, motif_idx]

        logger.info(f"Plotting motif at {motif_idx} and neighbor at {neighbor_idx}")

        fig, axs = plt.subplots(
            self.n_components, 1, figsize=(15, 2 * self.n_components), sharex=True
        )
        fig.suptitle(
            f"Motif Detail (k={natural_k}): SVD Components for {Path(file_to_plot).name}",
            fontsize=16,
            y=1.02,
        )

        for i in range(self.n_components):
            axs[i].plot(svd_time_series[i, :])
            axs[i].set_ylabel(f"SVD {i}")

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


class PlotTopDiscords(luigi.Task):
    """
    Finds and plots the top K discords (anomalies) from the k-dim profile.
    Also saves the discord locations to a CSV file.
    """

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)
    top_k: int = luigi.IntParameter(default=3)

    def requires(self):
        return PlotNaturalDimensionality(
            input_path=self.input_path,
            output_path=self.output_path,
            n_components=self.n_components,
            window_size=self.window_size,
        )

    def output(self):
        return {
            "plot": luigi.LocalTarget(Path(self.output_path) / "top_discords.png"),
            "csv": luigi.LocalTarget(Path(self.output_path) / "top_discords.csv"),
        }

    def _get_data(self):
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(embeddings)
        return components, df_filtered, Path(file_to_plot).name

    def run(self):
        with open(self.input()["json"].path, "r") as f:
            natural_k = json.load(f)["natural_k"]

        components, df_filtered, file_name = self._get_data()
        svd_time_series = np.ascontiguousarray(components.T)
        m = self.window_size

        logger.info(f"Running stumpy.mstump for discords (k={natural_k})...")
        mp, mpi = stumpy.mstump(svd_time_series, m=m)
        k_dim_mp = mp[natural_k - 1, :]

        # Find top k discords (highest values)
        exclusion_zone = m // 2
        sorted_indices = np.argsort(k_dim_mp)[::-1]  # Sort high to low
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

        # --- Save to CSV ---
        discord_data = []
        for i, idx in enumerate(discord_indices):
            discord_data.append(
                {
                    "discord_rank": i + 1,
                    "natural_k": natural_k,
                    "distance": k_dim_mp[idx],
                    "idx": idx,
                    "start_time": df_filtered.iloc[idx]["start_time"],
                    "file_name": file_name,
                }
            )
        discord_df = pd.DataFrame(discord_data)
        discord_df.to_csv(self.output()["csv"].path, index=False)
        logger.info(f"Top discords saved to {self.output()['csv'].path}")

        # --- Plotting ---
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
            # Use axvspan to show the full duration of the discord
            plt.axvspan(
                idx,
                idx + m,
                color=colors[i],
                alpha=0.4,  # Use alpha for transparency like motifs
                label=f"Discord {i + 1} (idx {idx})",
            )

        plt.title(f"Top {self.top_k} Discords (k={natural_k}) for {file_name}")
        plt.xlabel("Time (clip index)")
        plt.ylabel("SVD Component Index")
        plt.legend()
        Path(self.output()["plot"].path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output()["plot"].path)
        plt.close()


class PlotTopMotifs(luigi.Task):
    """
    Finds and plots the top K motifs (repeating patterns) from the k-dim profile.
    Also saves the motif locations to a CSV file.
    """

    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)
    top_k: int = luigi.IntParameter(default=3)

    def requires(self):
        return PlotNaturalDimensionality(
            input_path=self.input_path,
            output_path=self.output_path,
            n_components=self.n_components,
            window_size=self.window_size,
        )

    def output(self):
        return {
            "plot": luigi.LocalTarget(Path(self.output_path) / "top_motifs.png"),
            "csv": luigi.LocalTarget(Path(self.output_path) / "top_motifs.csv"),
        }

    def _get_data(self):
        df = pd.read_parquet(f"{self.input_path}/embed.parquet")
        file_to_plot = df["file"].unique()[0]
        df_filtered = df[df["file"] == file_to_plot].sort_values("start_time")
        embeddings = np.stack(df_filtered["embedding"].values)

        pca = PCA(n_components=self.n_components)
        components = pca.fit_transform(embeddings)
        return components, df_filtered, Path(file_to_plot).name

    def run(self):
        with open(self.input()["json"].path, "r") as f:
            natural_k = json.load(f)["natural_k"]

        components, df_filtered, file_name = self._get_data()
        svd_time_series = np.ascontiguousarray(components.T)
        m = self.window_size

        logger.info(f"Running stumpy.mstump for motifs (k={natural_k})...")
        mp, mpi = stumpy.mstump(svd_time_series, m=m)
        k_dim_mp = mp[natural_k - 1, :]
        k_dim_mpi = mpi[natural_k - 1, :]

        # Manually find top k motifs (lowest values), respecting exclusion zones
        exclusion_zone = m // 2
        sorted_indices = np.argsort(k_dim_mp)  # Sort low to high

        motif_groups = []  # Will store tuples of (distance, [idx, neighbor_idx])
        used_indices = np.zeros_like(k_dim_mp, dtype=bool)

        for idx in sorted_indices:
            if used_indices[idx]:
                continue

            neighbor_idx = k_dim_mpi[idx]

            if used_indices[neighbor_idx]:
                continue

            motif_dist = k_dim_mp[idx]
            motif_groups.append((motif_dist, [idx, neighbor_idx]))

            used_indices[
                max(0, idx - exclusion_zone) : min(
                    len(k_dim_mp), idx + exclusion_zone + 1
                )
            ] = True
            used_indices[
                max(0, neighbor_idx - exclusion_zone) : min(
                    len(k_dim_mp), neighbor_idx + exclusion_zone + 1
                )
            ] = True

            if len(motif_groups) == self.top_k:
                break

        logger.info(f"Top {len(motif_groups)} motif groups found.")

        # --- Save to CSV ---
        motif_data = []
        for i, (distance, indices) in enumerate(motif_groups):
            idx_1, idx_2 = indices
            time_1 = df_filtered.iloc[idx_1]["start_time"]
            time_2 = df_filtered.iloc[idx_2]["start_time"]
            motif_data.append(
                {
                    "motif_rank": i + 1,
                    "natural_k": natural_k,
                    "distance": distance,
                    "idx_1": idx_1,
                    "idx_2": idx_2,
                    "start_time_1": time_1,
                    "start_time_2": time_2,
                    "file_name": file_name,
                }
            )
        motif_df = pd.DataFrame(motif_data)
        motif_df.to_csv(self.output()["csv"].path, index=False)
        logger.info(f"Top motifs saved to {self.output()['csv'].path}")

        # --- Plotting ---
        plt.figure(figsize=(15, 5))
        linthresh = np.abs(components).mean() * 0.1
        if linthresh == 0:
            linthresh = 1e-5
        norm = SymLogNorm(
            linthresh=linthresh, vmin=np.min(components), vmax=np.max(components)
        )
        plt.imshow(components.T, aspect="auto", cmap="viridis", norm=norm)

        colors = plt.cm.cool(np.linspace(0, 1, len(motif_groups)))
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

        plt.title(f"Top {len(motif_groups)} Motifs (k={natural_k}) for {file_name}")
        plt.xlabel("Time (clip index)")
        plt.ylabel("SVD Component Index")
        plt.legend()
        Path(self.output()["plot"].path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.output()["plot"].path)
        plt.close()
