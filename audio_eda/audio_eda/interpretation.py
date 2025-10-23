import luigi
import pandas as pd
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from pydub import AudioSegment
from matplotlib.colors import SymLogNorm
from pathlib import Path
import logging

# Import the tasks we depend on
from .stumpy_analysis import PlotTopMotifs, PlotTopDiscords

logger = logging.getLogger(__name__)

# --- NEW: Define a target sample rate for human listening ---
TARGET_SR = 44100  # Standard audio CD sample rate


def _get_svd_components(input_path: Path, file_name: str, n_components: int):
    """
    Loads embeddings from the processed folder and computes SVD components.

    Args:
        input_path: Path to the processed folder containing embed.parquet
        file_name: Name of the audio file to extract embeddings for
        n_components: Number of PCA components to use

    Returns:
        SVD components as array of shape (n_components, n_samples)
    """
    # Read embeddings from the processed folder
    embed_parquet_path = input_path / "embed.parquet"
    df = pd.read_parquet(embed_parquet_path)
    df_filtered = df[df["file"] == file_name].sort_values("start_time")

    if df_filtered.empty:
        raise ValueError(f"No data found for file: {file_name} in {embed_parquet_path}")

    embeddings = np.stack(df_filtered["embedding"].values)

    # Compute PCA components
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    components = pca.fit_transform(embeddings)  # (n_samples, n_components)

    # Return (n_components, n_samples)
    return np.ascontiguousarray(components.T)


def _plot_spectrogram(y, sr, title, save_path):
    """Helper to generate and save a Mel spectrogram plot."""

    # --- MODIFIED: Use n_mels=128 (appropriate for 44.1kHz) ---
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    # --- End Modification ---

    S_db = librosa.power_to_db(S, ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _plot_svd_clip(svd_clip, title, save_path):
    """Helper to plot a heatmap of the SVD component clip."""

    # Calculate normalization (copied from PlotSVDHeatmap)
    linthresh = np.abs(svd_clip).mean() * 0.1
    if linthresh == 0:
        linthresh = 1e-5
    norm = SymLogNorm(linthresh=linthresh, vmin=np.min(svd_clip), vmax=np.max(svd_clip))

    plt.figure(figsize=(10, 4))
    plt.imshow(svd_clip, aspect="auto", cmap="viridis", norm=norm)
    plt.title(title)
    plt.xlabel("Time (clip index)")
    plt.ylabel("SVD Component Index")
    plt.colorbar(label="Component Value")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


class ExportTopMotifs(luigi.Task):
    """
    Clips audio, spectrograms, and SVD heatmaps for top motifs.
    """

    # Parameters from stumpy_analysis
    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)
    top_k: int = luigi.IntParameter(default=3)

    # New parameters for this task
    audio_root_path: str = luigi.Parameter()
    clip_hop_seconds: float = luigi.FloatParameter(default=0.5)
    suffix: str = luigi.Parameter(default="wav")

    def requires(self):
        # We depend on the CSV file from PlotTopMotifs
        return PlotTopMotifs(
            input_path=self.input_path,
            output_path=self.output_path,
            n_components=self.n_components,
            window_size=self.window_size,
            top_k=self.top_k,
        )

    def output(self):
        # This task creates many files, so we output a marker file
        return luigi.LocalTarget(
            Path(self.output_path) / "export" / "_motifs_exported.txt"
        )

    def run(self):
        motifs_df = pd.read_csv(self.input()["csv"].path)
        export_dir = Path(self.output_path) / "export" / "motifs"
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_name = motifs_df.iloc[0]["file_name"]
            svd_components_t = _get_svd_components(
                Path(self.input_path), file_name, self.n_components
            )
            audio_file_path = (Path(self.audio_root_path) / file_name).with_suffix(
                f".{self.suffix}"
            )

            # Log the sample rate
            sr_native = librosa.get_samplerate(audio_file_path)
            logger.info(f"Loading {audio_file_path.name} (Native SR: {sr_native} Hz)")

            y, sr = librosa.load(audio_file_path, sr=None)  # sr will be 256000
        except Exception as e:
            logger.error(f"Failed to load source data for {file_name}: {e}")
            raise

        clip_duration_sec = self.window_size * self.clip_hop_seconds

        for _, row in motifs_df.iterrows():
            rank = row["motif_rank"]
            logger.info(f"Exporting Motif Rank {rank}...")

            # --- Process Pair A ---
            idx_a = int(row["idx_1"])
            start_a = row["start_time_1"]
            self._export_clip(
                y,
                sr,  # Pass original SR (e.g., 256000)
                svd_components_t,
                idx_a,
                start_a,
                export_dir,
                f"motif_{rank}_a",
                clip_duration_sec,
            )

            # --- Process Pair B ---
            idx_b = int(row["idx_2"])
            start_b = row["start_time_2"]
            self._export_clip(
                y,
                sr,  # Pass original SR (e.g., 256000)
                svd_components_t,
                idx_b,
                start_b,
                export_dir,
                f"motif_{rank}_b",
                clip_duration_sec,
            )

        # Touch the marker file
        with self.output().open("w") as f:
            f.write("Motif export complete.")

    def _export_clip(self, y, sr, svd_t, idx, start_time, out_dir, basename, duration):
        """Internal helper to export audio, spec, and svd plot for one clip."""

        # 1. Extract Audio Clip (at original 256kHz sample rate)
        audio_clip_np = librosa.util.fix_length(
            y[int(start_time * sr) : int((start_time + duration) * sr)],
            size=int(duration * sr),
        )

        # --- NEW: Resample to human-audible range (e.g., 44.1kHz) ---
        # This also applies a low-pass filter, removing ultrasonic noise
        audio_clip_resampled = librosa.resample(
            audio_clip_np, orig_sr=sr, target_sr=TARGET_SR
        )
        # --- End New ---

        # --- MODIFIED: Normalize the *resampled* clip ---
        if audio_clip_resampled.size > 0:
            # 1. Remove DC offset
            audio_clip_resampled = audio_clip_resampled - np.mean(audio_clip_resampled)
            # 2. Normalize to peak value of 1.0 (Amplify)
            audio_clip_resampled = librosa.util.normalize(audio_clip_resampled)
        # --- End Modification ---

        # 2. Export WAV (using resampled data and target SR)
        audio_path_wav = out_dir / f"{basename}_audio.wav"
        sf.write(audio_path_wav, audio_clip_resampled, TARGET_SR)

        # 3. Export MP3 (using resampled data and target SR)
        try:
            audio_clip_int16 = (audio_clip_resampled * 32767).astype(np.int16)

            if audio_clip_int16.size == 0:
                audio_segment = AudioSegment.silent(
                    duration=duration * 1000, frame_rate=TARGET_SR
                )
            else:
                audio_segment = AudioSegment(
                    audio_clip_int16.tobytes(),
                    frame_rate=TARGET_SR,
                    sample_width=audio_clip_int16.dtype.itemsize,
                    channels=1,  # Assuming mono
                )

            audio_path_mp3 = out_dir / f"{basename}_audio.mp3"
            audio_segment.export(audio_path_mp3, format="mp3")

        except Exception as e:
            logger.error(
                f"Failed to export MP3 for {basename}. Is FFmpeg installed? Error: {e}"
            )

        # 4. Export Spectrogram (using resampled data and target SR)
        spec_path = out_dir / f"{basename}_spectrogram.png"
        _plot_spectrogram(
            audio_clip_resampled,
            TARGET_SR,
            f"{basename} (start: {start_time}s)",
            spec_path,
        )

        # 5. Export SVD Heatmap (this still uses the original SVD clip)
        svd_clip = svd_t[:, idx : idx + self.window_size]
        svd_path = out_dir / f"{basename}_svd_heatmap.png"
        _plot_svd_clip(svd_clip, f"{basename} (index: {idx})", svd_path)


class ExportTopDiscords(luigi.Task):
    """
    Clips audio, spectrograms, and SVD heatmaps for top discords.
    """

    # Parameters from stumpy_analysis
    input_path: str = luigi.Parameter()
    output_path: str = luigi.Parameter()
    n_components: int = luigi.IntParameter(default=16)
    window_size: int = luigi.IntParameter(default=20)
    top_k: int = luigi.IntParameter(default=3)

    # New parameters for this task
    audio_root_path: str = luigi.Parameter()
    clip_hop_seconds: float = luigi.FloatParameter(default=0.5)
    suffix: str = luigi.Parameter(default="wav")

    def requires(self):
        # We depend on the CSV file from PlotTopDiscords
        return PlotTopDiscords(
            input_path=self.input_path,
            output_path=self.output_path,
            n_components=self.n_components,
            window_size=self.window_size,
            top_k=self.top_k,
        )

    def output(self):
        # This task creates many files, so we output a marker file
        return luigi.LocalTarget(
            Path(self.output_path) / "export" / "_discords_exported.txt"
        )

    def run(self):
        discords_df = pd.read_csv(self.input()["csv"].path)
        export_dir = Path(self.output_path) / "export" / "discords"
        export_dir.mkdir(parents=True, exist_ok=True)

        try:
            file_name = discords_df.iloc[0]["file_name"]
            svd_components_t = _get_svd_components(
                Path(self.input_path), file_name, self.n_components
            )
            audio_file_path = (Path(self.audio_root_path) / file_name).with_suffix(
                f".{self.suffix}"
            )

            # Log the sample rate
            sr_native = librosa.get_samplerate(audio_file_path)
            logger.info(f"Loading {audio_file_path.name} (Native SR: {sr_native} Hz)")

            y, sr = librosa.load(audio_file_path, sr=None)  # sr will be 256000
        except Exception as e:
            logger.error(f"Failed to load source data for {file_name}: {e}")
            raise

        clip_duration_sec = self.window_size * self.clip_hop_seconds

        for _, row in discords_df.iterrows():
            rank = row["discord_rank"]
            logger.info(f"Exporting Discord Rank {rank}...")

            idx = int(row["idx"])
            start_time = row["start_time"]

            # Use the helper from the Motifs task
            ExportTopMotifs._export_clip(
                self,
                y,
                sr,  # Pass original SR (e.g., 256000)
                svd_components_t,
                idx,
                start_time,
                export_dir,
                f"discord_{rank}",
                clip_duration_sec,
            )

        # Touch the marker file
        with self.output().open("w") as f:
            f.write("Discord export complete.")
