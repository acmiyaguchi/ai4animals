import json
import pickle
from pathlib import Path
from typing import List, Optional

import luigi
import numpy as np
import pandas as pd
import timm
import torch
import torchvision.transforms as T
import typer
from wildlife_datasets import datasets, splits
from wildlife_tools.data import ImageDataset
from wildlife_tools.features import (
    AlikedExtractor,
    DeepFeatures,
    DiskExtractor,
    SiftExtractor,
    SuperPointExtractor,
)
from wildlife_tools.inference import KnnClassifier
from wildlife_tools.similarity import (
    CosineSimilarity,
    MatchLightGlue,
    SimilarityPipeline,
    WildFusion,
)
from wildlife_tools.similarity.calibration import (
    IsotonicCalibration,
    LogisticCalibration,
    reliability_diagram,
)

app = typer.Typer(help="Wildlife Re-identification Pipeline")


class BaseTask(luigi.Task):
    """Base task with configurable raw and processed data paths."""

    data_root = luigi.Parameter(default="~/scratch/ai4animals/visual_eda")

    @property
    def raw_path(self):
        return str(Path(self.data_root).expanduser())

    @property
    def processed_path(self):
        return str(Path(self.data_root).expanduser() / "processed")

    def output_path(self, subfolder, *filenames):
        """
        Helper to create LocalTargets in the processed_path.

        Args:
            subfolder: Subfolder within processed_path (e.g., 'metadata', 'splits', 'knn', 'wildfusion')
            *filenames: Path components for the output file
        """
        output_dir = Path(self.processed_path) / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)
        return luigi.LocalTarget(output_dir / Path(*filenames))


class LightGlueMatcher:
    """
    Wrapper for MatchLightGlue to make it compatible with SimilarityPipeline
    by extracting the score grid from the collector's output dictionary.
    """

    def __init__(self, features_name, threshold=0.5, device="cuda"):
        self.matcher = MatchLightGlue(
            features=features_name, device=device, batch_size=256
        )
        self.threshold = threshold

    def __call__(self, query_features, db_features, pairs=None):
        return self.matcher(query_features, db_features, pairs=pairs)


class DownloadData(BaseTask):
    """Downloads the WhaleSharkID dataset and saves the metadata object."""

    def run(self):
        data_dir = Path(self.raw_path) / "whalesharkid"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Only download if not already present
        if not (data_dir / "already_downloaded").exists():
            datasets.WhaleSharkID.get_data(root=str(data_dir))

        metadata = datasets.WhaleSharkID(root=str(data_dir))

        # Use Path to write binary pickle file
        with open(self.output().path, "wb") as f:
            pickle.dump(metadata, f)

    def output(self):
        return self.output_path("metadata", "whalesharkid.pkl")


class SplitData(BaseTask):
    """
    Creates train, calibration, and test splits using ClosedSetSplit.
    All three sets are disjoint sets of *images* but share *identities*.
    """

    train_split = luigi.FloatParameter(default=0.6)
    cal_split = luigi.FloatParameter(default=0.2)
    # Test split is inferred (1.0 - train_split - cal_split)

    def requires(self):
        return DownloadData(data_root=self.data_root)

    def run(self):
        with open(self.input().path, "rb") as f:
            metadata = pickle.load(f)
        df = metadata.df

        # Calculate the ratio for the first split (train+cal vs. test)
        # e.g., 0.6 + 0.2 = 0.8
        ratio_train_cal = self.train_split + self.cal_split

        # --- Split 1: (Train + Cal) vs (Test) ---
        # Use ClosedSetSplit to split images per identity [cite: 1505-1508, 1529]
        splitter1 = splits.ClosedSetSplit(ratio_train=ratio_train_cal)
        idx_train_cal, idx_test = splitter1.split(df)[0]

        # Create a new dataframe for the (Train + Cal) pool
        # .loc is crucial as split returns original indices [cite: 260]
        df_train_cal = df.loc[idx_train_cal]

        # --- Split 2: (Train) vs (Cal) ---
        # Calculate the ratio needed to split df_train_cal into train and cal
        # e.g., 0.6 / 0.8 = 0.75
        ratio_train_final = self.train_split / ratio_train_cal

        splitter2 = splits.ClosedSetSplit(ratio_train=ratio_train_final)
        # These indices are from df_train_cal, which are original indices
        idx_train, idx_cal = splitter2.split(df_train_cal)[0]

        # All idx_ variables now contain disjoint sets of original indices
        splits_data = {"idx_train": idx_train, "idx_cal": idx_cal, "idx_test": idx_test}

        with open(self.output().path, "wb") as f:
            pickle.dump(splits_data, f)

    def output(self):
        # I suggest changing the output filename to reflect the new method
        return self.output_path(
            "splits", f"closed_train{self.train_split}_cal{self.cal_split}.pkl"
        )


class ExtractFeatures(BaseTask):
    """
    (Basic Pipeline) Extracts features for train, calibration, and test sets.
    """

    extractor_name = luigi.Parameter(default="megadescriptor-t")
    train_split = luigi.FloatParameter(default=0.6)
    cal_split = luigi.FloatParameter(default=0.2)
    max_identities = luigi.IntParameter(default=None)
    max_images_per_identity = luigi.IntParameter(default=None)

    def requires(self):
        return {
            "data": DownloadData(data_root=self.data_root),
            "splits": SplitData(
                data_root=self.data_root,
                train_split=self.train_split,
                cal_split=self.cal_split,
            ),
        }

    def get_extractor_and_transform(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.extractor_name == "megadescriptor-t":
            model = timm.create_model(
                "hf-hub:BVRA/MegaDescriptor-T-224", num_classes=0, pretrained=True
            )
            extractor = DeepFeatures(
                model, device=device, num_workers=32, batch_size=512
            )
            transform = T.Compose(
                [
                    T.Resize([224, 224]),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        elif self.extractor_name == "dinov2-vits14":
            model = timm.create_model(
                "hf-hub:timm/vit_small_patch14_dinov2.lvd142m",
                num_classes=0,
                pretrained=True,
            )
            extractor = DeepFeatures(
                model, device=device, num_workers=32, batch_size=512
            )
            transform = T.Compose(
                [
                    T.Resize([518, 518]),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        elif self.extractor_name == "clip-vit-b-32":
            model = timm.create_model(
                "hf-hub:timm/vit_base_patch32_clip_224.openai",
                num_classes=0,
                pretrained=True,
            )
            extractor = DeepFeatures(
                model, device=device, num_workers=32, batch_size=512
            )
            transform = T.Compose(
                [
                    T.Resize([224, 224]),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
        elif self.extractor_name == "superpoint":
            extractor = SuperPointExtractor(device=device, num_workers=32)
            transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
        else:
            raise ValueError(f"Unknown extractor_name: {self.extractor_name}")
        return extractor, transform

    def run(self):
        with open(self.input()["data"].path, "rb") as f:
            metadata = pickle.load(f)
        with open(self.input()["splits"].path, "rb") as f:
            splits_idx = pickle.load(f)

        # Apply dataset filtering if specified
        df_train = metadata.df.iloc[splits_idx["idx_train"]]
        df_cal = metadata.df.iloc[splits_idx["idx_cal"]]
        df_test = metadata.df.iloc[splits_idx["idx_test"]]

        if self.max_identities is not None:
            # Get the most common identities from training set
            top_identities = (
                df_train[metadata.col_label]
                .value_counts()
                .head(self.max_identities)
                .index.tolist()
            )
            print(
                f"Limiting to {self.max_identities} most common identities: {len(top_identities)} found"
            )

            # Filter all datasets to only include these identities
            df_train = df_train[df_train[metadata.col_label].isin(top_identities)]
            df_cal = df_cal[df_cal[metadata.col_label].isin(top_identities)]
            df_test = df_test[df_test[metadata.col_label].isin(top_identities)]

            print(
                f"After identity filter - Train: {len(df_train)}, Cal: {len(df_cal)}, Test: {len(df_test)}"
            )

        if self.max_images_per_identity is not None:
            print(f"Limiting to {self.max_images_per_identity} images per identity")
            df_train = df_train.groupby(metadata.col_label).head(
                self.max_images_per_identity
            )
            df_cal = df_cal.groupby(metadata.col_label).head(
                self.max_images_per_identity
            )
            df_test = df_test.groupby(metadata.col_label).head(
                self.max_images_per_identity
            )

            print(
                f"After image limit - Train: {len(df_train)}, Cal: {len(df_cal)}, Test: {len(df_test)}"
            )

        extractor, transform = self.get_extractor_and_transform()

        datasets_out = {}
        features_out = {}
        for split, df in [("train", df_train), ("cal", df_cal), ("test", df_test)]:
            dataset = ImageDataset(df, metadata.root, transform=transform)
            datasets_out[f"labels_{split}"] = dataset.labels_string
            features_out[f"features_{split}"] = extractor(dataset)

        with open(self.output().path, "wb") as f:
            pickle.dump({**datasets_out, **features_out}, f)

    def output(self):
        suffix = ""
        if self.max_identities is not None:
            suffix += f"_maxid{self.max_identities}"
        if self.max_images_per_identity is not None:
            suffix += f"_maximg{self.max_images_per_identity}"
        return self.output_path("features", f"{self.extractor_name}{suffix}.pkl")


class CalculateSimilarity(BaseTask):
    """(Basic Pipeline) Calculates similarity matrices."""

    similarity_name = luigi.Parameter(default="cosine")
    extractor_name = luigi.Parameter(default="megadescriptor-t")
    train_split = luigi.FloatParameter(default=0.6)
    cal_split = luigi.FloatParameter(default=0.2)
    max_identities = luigi.IntParameter(default=None)
    max_images_per_identity = luigi.IntParameter(default=None)

    def requires(self):
        return ExtractFeatures(
            data_root=self.data_root,
            extractor_name=self.extractor_name,
            train_split=self.train_split,
            cal_split=self.cal_split,
            max_identities=self.max_identities,
            max_images_per_identity=self.max_images_per_identity,
        )

    def run(self):
        with open(self.input().path, "rb") as f:
            features = pickle.load(f)

        if self.similarity_name == "cosine":
            sim_func = CosineSimilarity()
            cal_sim = sim_func(features["features_cal"], features["features_train"])
            eval_sim = sim_func(features["features_test"], features["features_train"])
        elif self.similarity_name == "lightglue":
            matcher = LightGlueMatcher(features_name=self.extractor_name, device="cuda")
            cal_sim = matcher(features["features_cal"], features["features_train"])
            eval_sim = matcher(features["features_test"], features["features_train"])
        else:
            raise ValueError(f"Unknown similarity_name: {self.similarity_name}")

        with open(self.output().path, "wb") as f:
            pickle.dump({"cal_sim": cal_sim, "eval_sim": eval_sim}, f)

    def output(self):
        suffix = ""
        if self.max_identities is not None:
            suffix += f"_maxid{self.max_identities}"
        if self.max_images_per_identity is not None:
            suffix += f"_maximg{self.max_images_per_identity}"
        return self.output_path(
            "similarity",
            f"ext-{self.extractor_name}_sim-{self.similarity_name}{suffix}.pkl",
        )


class EvaluateModel(BaseTask):
    """(Basic Pipeline) Runs k-NN classification with optional calibration."""

    k = luigi.IntParameter(default=1)
    calibration_name = luigi.Parameter(default="none")
    similarity_name = luigi.Parameter(default="cosine")
    extractor_name = luigi.Parameter(default="megadescriptor-t")
    train_split = luigi.FloatParameter(default=0.6)
    cal_split = luigi.FloatParameter(default=0.2)
    max_identities = luigi.IntParameter(default=None)
    max_images_per_identity = luigi.IntParameter(default=None)

    def _validate_combination(self):
        """Validate that extractor and similarity are compatible."""
        # Global extractors (produce single feature vector)
        global_extractors = ["megadescriptor-t"]
        # Local extractors (produce keypoints + descriptors)
        local_extractors = ["superpoint", "disk", "aliked", "sift"]

        # Global similarities (work with single vectors)
        global_similarities = ["cosine"]
        # Local similarities (work with keypoint sets)
        local_similarities = ["lightglue"]

        is_global_extractor = self.extractor_name in global_extractors
        is_local_extractor = self.extractor_name in local_extractors
        is_global_similarity = self.similarity_name in global_similarities
        is_local_similarity = self.similarity_name in local_similarities

        if is_global_extractor and is_local_similarity:
            raise ValueError(
                f"Invalid combination: {self.extractor_name} (global extractor) "
                f"cannot be used with {self.similarity_name} (local similarity). "
                f"Use a global similarity like 'cosine' instead."
            )

        if is_local_extractor and is_global_similarity:
            raise ValueError(
                f"Invalid combination: {self.extractor_name} (local extractor) "
                f"cannot be used with {self.similarity_name} (global similarity). "
                f"Use a local similarity like 'lightglue' instead."
            )

    def requires(self):
        self._validate_combination()
        return {
            "features": ExtractFeatures(
                data_root=self.data_root,
                extractor_name=self.extractor_name,
                train_split=self.train_split,
                cal_split=self.cal_split,
                max_identities=self.max_identities,
                max_images_per_identity=self.max_images_per_identity,
            ),
            "similarity": CalculateSimilarity(
                data_root=self.data_root,
                extractor_name=self.extractor_name,
                similarity_name=self.similarity_name,
                train_split=self.train_split,
                cal_split=self.cal_split,
                max_identities=self.max_identities,
                max_images_per_identity=self.max_images_per_identity,
            ),
        }

    def run(self):
        with open(self.input()["features"].path, "rb") as f:
            features = pickle.load(f)
        with open(self.input()["similarity"].path, "rb") as f:
            similarity = pickle.load(f)

        eval_sim = similarity["eval_sim"]
        cal_sim = similarity["cal_sim"]

        if self.calibration_name != "none":
            if self.calibration_name == "isotonic":
                calibrator = IsotonicCalibration()
            elif self.calibration_name == "logistic":
                calibrator = LogisticCalibration()
            else:
                raise ValueError(f"Unknown calibration_name: {self.calibration_name}")

            hits = features["labels_cal"][:, None] == features["labels_train"][None, :]
            calibrator.fit(cal_sim.flatten(), hits.flatten())
            eval_sim = calibrator.predict(eval_sim)

        # Calculate accuracy
        classifier = KnnClassifier(k=self.k, database_labels=features["labels_train"])
        predictions = classifier(eval_sim)
        accuracy = np.mean(features["labels_test"] == predictions)

        # Calculate ECE using reliability_diagram
        hits_test = (
            features["labels_test"][:, None] == features["labels_train"][None, :]
        )
        ece = reliability_diagram(
            eval_sim.flatten(), hits_test.flatten(), skip_plot=True
        )

        # Prepare results as JSON
        results = {
            "extractor": self.extractor_name,
            "similarity": self.similarity_name,
            "calibration": self.calibration_name,
            "k": self.k,
            "train_split": self.train_split,
            "cal_split": self.cal_split,
            "accuracy": float(accuracy),
            "ece": float(ece),
        }

        with self.output().open("w") as f:
            json.dump(results, f, indent=2)

    def output(self):
        suffix = ""
        if self.max_identities is not None:
            suffix += f"_maxid{self.max_identities}"
        if self.max_images_per_identity is not None:
            suffix += f"_maximg{self.max_images_per_identity}"
        filename = f"ext-{self.extractor_name}_sim-{self.similarity_name}_cal-{self.calibration_name}_k-{self.k}{suffix}.json"
        return self.output_path("results_knn", filename)


class EvaluateWildFusion(BaseTask):
    """(Complex Pipeline) Runs the full WildFusion pipeline."""

    k = luigi.IntParameter(default=1)
    B = luigi.IntParameter(default=100)
    priority_extractor = luigi.Parameter(default="megadescriptor-t")
    calibrated_extractors = luigi.ListParameter(default=["sift-lightglue"])
    calibration_name = luigi.Parameter(default="isotonic")
    train_split = luigi.FloatParameter(default=0.6)
    cal_split = luigi.FloatParameter(default=0.2)
    max_identities = luigi.IntParameter(default=None)
    max_images_per_identity = luigi.IntParameter(default=None)

    def requires(self):
        return {
            "data": DownloadData(data_root=self.data_root),
            "splits": SplitData(
                data_root=self.data_root,
                train_split=self.train_split,
                cal_split=self.cal_split,
            ),
        }

    def get_pipeline(self, extractor_name, calibration_obj):
        """Helper factory to create SimilarityPipeline objects."""
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if extractor_name == "megadescriptor-t":
            model = timm.create_model(
                "hf-hub:BVRA/MegaDescriptor-T-224", num_classes=0, pretrained=True
            )
            extractor = DeepFeatures(
                model, device=device, num_workers=32, batch_size=512
            )
            transform = T.Compose(
                [
                    T.Resize([224, 224]),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            matcher = CosineSimilarity()
        elif extractor_name == "superpoint-lightglue":
            extractor = SuperPointExtractor(device=device, num_workers=32)
            transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
            matcher = LightGlueMatcher(features_name="superpoint", device=device)
        elif extractor_name == "sift-lightglue":
            extractor = SiftExtractor()
            transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
            matcher = LightGlueMatcher(features_name="sift", device=device)
        elif extractor_name == "aliked-lightglue":
            extractor = AlikedExtractor(device=device, num_workers=32)
            transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
            matcher = LightGlueMatcher(features_name="aliked", device=device)
        elif extractor_name == "disk-lightglue":
            extractor = DiskExtractor(device=device, num_workers=32)
            transform = T.Compose([T.Resize([224, 224]), T.ToTensor()])
            matcher = LightGlueMatcher(features_name="disk", device=device)
        else:
            raise ValueError(f"Unknown extractor_name: {extractor_name}")

        return SimilarityPipeline(
            matcher=matcher,
            extractor=extractor,
            calibration=calibration_obj,
            transform=transform,
        )

    def run(self):
        with open(self.input()["data"].path, "rb") as f:
            metadata = pickle.load(f)
        with open(self.input()["splits"].path, "rb") as f:
            splits_idx = pickle.load(f)

        # 1. Define pipelines
        priority_pipeline = self.get_pipeline(
            self.priority_extractor, calibration_obj=None
        )

        calibrated_pipelines_list = []
        for extractor_name in self.calibrated_extractors:
            if self.calibration_name == "isotonic":
                calibration_obj = IsotonicCalibration()
            elif self.calibration_name == "logistic":
                calibration_obj = LogisticCalibration()
            else:  # 'none'
                calibration_obj = None

            calibrated_pipelines_list.append(
                self.get_pipeline(extractor_name, calibration_obj=calibration_obj)
            )

        # 2. Create datasets (WildFusion takes datasets, not pre-computed features)
        df_train = metadata.df.iloc[splits_idx["idx_train"]]
        df_cal = metadata.df.iloc[splits_idx["idx_cal"]]
        df_test = metadata.df.iloc[splits_idx["idx_test"]]

        # Optionally limit to a subset of identities for faster training
        if self.max_identities is not None:
            # Get the most common identities from training set
            top_identities = (
                df_train[metadata.col_label]
                .value_counts()
                .head(self.max_identities)
                .index.tolist()
            )
            print(
                f"Limiting to {self.max_identities} most common identities: {len(top_identities)} found"
            )

            # Filter all datasets to only include these identities
            df_train = df_train[df_train[metadata.col_label].isin(top_identities)]
            df_cal = df_cal[df_cal[metadata.col_label].isin(top_identities)]
            df_test = df_test[df_test[metadata.col_label].isin(top_identities)]

            print(
                f"After identity filter - Train: {len(df_train)}, Cal: {len(df_cal)}, Test: {len(df_test)}"
            )

        # Optionally limit images per identity to reduce dataset size
        if self.max_images_per_identity is not None:
            print(f"Limiting to {self.max_images_per_identity} images per identity")
            df_train = df_train.groupby(metadata.col_label).head(
                self.max_images_per_identity
            )
            df_cal = df_cal.groupby(metadata.col_label).head(
                self.max_images_per_identity
            )
            df_test = df_test.groupby(metadata.col_label).head(
                self.max_images_per_identity
            )

            print(
                f"After image limit - Train: {len(df_train)}, Cal: {len(df_cal)}, Test: {len(df_test)}"
            )

        dataset_train = ImageDataset(df_train, metadata.root)
        dataset_cal = ImageDataset(df_cal, metadata.root)
        dataset_test = ImageDataset(df_test, metadata.root)

        # 3. Instantiate WildFusion
        wildfusion = WildFusion(
            calibrated_pipelines=calibrated_pipelines_list,
            priority_pipeline=priority_pipeline,
        )

        # 4. Fit Calibration (if specified)
        if self.calibration_name != "none":
            print(f"Fitting {self.calibration_name} calibration models...")
            wildfusion.fit_calibration(dataset_cal, dataset_train)

        # 5. Run Inference
        print(f"Running WildFusion inference with B={self.B}...")
        similarity = wildfusion(dataset_test, dataset_train, B=self.B)

        # 6. Evaluate
        classifier = KnnClassifier(
            k=self.k, database_labels=dataset_train.labels_string
        )
        # 6. Evaluate
        classifier = KnnClassifier(
            k=self.k, database_labels=dataset_train.labels_string
        )
        predictions = classifier(similarity)
        accuracy = np.mean(dataset_test.labels_string == predictions)

        # Calculate ECE using reliability_diagram
        hits = (
            dataset_test.labels_string[:, None] == dataset_train.labels_string[None, :]
        )
        # Convert to float32 to avoid pandas float16 issue
        ece = reliability_diagram(
            similarity.astype(np.float32).flatten(), hits.flatten(), skip_plot=True
        )

        # Prepare results as JSON
        results = {
            "priority_extractor": self.priority_extractor,
            "calibrated_extractors": list(self.calibrated_extractors),
            "calibration": self.calibration_name,
            "k": self.k,
            "B": self.B,
            "train_split": self.train_split,
            "cal_split": self.cal_split,
            "accuracy": float(accuracy),
            "ece": float(ece),
        }

        with self.output().open("w") as f:
            json.dump(results, f, indent=2)

    def output(self):
        priority_short = self.priority_extractor.replace("megadescriptor-t", "mega")
        calibrated_short = "_".join(
            [
                c.replace("superpoint-lightglue", "sp-lg")
                for c in self.calibrated_extractors
            ]
        )
        filename = f"p-{priority_short}_c-{calibrated_short}_cal-{self.calibration_name}_k-{self.k}_B-{self.B}.json"
        return self.output_path("results_wildfusion", filename)


@app.command()
def validate(
    extractor: str = typer.Argument(..., help="Feature extractor name"),
    similarity: str = typer.Argument(..., help="Similarity metric"),
):
    """Check if an extractor-similarity combination is valid."""
    # Global extractors (produce single feature vector)
    global_extractors = ["megadescriptor-t"]
    # Local extractors (produce keypoints + descriptors)
    local_extractors = ["superpoint", "disk", "aliked", "sift"]

    # Global similarities (work with single vectors)
    global_similarities = ["cosine"]
    # Local similarities (work with keypoint sets)
    local_similarities = ["lightglue"]

    is_global_extractor = extractor in global_extractors
    is_local_extractor = extractor in local_extractors
    is_global_similarity = similarity in global_similarities
    is_local_similarity = similarity in local_similarities

    if is_global_extractor and is_local_similarity:
        typer.secho(
            f"❌ INVALID: {extractor} (global) cannot use {similarity} (local)",
            fg=typer.colors.RED,
        )
        typer.echo(
            f"\nValid similarities for {extractor}: {', '.join(global_similarities)}"
        )
        raise typer.Exit(1)

    if is_local_extractor and is_global_similarity:
        typer.secho(
            f"❌ INVALID: {extractor} (local) cannot use {similarity} (global)",
            fg=typer.colors.RED,
        )
        typer.echo(
            f"\nValid similarities for {extractor}: {', '.join(local_similarities)}"
        )
        raise typer.Exit(1)

    typer.secho(f"✓ VALID: {extractor} + {similarity}", fg=typer.colors.GREEN)


@app.command()
def list_combinations():
    """Show all valid extractor-similarity combinations."""
    typer.echo("Valid Extractor-Similarity Combinations:\n")

    typer.secho(
        "Global Extractors (single feature vector):", fg=typer.colors.CYAN, bold=True
    )
    typer.echo("  • megadescriptor-t")
    typer.echo("    Compatible similarities: cosine")
    typer.echo()

    typer.secho(
        "Local Extractors (keypoints + descriptors):", fg=typer.colors.CYAN, bold=True
    )
    typer.echo("  • superpoint")
    typer.echo("  • disk")
    typer.echo("  • aliked")
    typer.echo("  • sift")
    typer.echo("    Compatible similarities: lightglue")
    typer.echo()

    typer.secho("Examples:", fg=typer.colors.YELLOW, bold=True)
    typer.echo(
        "  python inference.py knn --extractor megadescriptor-t --similarity cosine"
    )
    typer.echo(
        "  python inference.py knn --extractor superpoint --similarity lightglue"
    )


@app.command()
def knn(
    data_root: str = typer.Option(
        "~/scratch/ai4animals/visual_eda",
        "--data-root",
        "-d",
        help="Root directory for data storage",
    ),
    extractor: str = typer.Option(
        "megadescriptor-t", "--extractor", "-e", help="Feature extractor name"
    ),
    similarity: str = typer.Option(
        "cosine", "--similarity", "-s", help="Similarity metric"
    ),
    calibration: str = typer.Option(
        "isotonic",
        "--calibration",
        "-c",
        help="Calibration method (none/isotonic/logistic)",
    ),
    k: int = typer.Option(1, "--k", help="Number of nearest neighbors"),
    train_split: float = typer.Option(0.6, "--train", help="Train split ratio"),
    cal_split: float = typer.Option(0.2, "--cal", help="Calibration split ratio"),
    max_identities: int = typer.Option(
        None,
        "--max-identities",
        help="Limit to N most common identities (for faster training)",
    ),
    max_images: int = typer.Option(
        None,
        "--max-images",
        help="Limit to N images per identity (reduces dataset size)",
    ),
):
    """Run basic k-NN classification pipeline."""

    # Display configuration
    typer.echo("=" * 80)
    typer.echo("k-NN Pipeline Configuration")
    typer.echo("=" * 80)
    typer.echo()
    typer.echo(f"Data Root:            {data_root}")
    typer.echo(f"Extractor:            {extractor}")
    typer.echo(f"Similarity:           {similarity}")
    typer.echo(f"Calibration Method:   {calibration}")
    typer.echo(f"k-NN:                 {k}")
    typer.echo(f"Train Split:          {train_split}")
    typer.echo(f"Calibration Split:    {cal_split}")

    if max_identities is not None or max_images is not None:
        typer.secho("\nDataset Limits (FAST MODE):", fg=typer.colors.YELLOW)
        if max_identities is not None:
            typer.secho(
                f"  Max Identities:     {max_identities}", fg=typer.colors.YELLOW
            )
        if max_images is not None:
            typer.secho(f"  Max Images/Identity: {max_images}", fg=typer.colors.YELLOW)

    typer.echo()
    typer.echo("=" * 80)
    typer.echo()

    luigi.build(
        [
            EvaluateModel(
                data_root=data_root,
                extractor_name=extractor,
                similarity_name=similarity,
                calibration_name=calibration,
                k=k,
                train_split=train_split,
                cal_split=cal_split,
                max_identities=max_identities,
                max_images_per_identity=max_images,
            )
        ],
        local_scheduler=True,
    )


@app.command()
def fusion(
    data_root: str = typer.Option(
        "~/scratch/ai4animals/visual_eda",
        "--data-root",
        "-d",
        help="Root directory for data storage",
    ),
    priority: str = typer.Option(
        "megadescriptor-t", "--priority", "-p", help="Priority extractor name"
    ),
    calibrated: str = typer.Option(
        "sift-lightglue",
        "--calibrated",
        help="Comma-separated calibrated extractors",
    ),
    calibration: str = typer.Option(
        "isotonic",
        "--calibration",
        "-c",
        help="Calibration method (none/isotonic/logistic)",
    ),
    k: int = typer.Option(1, "--k", help="Number of nearest neighbors"),
    budget: int = typer.Option(100, "--budget", "-b", help="Shortlisting budget B"),
    train_split: float = typer.Option(0.6, "--train", help="Train split ratio"),
    cal_split: float = typer.Option(0.2, "--cal", help="Calibration split ratio"),
    max_identities: int = typer.Option(
        None,
        "--max-identities",
        help="Limit to N most common identities (for faster training)",
    ),
    max_images_per_identity: int = typer.Option(
        None,
        "--max-images",
        help="Limit to N images per identity (reduces dataset size)",
    ),
):
    """Run WildFusion multi-pipeline system."""
    calibrated_extractors = [x.strip() for x in calibrated.split(",")]

    # Print configuration
    typer.secho("\n" + "=" * 80, fg=typer.colors.CYAN)
    typer.secho("WildFusion Pipeline Configuration", fg=typer.colors.CYAN, bold=True)
    typer.secho("=" * 80 + "\n", fg=typer.colors.CYAN)
    typer.echo(f"Data Root:            {data_root}")
    typer.echo(f"Priority Extractor:   {priority}")
    typer.echo(f"Calibrated Extractors: {', '.join(calibrated_extractors)}")
    typer.echo(f"Calibration Method:   {calibration}")
    typer.echo(f"k-NN:                 {k}")
    typer.echo(f"Shortlisting Budget:  {budget}")
    typer.echo(f"Train Split:          {train_split}")
    typer.echo(f"Calibration Split:    {cal_split}")
    typer.echo(f"Test Split:           {1.0 - train_split - cal_split}")
    if max_identities or max_images_per_identity:
        typer.secho("\nDataset Limits (FAST MODE):", fg=typer.colors.YELLOW, bold=True)
        if max_identities:
            typer.echo(f"  Max Identities:     {max_identities}")
        if max_images_per_identity:
            typer.echo(f"  Max Images/Identity: {max_images_per_identity}")
    typer.secho("\n" + "=" * 80 + "\n", fg=typer.colors.CYAN)

    luigi.build(
        [
            EvaluateWildFusion(
                data_root=data_root,
                priority_extractor=priority,
                calibrated_extractors=calibrated_extractors,
                calibration_name=calibration,
                k=k,
                B=budget,
                train_split=train_split,
                cal_split=cal_split,
                max_identities=max_identities,
                max_images_per_identity=max_images_per_identity,
            )
        ],
        local_scheduler=True,
    )


@app.command()
def search_knn(
    data_root: str = typer.Option(
        "~/scratch/ai4animals/visual_eda",
        "--data-root",
        "-d",
        help="Root directory for data storage",
    ),
    extractors: str = typer.Option(
        "megadescriptor-t,dinov2-vits14,clip-vit-b-32",
        "--extractors",
        "-e",
        help="Comma-separated extractors",
    ),
    similarities: str = typer.Option(
        "cosine", "--similarities", "-s", help="Comma-separated similarities"
    ),
    calibrations: str = typer.Option(
        "isotonic",
        "--calibrations",
        "-c",
        help="Comma-separated calibrations",
    ),
    k_values: str = typer.Option(
        "1,3,5,10", "--k-values", "-k", help="Comma-separated k values"
    ),
    train_split: float = typer.Option(0.6, "--train", help="Train split ratio"),
    cal_split: float = typer.Option(0.2, "--cal", help="Calibration split ratio"),
    max_identities: int = typer.Option(
        30, "--max-identities", help="Limit to N most common identities"
    ),
    max_images: int = typer.Option(
        3, "--max-images", help="Limit to N images per identity"
    ),
):
    """Run hyperparameter search for k-NN pipeline with fixed dataset limits."""
    extractors_list = [x.strip() for x in extractors.split(",")]
    similarities_list = [x.strip() for x in similarities.split(",")]
    calibrations_list = [x.strip() for x in calibrations.split(",")]
    k_list = [int(x.strip()) for x in k_values.split(",")]

    typer.secho(
        f"\nk-NN Hyperparameter Search (Fixed: {max_identities} identities × {max_images} images, calibration=isotonic)",
        fg=typer.colors.CYAN,
        bold=True,
    )

    tasks = []
    for extractor in extractors_list:
        for similarity in similarities_list:
            for calibration in calibrations_list:
                for k in k_list:
                    tasks.append(
                        EvaluateModel(
                            data_root=data_root,
                            extractor_name=extractor,
                            similarity_name=similarity,
                            calibration_name=calibration,
                            k=k,
                            train_split=train_split,
                            cal_split=cal_split,
                            max_identities=max_identities,
                            max_images_per_identity=max_images,
                        )
                    )

    typer.echo(f"Running {len(tasks)} k-NN experiments...")
    luigi.build(tasks, local_scheduler=True)


@app.command()
def search_fusion(
    data_root: str = typer.Option(
        "~/scratch/ai4animals/visual_eda",
        "--data-root",
        "-d",
        help="Root directory for data storage",
    ),
    priorities: str = typer.Option(
        "megadescriptor-t",
        "--priorities",
        "-p",
        help="Comma-separated priority extractors",
    ),
    calibrated: str = typer.Option(
        "aliked-lightglue,superpoint-lightglue,sift-lightglue,disk-lightglue",
        "--calibrated",
        help="Comma-separated calibrated extractors",
    ),
    calibrations: str = typer.Option(
        "isotonic",
        "--calibrations",
        "-c",
        help="Comma-separated calibrations",
    ),
    k_values: str = typer.Option(
        "1", "--k-values", "-k", help="Comma-separated k values"
    ),
    budgets: str = typer.Option(
        "20", "--budgets", "-b", help="Comma-separated budget values"
    ),
    train_split: float = typer.Option(0.6, "--train", help="Train split ratio"),
    cal_split: float = typer.Option(0.2, "--cal", help="Calibration split ratio"),
    max_identities: int = typer.Option(
        30, "--max-identities", help="Limit to N most common identities"
    ),
    max_images: int = typer.Option(
        3, "--max-images", help="Limit to N images per identity"
    ),
):
    """Run hyperparameter search for WildFusion pipeline with fixed dataset limits."""
    priorities_list = [x.strip() for x in priorities.split(",")]
    calibrated_list = [x.strip() for x in calibrated.split(",")]
    calibrations_list = [x.strip() for x in calibrations.split(",")]
    k_list = [int(x.strip()) for x in k_values.split(",")]
    budgets_list = [int(x.strip()) for x in budgets.split(",")]

    typer.secho(
        f"\nWildFusion Hyperparameter Search (Fixed: {max_identities} identities × {max_images} images, calibration=isotonic)",
        fg=typer.colors.CYAN,
        bold=True,
    )

    tasks = []
    for priority in priorities_list:
        for calibrated_extractor in calibrated_list:
            for calibration in calibrations_list:
                for k in k_list:
                    for budget in budgets_list:
                        tasks.append(
                            EvaluateWildFusion(
                                data_root=data_root,
                                priority_extractor=priority,
                                calibrated_extractors=[calibrated_extractor],
                                calibration_name=calibration,
                                k=k,
                                B=budget,
                                train_split=train_split,
                                cal_split=cal_split,
                                max_identities=max_identities,
                                max_images_per_identity=max_images,
                            )
                        )

    typer.echo(f"Running {len(tasks)} WildFusion experiments...")
    luigi.build(tasks, local_scheduler=True)


@app.command()
def summarize(
    data_root: str = typer.Option(
        "~/scratch/ai4animals/visual_eda",
        "--data-root",
        "-d",
        help="Root directory for data storage",
    ),
    pipeline: str = typer.Option(
        "all", "--pipeline", "-p", help="Pipeline to summarize (knn/fusion/all)"
    ),
    sort_by: str = typer.Option(
        "accuracy", "--sort-by", "-s", help="Sort results by (accuracy/ece)"
    ),
    fmt: str = typer.Option("table", "--format", "-f", help="Output format: table|csv"),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Optional path to write the summarized table (CSV if --format=csv)",
    ),
):
    """Summarize results from JSON files as a pandas table."""
    # Expand '~' in data_root for correct path resolution
    processed_path = Path(data_root).expanduser() / "processed"

    results = []

    if pipeline in ["knn", "all"]:
        knn_results_dir = processed_path / "results_knn"
        if knn_results_dir.exists():
            for json_file in knn_results_dir.glob("*.json"):
                with open(json_file) as f:
                    result = json.load(f)
                    result["pipeline"] = "knn"
                    result["file"] = json_file.name
                    results.append(result)

    if pipeline in ["fusion", "all"]:
        fusion_results_dir = processed_path / "results_wildfusion"
        if fusion_results_dir.exists():
            for json_file in fusion_results_dir.glob("*.json"):
                with open(json_file) as f:
                    result = json.load(f)
                    result["pipeline"] = "wildfusion"
                    result["file"] = json_file.name
                    results.append(result)

    if not results:
        typer.secho("No results found!", fg=typer.colors.RED)
        return

    # Build pandas DataFrame and harmonize columns across pipelines
    df = pd.DataFrame(results)
    if df.empty:
        typer.secho("No results found!", fg=typer.colors.RED)
        return

    # Harmonize columns
    df["model"] = df.apply(
        lambda r: r.get("extractor")
        if r.get("pipeline") == "knn"
        else r.get("priority_extractor"),
        axis=1,
    )
    df["sim"] = df.apply(
        lambda r: r.get("similarity") if r.get("pipeline") == "knn" else "-",
        axis=1,
    )
    df["calibrated"] = df.apply(
        lambda r: ",".join(r.get("calibrated_extractors", []))
        if r.get("pipeline") == "wildfusion"
        else "-",
        axis=1,
    )

    # Ensure numeric
    for col in ["accuracy", "ece", "k", "B"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sorting: accuracy desc, ece asc
    ascending = False if sort_by == "accuracy" else True
    if sort_by in df.columns:
        df = df.sort_values(by=sort_by, ascending=ascending, kind="mergesort")

    # Select columns to display
    cols = [
        "pipeline",
        "model",
        "sim",
        "calibrated",
        "calibration",
        "k",
        "B",
        "accuracy",
        "ece",
        "file",
    ]
    cols = [c for c in cols if c in df.columns]
    view = df[cols].copy()

    # Pretty percentages
    if "accuracy" in view.columns:
        view["accuracy"] = (view["accuracy"] * 100).map(lambda x: f"{x:.2f}%")
    if "ece" in view.columns:
        view["ece"] = (view["ece"] * 100).map(lambda x: f"{x:.2f}%")

    if fmt == "csv":
        if output is None:
            typer.echo(view.to_csv(index=False))
        else:
            out_path = Path(output).expanduser()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            view.to_csv(out_path, index=False)
            typer.secho(f"Wrote CSV to {out_path}", fg=typer.colors.GREEN)
    else:
        typer.secho(f"\n{'=' * 80}", fg=typer.colors.CYAN)
        typer.secho(
            f"Results Summary ({len(view)} experiments)",
            fg=typer.colors.CYAN,
            bold=True,
        )
        typer.secho(f"{'=' * 80}\n", fg=typer.colors.CYAN)
        typer.echo(view.to_string(index=False))


if __name__ == "__main__":
    app()
