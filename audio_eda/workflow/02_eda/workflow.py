"""
Main workflow orchestration for audio EDA.

This module coordinates tasks from representation and stumpy_analysis submodules.
"""

import logging
from pathlib import Path

import luigi

from representation import (
    SchemaAndCountsTask,
    PlotUMAPScatter,
    PlotScreePlot,
    PlotSVDHeatmap,
)
from stumpy_analysis import (
    PlotMatrixProfile,
    PlotMotifDetail,
    PlotTopDiscords,
    PlotTopMotifs,
)
from interpretation import (
    ExportTopMotifs,
    ExportTopDiscords,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("luigi-interface")


class Workflow(luigi.Task):
    def run(self):
        # this contains the wav files
        raw_root = Path(
            "~/scratch/ai4animals/audio_eda/pacific_sounds/raw"
        ).expanduser()
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
            # Note: PlotNaturalDimensionality is required by other tasks,
            # so it will run automatically.
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
            # Audio clipping and export tasks
            ExportTopMotifs(
                input_path=str(input_root),
                output_path=str(output_root),
                audio_root_path=str(raw_root),
                n_components=16,
                window_size=20,
                top_k=3,
            ),
            ExportTopDiscords(
                input_path=str(input_root),
                output_path=str(output_root),
                audio_root_path=str(raw_root),
                n_components=16,
                window_size=20,
                top_k=3,
            ),
        ]


def main():
    luigi.build([Workflow()], local_scheduler=True)


if __name__ == "__main__":
    main()
