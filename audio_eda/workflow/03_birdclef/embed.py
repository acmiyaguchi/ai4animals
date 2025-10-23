import logging
from pathlib import Path

import luigi
import random
from audio_eda.embed import EmbedAudio, AggregateEmbeddings

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


class Workflow(luigi.Task):
    def run(self):
        input_root = Path(
            "~/scratch/ai4animals/audio_eda/birdclef/raw/train_soundscapes"
        ).expanduser()
        output_root = Path(
            "~/scratch/ai4animals/audio_eda/birdclef/processed"
        ).expanduser()

        # randomly sample 10 files (with a seed)
        random.seed(42)
        ogg_files = random.sample(sorted(input_root.glob("*.ogg")), 10)

        logger.info(f"Workflow found {len(ogg_files)} .ogg files to process.")

        yield [
            EmbedAudio(input_path=str(p), output_root=str(output_root), clip_step=0.5)
            for p in ogg_files
        ]

        yield AggregateEmbeddings(
            input_root=str(output_root), output_root=str(output_root)
        )


def main():
    luigi.build([Workflow()], local_scheduler=True, workers=4)


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("spawn")
    main()
