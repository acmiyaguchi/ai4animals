import logging
from pathlib import Path

import luigi
from audio_eda.embed import EmbedAudio, AggregateEmbeddings

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("luigi-interface")


class Workflow(luigi.Task):
    def run(self):
        input_root = Path(
            "~/scratch/ai4animals/audio_eda/pacific_sounds/raw"
        ).expanduser()
        output_root = Path(
            "~/scratch/ai4animals/audio_eda/pacific_sounds/processed"
        ).expanduser()

        wav_files = list(input_root.glob("*.wav"))
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
