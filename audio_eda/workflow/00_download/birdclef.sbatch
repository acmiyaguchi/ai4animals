#!/bin/bash
#SBATCH -Jdownload --account=gts-cpradalier7
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t 2:00:00 -qembers -ologs/download/%j.log
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu
set -xeu

output_path=~/scratch/ai4animals/audio_eda/birdclef/raw
mkdir -p $output_path
cd $output_path

if [[ ! -f birdclef-2025.zip ]]; then
    echo "Downloading BirdCLEF 2025 dataset from Kaggle..."
    uv run kaggle competitions download -c birdclef-2025
fi

unzip -n birdclef-2025.zip