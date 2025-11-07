#!/bin/bash
#SBATCH -Jdownload --account=gts-cpradalier7
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=8G
#SBATCH -t 2:00:00 -qembers -ologs/download/%j.log
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu
set -xeu

output_path=$HOME/scratch/ai4animals/visual_eda/mabe/raw
mkdir -p $output_path
cd $output_path

if [[ ! -f MABe-mouse-behavior-detection.zip ]]; then
    echo "Downloading MABe mouse behavior detection dataset from Kaggle..."
    kaggle competitions download -c MABe-mouse-behavior-detection
fi

unzip -n MABe-mouse-behavior-detection.zip
