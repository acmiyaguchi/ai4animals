#!/bin/bash
#SBATCH -Jembed --account=gts-cpradalier7
#SBATCH --nodes=1 --partition=gpu-l40s --gres=gpu:L40s
#SBATCH -qembers -t 02:00:00
#SBATCH -o logs/embed/%j.log
#SBATCH --mail-type=END,FAIL --mail-user=amiyaguchi3@gatech.edu

set -eu
hostname; nvidia-smi;

# get some of the uv setup out of the way
PROJECT_ROOT=$HOME/ai4animals/audio_eda
cd ${PROJECT_ROOT}

export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=${TMPDIR:-/tmp}/ai4animals/.venv
time uv sync --all-extras
set +x
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
set -x

python workflow/03_birdclef/embed.py
