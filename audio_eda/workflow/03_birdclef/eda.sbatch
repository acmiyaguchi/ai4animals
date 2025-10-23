#!/bin/bash
#SBATCH -Jembed --account=gts-cpradalier7
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8 --mem-per-cpu=4G
#SBATCH -qembers -t 02:00:00
#SBATCH -o logs/eda/%j.log
#SBATCH --mail-type=END,FAIL --mail-user=amiyaguchi3@gatech.edu

set -eu
hostname;

# get some of the uv setup out of the way
PROJECT_ROOT=$HOME/ai4animals/audio_eda
cd ${PROJECT_ROOT}

export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT=${TMPDIR:-/tmp}/ai4animals/.venv
time uv sync --all-extras
set +x
source "${UV_PROJECT_ENVIRONMENT}/bin/activate"
set -x

python workflow/03_birdclef/eda.py
