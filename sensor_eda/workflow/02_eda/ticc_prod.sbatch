#!/bin/bash
#SBATCH -Jticc --account=gts-cpradalier7
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=12 --mem-per-cpu=6G
#SBATCH -t 4:00:00 -q embers -ologs/ticc/%j.log
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu

source ~/ai4animals/.venv/bin/activate
set -xeu
cd ~/ai4animals/sensor_eda/workflow/02_eda
time python ticc_prod.py "$@"
tree ~/scratch/ai4animals/sensor_eda/ticc
