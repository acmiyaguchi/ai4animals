#!/bin/bash
#SBATCH -Jenwiki-dl --account=gts-cpradalier7
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t 2:00:00 -qembers -ologs/download/%j.log
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu
set -xeu

path=~/scratch/ai4animals/sensor_eda
mkdir -p $path
aria2c -c -j 3 -x 8 -i urls.txt -d $path

# unzip the codes
cd $path
unzip *.zip

