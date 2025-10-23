#!/bin/bash
#SBATCH -Jdownload --account=gts-cpradalier7
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=4 --mem-per-cpu=4G
#SBATCH -t 2:00:00 -qembers -ologs/download/%j.log
#SBATCH --mail-type=END,FAIL --mail-user=acmiyaguchi@gatech.edu
set -xeu

output_path=~/scratch/ai4animals/audio_eda/pacific_sounds/raw
mkdir -p $output_path
cd $output_path

# check _SUCCESS file
if [ -f _SUCCESS ]; then
    echo "Data already downloaded."
    cat filenames.txt
    tree $output_path -L 2
    exit 0
fi

# aws s3 ls --no-sign-request s3://pacific-sound-256khz-2025/09/

base=s3://pacific-sound-256khz-2025/09
# let's download 10 random ones for testing
echo "42" > /tmp/random_seed.txt
aws s3 ls --no-sign-request $base/ | shuf -n 10 | sort | awk '{print $4}' > filenames.txt
for file in $(cat filenames.txt); do
    aws s3 cp --no-sign-request $base/$file $output_path/
done

touch _SUCCESS

tree $output_path -L 2
