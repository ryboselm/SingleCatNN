#!/bin/sh
#
#Submit Script for neural network training for Cloud Moment Analysis
#
#SBATCH --account=glab
#SBATCH --job-name=4_moment_set_random_trials
#SBATCH -c 32
#SBATCH --time=16:00:00
#SBATCH --mem-per-cpu=5G
#SBATCH --mail-user=raa2218@columbia.edu

echo "Starting Python Job!"

# Setup Environment
module load anaconda
source activate /burg/glab/users/raa2218/envs/geo_scipy_torch

export XDG_RUNTIME_DIR=""

python 4_moment_set_random_trials.py 

#End of script
