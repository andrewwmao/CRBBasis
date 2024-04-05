#!/bin/bash
#SBATCH -p cpu_short,cpu_medium,cpu_long
#SBATCH --mem=4G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=1-50
#SBATCH -t 00-12:00:00
#SBATCH --job-name=SimTrainingData
#SBATCH --mail-type=FAIL

set -Eeo pipefail

echo $HOSTNAME
echo $SLURM_ARRAY_TASK_ID

mkdir fingerprints
mkdir fingerprints_ograd
/gpfs/scratch/am4827/julia-1.10.0/bin/julia --heap-size-hint=${SLURM_MEM_PER_NODE}M sim.jl

wait
