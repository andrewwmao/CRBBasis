#!/bin/bash
#SBATCH --partition cpu_short,cpu_medium,cpu_long
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --array=0-10
#SBATCH -t 00-12:00:00
#SBATCH --job-name=CalcBasis
#SBATCH --mail-type=FAIL

export JULIA_BIN=/gpfs/scratch/asslaj01/julia-1.10.0/bin/julia

set -Eeo pipefail

echo $HOSTNAME
echo $SLURM_ARRAY_TASK_ID
$JULIA_BIN --threads=$SLURM_CPUS_PER_TASK --heap-size-hint=${SLURM_MEM_PER_NODE}M basis.jl

rm -f fingerprints/*.mat
rm -f fingerprints_ograd/*.mat

wait