#!/bin/bash
#SBATCH -p radiology,fn_short,fn_medium,fn_long
#SBATCH --mem=700G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH -t 00-12:00:00
#SBATCH --array=5
#SBATCH --job-name=CalcBasis
#SBATCH --mail-type=FAIL

set -Eeo pipefail

echo $HOSTNAME
echo $SLURM_ARRAY_TASK_ID
/gpfs/scratch/asslaj01/julia-1.10.0/bin/julia --threads=$SLURM_CPUS_PER_TASK --heap-size-hint=${SLURM_MEM_PER_NODE}M basis.jl $1

rm -f fingerprints/*.mat
rm -f fingerprints_ograd/*.mat

wait