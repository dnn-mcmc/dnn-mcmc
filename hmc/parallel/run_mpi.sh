#!/bin/bash

#SBATCH -J make-moon
#SBATCH -e experiments/make_moon_10k.err
#SBATCH -o experiments/make_moon_10k.out
#SBATCH -n 64
#SBATCH --mem-per-cpu=500

export PATH=/opt/apps/rhel8/miniconda3/bin:$PATH
module load CUDA/10.2-rhel8
module load OpenMPI/4.0.5-rhel8
mpirun -n $SLURM_NTASKS python make_moon_mpi.py
