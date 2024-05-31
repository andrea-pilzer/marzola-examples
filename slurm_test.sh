#!/bin/bash

#SBATCH -N 2                               # Node count to be allocated for the job
#SBATCH --ntasks-per-node=1                # task per node
#SBATCH --cpus-per-task=1
#SBATCH --job-name=firstSlurmJob           # Job name
#SBATCH -o logs_%j.out                     # Outputs log file
#SBATCH -e logs_%j.err                     # Errors log file
#SBATCH -A your_account                    # billing account
#SBATCH -p short                           # partition
#SBATCH --time 00:01:00                    # allocated job execution time
#SBATCH --gres=gpu:2                       # gpus

srun bash test.sh  

