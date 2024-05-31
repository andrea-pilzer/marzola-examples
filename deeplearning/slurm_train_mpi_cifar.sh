#!/bin/bash
#SBATCH --job-name=DDP_MPI     # name of the job
#SBATCH --partition=medium     # set the partition you need, e.g. disi-long
#SBATCH --time=00:10:00        # time, d-hh:mm:ss
#SBATCH --account=your_account # account to use to bill compute
#SBATCH --nodes=1              # nodes
#SBATCH --ntasks-per-node=1    # tasks
#SBATCH --gres=gpu:1           # gpus
#SBATCH --cpus-per-task=12     # cpu cores per task

GPUS_PER_NODE=8

echo "SLURM_NTASKS="$SLURM_NTASKS
NTASKS_PER_NODE=$((SLURM_NTASKS / SLURM_JOB_NUM_NODES))
echo "NTASKS_PER_NODE="$NTASKS_PER_NODE
export WORLD_SIZE=$((NTASKS_PER_NODE * SLURM_NNODES)) # $((GPUS_PER_NODE * SLURM_NNODES))
echo "WORLD_SIZE=$WORLD_SIZE"

### change 5-digit MASTER_PORT as you wish, slurm will raise Error if duplicated with others
### change WORLD_SIZE as gpus/node * num_nodes
export MASTER_PORT=11111

### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# you could add here a data copy to /scratch
# cp /path/to/your/data /scratch/your.username/.

### init virtual environment if needed
### I created a python3.9 -m venv pytorch in my $HOME folder
module load cuda/12.1
module load openmpi/4.1.5-cudaaware
source $HOME/pytorch/bin/activate

### the command to run
mpirun -np $WORLD_SIZE -x MASTER_ADDR=$MASTER_ADDR -x MASTER_PORT=$MASTER_PORT \
    python3 train_mpi_cifar.py --num_epochs 5

### remove the data to keep scratch tidy
### don't remove it if you are using it every day
### remove it when you are done with it
rm -r /scratch/apilzer

