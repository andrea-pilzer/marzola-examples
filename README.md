# marzola-examples
May 31st 2024 Tutorial

## Init with Slurm
The first example you find in this folder is simply submitting a job and printing for you the `nvidia-smi` output.
Try to play with it and check what the outputs look like
- if you run it multi-node?
- if you run it with multiple tasks?
The output will change a lot with these two simple tests.

## Init with a toy DL training
The second example under `deeplearning` folder is a simple cifar-10 training.
It is better to start from a distributed code from the beginning, if at any time you need to scale to many GPUs or nodes it will be ready.

Use tools that are available to scale easier, here I coded in pure PyTorch for teaching purposes, a non complete list in random order is DeepSpeed, PyTorch Lightning, Accelerate, etc.

Some important parts of the code are:
    - read slurm/mpi env variables
    - init distributed process
    - init distributed data sampler
    - wrap model in DDP class

Performance tricks:
    - mixed precision
    - dataloader workers
    - asyncronous copy (non_clocking=True)
    - cudnn benchmark
    - channel last
    - store data on node /scratch
    - batch size (power of 2)

## Environment
In order to execute these scripts two things are needed.
1. load cuda and openmpi modules
`module load cuda/12.1
module load openmpi/4.1.5-cudaaware`
2. create a python virtual environment
`python3.9 -m venv myenv
source activate myenv/bin/activate
pip install -r requirements.txt`

### Open issues if you have problems with the code!

