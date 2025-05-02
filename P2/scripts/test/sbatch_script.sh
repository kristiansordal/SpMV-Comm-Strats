#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --job-name=1a_bone010_1_nodes_2_tasks_32_threads_mpi
#SBATCH --distribution=block:block
#SBATCH --exclusive
#SBATCH --time=0-0:10:00
#SBATCH --output=/home/krisor99/SpMV-Comm-Strats/P2/results/multi/defq/test/%x-%j-stdout.txt
#SBATCH --error=/home/krisor99/SpMV-Comm-Strats/P2/results/multi/defq/test/%x-%j-stderr.txt

module load openmpi-4.1.6
module load cmake-3.22.3
export LC_ALL=C
srun --verbose numactl -C0-63 /home/krisor99/SpMV-Comm-Strats/P2/build/Debug/1a /global/D1/projects/mtx/datasets/suitesparse/Oberwolfach/bone010/bone010.mtx
