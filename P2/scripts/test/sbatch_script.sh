#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=32
#SBATCH --job-name=1a_Lynx144_4_nodes_2_tasks_32_threads_mpi
#SBATCH --distribution=block:block
#SBATCH --exclusive
#SBATCH --time=0-0:10:00
#SBATCH --output=/home/krisor99/SpMV-Comm-Strats/P2/results/multi/defq/%x-%j-stdout.txt
#SBATCH --error=/home/krisor99/SpMV-Comm-Strats/P2/results/multi/defq/%x-%j-stderr.txt

module load openmpi-4.1.6
module load cmake-3.22.3
export LC_ALL=C
srun --verbose numactl -C0-63 /home/krisor99/SpMV-Comm-Strats/P2/build/Debug/1a /global/D1/projects/HPC-data/Simula_collection/Lynx_traditional/Lynx144.mtx
