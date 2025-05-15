#!/bin/bash
#SBATCH --partition=rome16q
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --job-name=1a_bone010_6_nodes_1_tasks_16_threads
#SBATCH --distribution=block:block
#SBATCH --exclusive
#SBATCH --time=0-0:10:00
#SBATCH --output=/home/krisor99/SpMV-Comm-Strats/P2/results/multi/rome16q/%x-%j-stdout.txt
#SBATCH --error=/home/krisor99/SpMV-Comm-Strats/P2/results/multi/rome16q/%x-%j-stderr.txt

module load openmpi-4.1.6
module load cmake-3.22.3
export LC_ALL=C
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_btl_tcp_if_exclude=lo,eno1,eno2,docker0,docker_gwbridge
export UCX_TLS=rc,ud,self
export UCX_NET_DEVICES=mlx5_1:1
export OMP_NUM_THREADS=16

srun --verbose numactl -C0-15 /home/krisor99/SpMV-Comm-Strats/P2/build/rome16q/1a /global/D1/projects/mtx/datasets/suitesparse/Oberwolfach/bone010/bone010.mtx
