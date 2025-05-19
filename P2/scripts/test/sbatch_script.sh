#!/bin/bash
#SBATCH --partition=fpgaq
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=24
#SBATCH --job-name=1a_bone010_4_nodes_2_tasks_24_threads_mpi
#SBATCH --distribution=block:block
#SBATCH --exclusive
#SBATCH --time=0-0:10:00
#SBATCH --output=/home/krisor99/SpMV-Comm-Strats/P2/results/multi/fpgaq/%x-%j-stdout.txt
#SBATCH --error=/home/krisor99/SpMV-Comm-Strats/P2/results/multi/fpgaq/%x-%j-stderr.txt

module load openmpi-4.1.6
module load cmake-3.22.3
export LC_ALL=C
export OMPI_MCA_coll_tuned_allgatherv_algorithm=2
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_btl_tcp_if_exclude=lo,eno1,eno2,docker0,docker_gwbridge
export UCX_TLS=rc,ud,self
export UCX_NET_DEVICES=mlx5_4:1
export OMP_NUM_THREADS=24

srun --verbose numactl -C0-47 /home/krisor99/SpMV-Comm-Strats/P2/build/fpgaq/1a /global/D1/projects/mtx/datasets/suitesparse/Oberwolfach/bone010/bone010.mtx
