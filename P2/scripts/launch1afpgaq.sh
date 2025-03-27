partition=$1
matrix=$2
last_entry="${matrix##*/}"
matrix_name=${last_entry%.mtx}
nodes=$3
tasks_per_node=$4
omp_num_threads=$5

job_name="${matrix_name}_${nodes}_nodes_${tasks_per_node}_tasks_${omp_num_threads}_threads"
if [ "$tasks_per_node" == "2" ]; then
    job_name="${job_name}_mpi"
fi

ulimit -s 10240

# module purge
# Modules on all architectures
module load slurm/21.08.8
module load numactl/gcc/2.0.18
module load hwloc/gcc/2.10.0
module load metis
module load ucx

export OMPI_MCA_opal_common_ucx_opal_mem_hooks=1
export OMPI_MCA_pml_ucx_verbose=100
export OMPI_MCA_btl_openib_allow_ib=1
export OMPI_MCA_btl_openib_warn_no_device_params_found=1
export OMPI_MCA_btl_openib_if_include="mlx5_1:1"          # Use 'ibstat' and look for active HCA(s) as there are 2 IB topologies in partition 'fpgaq16q'
# export OMPI_MCA_pml="^ucx"
# export OMPI_MCA_btl=self,vader
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=self,vader,tcp
export OMPI_MCA_btl_tcp_if_exclude=lo,dis0,eno1,eno2,enp113s0f0,ib0,ib1,enp33s0f0,enp33s0f1,docker0,docker_gwbridge

export OMP_NUM_THREADS=$omp_num_threads
export OMPI_MCA_opal_cuda_support=0                     # new option for above


sbatch_script=$(cat <<EOF
#!/bin/bash
#SBATCH --partition=${partition}
#SBATCH --nodes=${nodes}
#SBATCH --ntasks-per-node=${tasks_per_node}
#SBATCH --cpus-per-task=${omp_num_threads}
#SBATCH --job-name=${job_name}
#SBATCH --distribution=block:block
#SBATCH --exclusive
#SBATCH --time=0-0:10:00
#SBATCH --output=/home/krisor99/SpMV-Comm-Strats/P2/results/new/1a/fpgaq/%x-%j-stdout.txt
#SBATCH --error=/home/krisor99/SpMV-Comm-Strats/P2/results/new/1a/fpgaq/%x-%j-stderr.txt

module load openmpi-4.1.6
module load cmake-3.22.3
export LC_ALL=C
srun --verbose numactl -C0-47 /home/krisor99/SpMV-Comm-Strats/P2/build/Debug/1a /global/D1/projects/mtx/datasets/suitesparse/$matrix
EOF
)

# Submit the job
job_id=$(echo "$sbatch_script" | sbatch | awk '{print $4}')

# Print job start message
if [ -e "/global/D1/projects/mtx/datasets/suitesparse/$matrix" ]; then
    echo "Started job '${job_name}' with ID ${job_id}."
else
    echo "File doesn't exist."
fi
