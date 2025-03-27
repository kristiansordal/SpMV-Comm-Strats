#!/bin/bash

# Input arguments
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

ulimit -s 10240  # Increase stack size limit

# Load necessary modules
module purge
module load slurm/21.08.8
module load numactl/gcc/2.0.18
module load hwloc/gcc/2.10.0
module load openmpi-4.1.6
module load ucx  # Ensure UCX is loaded
module load metis

# Force OpenMPI to use UCX
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^vader,tcp,openib  # Disable other transports
export OMPI_MCA_osc=ucx
export OMPI_MCA_coll=ucx
export OMPI_MCA_opal_common_ucx_opal_mem_hooks=1
export OMPI_MCA_pml_ucx_verbose=100   # Enable debugging output

# UCX tuning for large messages
export UCX_RNDV_THRESH=8192
export UCX_MAX_RNDV_RAILS=2
export UCX_IB_RX_QUEUE_LEN=1024
export UCX_IB_TX_QUEUE_LEN=1024

# Detect UCX network device and use the correct one
UCX_DEVICE=$(ucx_info -d | grep mlx | awk '{print $1}' | head -n 1)
if [[ -n "$UCX_DEVICE" ]]; then
    export UCX_NET_DEVICES=${UCX_DEVICE}:1
else
    echo "Warning: No UCX network device detected. Using default settings."
fi

export OMP_NUM_THREADS=$omp_num_threads
export OMPI_MCA_opal_cuda_support=0  # Disable CUDA support if not needed

# Submit job with Slurm
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
module load ucx
module load cmake-3.22.3
export LC_ALL=C

# Run with srun and UCX
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
