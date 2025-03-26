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


# srun --verbose numactl -C0-31 /home/krisor99/aCG/bin/acg-baseline /global/D1/projects/mtx/datasets/suitesparse/$matrix
# Define the sbatch job script content
sbatch_script=$(cat <<EOF
#!/bin/bash
module load openmpi-4.1.6
module load cmake-3.22.3
export LC_ALL=C
srun --verbose numactl -C0-48 /home/krisor99/aCG/build/Debug/1a $matrix
EOF
)

# srun --verbose numactl -C0-31 /home/krisor99/aCG/build/Debug/1a /global/D1/projects/mtx/datasets/suitesparse/$matrix

# Submit the job
job_id=$(echo "$sbatch_script" | sbatch \
    --partition="${partition}" \
    --nodes="${nodes}" \
    --ntasks-per-node="${tasks_per_node}" \
    --cpus-per-task="${omp_num_threads}" \
    --job-name="${job_name}" \
    --distribution=block:block \
    --exclusive \
    --time=0-0:10:00 \
    --output=/home/krisor99/aCG/results/output/%x-%j-stdout.txt \
    --error=/home/krisor99/aCG/results/test_rome/%x-%j-stderr.txt \
    | awk '{print $4}')

# Print job start message
if [ -e "/global/D1/projects/mtx/datasets/suitesparse/$matrix" ]; then
    echo "Started job '${job_name}' with ID ${job_id}."
else
    echo "File doesn't exist."
fi
