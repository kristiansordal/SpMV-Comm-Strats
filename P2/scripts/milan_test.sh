#!/bin/bash

# GROUP_FILE="/global/D1/projects/mtx/matrixlists/"
GROUP_FILE="output.txt"
# MTX_PATH="/global/D1/projects/mtx/datasets/suitesparse/"

wait_for_jobs() {
    echo "Waiting for all Slurm jobs to complete..."
    while true; do
        job_count=$(squeue -u krisor99 | wc -l)
        # If there are more than 1 line (header + jobs), jobs are still running
        if [[ $job_count -le 1 ]]; then
            break
        fi
        sleep 5
    done
}

if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
    while IFS= read -r line; do
        sh launch.sh "milanq" "$line" 1 1 128
        sh launch.sh "milanq" "$line" 2 1 128
        sh launch.sh "milanq" "$line" 3 1 128 
        sh launch.sh "milanq" "$line" 4 1 128 

        # Wait for all jobs to complete before moving to the next set
        wait_for_jobs

        # Additional configurations for 24 threads
        #sh launch.sh "milanq" "$line" 1 2 64
        #sh launch.sh "milanq" "$line" 2 2 64
        #sh launch.sh "milanq" "$line" 3 2 64
        #sh launch.sh "milanq" "$line" 4 2 64

        # Wait for all jobs to complete before processing the next matrix
        #wait_for_jobs
    done < "$GROUP_FILE"
else
    echo "Error: File '$GROUP_FILE' does not exist or is not readable."
    exit 1
fi
