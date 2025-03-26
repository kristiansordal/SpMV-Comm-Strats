#!/bin/bash

# GROUP_FILE="/global/D1/projects/mtx/matrixlists/"
GROUP_FILE="one_mat.txt"
# MTX_PATH="/global/D1/projects/mtx/datasets/suitesparse/"

if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
    while IFS= read -r line; do
        # Additional configurations for 48 threads
        sh launch.sh "fpgaq" "$line" 1 1 48
        sh launch.sh "fpgaq" "$line" 2 1 48
        sh launch.sh "fpgaq" "$line" 3 1 48 


    done < "$GROUP_FILE"
else
    echo "Error: File '$GROUP_FILE' does not exist or is not readable."
    exit 1
fi
