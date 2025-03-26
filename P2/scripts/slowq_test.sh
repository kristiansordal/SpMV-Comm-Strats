#!/bin/bash

GROUP_FILE="/home/krisor99/SpMV-Comm-Strats/P2/scripts/output.txt"

if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
    while IFS= read -r line; do
        sh launch$1slowq.sh "slowq" "$line" 1 1 4
        sh launch$1slowq.sh "slowq" "$line" 2 1 4
        sh launch$1slowq.sh "slowq" "$line" 3 1 4
        sh launch$1slowq.sh "slowq" "$line" 4 1 4
        sh launch$1slowq.sh "slowq" "$line" 5 1 4
        sh launch$1slowq.sh "slowq" "$line" 6 1 4
        sh launch$1slowq.sh "slowq" "$line" 7 1 4 
        sh launch$1slowq.sh "slowq" "$line" 8 1 4 
    done < "$GROUP_FILE"
else
    echo "Error: File '$GROUP_FILE' does not exist or is not readable."
    exit 1
fi
