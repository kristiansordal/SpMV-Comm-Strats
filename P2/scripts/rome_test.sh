#!/bin/bash

GROUP_FILE="/home/krisor99/SpMV-Comm-Strats/P2/scripts/DIMACS.txt"

if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
    while IFS= read -r line; do
        sh launch$1rome.sh "rome16q" "$line" 1 1 16
        sh launch$1rome.sh "rome16q" "$line" 2 1 16
        sh launch$1rome.sh "rome16q" "$line" 3 1 16
        sh launch$1rome.sh "rome16q" "$line" 4 1 16
        sh launch$1rome.sh "rome16q" "$line" 5 1 16
        sh launch$1rome.sh "rome16q" "$line" 6 1 16
        sh launch$1rome.sh "rome16q" "$line" 7 1 16 
        sh launch$1rome.sh "rome16q" "$line" 8 1 16 
    done < "$GROUP_FILE"
else
    echo "Error: File '$GROUP_FILE' does not exist or is not readable."
    exit 1
fi
