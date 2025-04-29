#!/bin/bash
GROUP_FILE="../Lynx.txt"

if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
    while IFS= read -r line; do
        sh launch$1.sh $2 "$line" 1 1 128 
        sh launch$1.sh $2 "$line" 2 1 128 
        sh launch$1.sh $2 "$line" 3 1 128
        sh launch$1.sh $2 "$line" 4 1 128 
    done < "$GROUP_FILE"
else
    echo "Error: File '$GROUP_FILE' does not exist or is not readable."
    exit 1
fi

if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
    while IFS= read -r line; do
        sh launch$1.sh $2 "$line" 1 2 64
        sh launch$1.sh $2 "$line" 2 2 64
        sh launch$1.sh $2 "$line" 3 2 64
        sh launch$1.sh $2 "$line" 4 2 64
    done < "$GROUP_FILE"
else
    echo "Error: File '$GROUP_FILE' does not exist or is not readable."
    exit 1
fi
