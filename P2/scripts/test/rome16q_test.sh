#!/bin/bash

GROUP_FILE="../output.txt"

if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
    while IFS= read -r line; do
        sh launch$1.sh $2 "$line" 1 1 16 
        sh launch$1.sh $2 "$line" 2 1 16
        sh launch$1.sh $2 "$line" 3 1 16
        sh launch$1.sh $2 "$line" 4 1 16
        sh launch$1.sh $2 "$line" 5 1 16
        sh launch$1.sh $2 "$line" 6 1 16
        sh launch$1.sh $2 "$line" 7 1 16
        sh launch$1.sh $2 "$line" 8 1 16
    done < "$GROUP_FILE"
else
    echo "Error: File '$GROUP_FILE' does not exist or is not readable."
    exit 1
fi

#if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
    #while IFS= read -r line; do
    #    sh launch$1.sh $2 "$line" 1 1 1
    #    sh launch$1.sh $2 "$line" 1 2 1
    #    sh launch$1.sh $2 "$line" 1 4 1
    #    sh launch$1.sh $2 "$line" 1 8 1
    #    sh launch$1.sh $2 "$line" 1 16 1
    #    sh launch$1.sh $2 "$line" 1 32 1
    #    sh launch$1.sh $2 "$line" 1 48 1
    #done < "$GROUP_FILE"
#else
#    echo "Error: File '$GROUP_FILE' does not exist or is not readable."
#    exit 1
#fi
