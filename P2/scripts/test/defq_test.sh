#!/bin/bash
GROUP_FILE="../output.txt"

#if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
#    while IFS= read -r line; do
       # sh launch$1.sh $2 "$line" 1 1 64
        #sh launch$1.sh $2 "$line" 2 1 64 
        #sh launch$1.sh $2 "$line" 3 1 64
        #sh launch$1.sh $2 "$line" 4 1 64 
#j    done < "$GROUP_FILE"
#else
    #echo "Error: File '$GROUP_FILE' does not exist or is not readable."
    #exit 1
#fi

if [[ -f "$GROUP_FILE" && -r "$GROUP_FILE" ]]; then
    while IFS= read -r line; do
        sh launch$1.sh $2 "$line" 1 2 32
        #jsh launch$1.sh $2 "$line" 2 2 32
        #sh launch$1.sh $2 "$line" 3 2 32
        #sh launch$1.sh $2 "$line" 4 2 32 
    done < "$GROUP_FILE"
else
    echo "Error: File '$GROUP_FILE' does not exist or is not readable."
    exit 1
fi
