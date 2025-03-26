#!/bin/bash

# Check if a file is provided as an argument

input_file="$1"
base_path="/global/D1/projects/mtx/datasets/suitesparse"
output_file="output.txt"

# Check if the input file exists and is readable
if [[ ! -f "$input_file" || ! -r "$input_file" ]]; then
    echo "Error: Cannot read file $input_file"
    exit 1
fi

# Clear the output file if it exists
> "$output_file"

# Loop through each line in the file
while IFS= read -r line; do
    # Concatenate the line with the base path
    full_path="$base_path/$line"

    # Check if the file exists and is readable
    if [[ -f "$full_path" && -r "$full_path" ]]; then
        # Read the first line of the file
        first_line=$(head -n 1 "$full_path")

        # Check if the first line contains the word "pattern"
        if [[ "$first_line" == *real* ]]; then
            echo "Match found in file: $full_path"
            # Append the entry to the output file
            echo "$line" >> "$output_file"
        else
            echo "No match in file: $full_path"
        fi
    else
        echo "Error: File does not exist or cannot be read: $full_path"
    fi
done < "$input_file"

echo "Filtered entries saved to $output_file"

