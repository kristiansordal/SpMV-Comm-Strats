#!/usr/bin/env bash

# Usage: ./rename_strip_mpi.sh [directory]
# Defaults to current directory if none provided.

dir="${1:-.}"

# Loop over all files in the directory whose names contain "_mpi"
shopt -s nullglob
for filepath in "$dir"/*_mpi*; do
    filename="$(basename -- "$filepath")"
    dirname="$(dirname -- "$filepath")"

    # Strip out all occurrences of "_mpi"
    newname="${filename//_mpi/}"
    newpath="$dirname/$newname"

    if [[ -e "$newpath" ]]; then
        echo "Skipping '$filename': target '$newname' already exists."
    else
        echo "Renaming '$filename' → '$newname'"
        mv -- "$filepath" "$newpath"
    fi
done
shopt -u nullglob
