#!/usr/bin/env bash
# delete_if_starts_with_bracket.sh
# Deletes any regular file in the cwd whose first character of the first line is '['

shopt -s nullglob

for file in *; do
  # skip non-regular files
  [[ -f "$file" ]] || continue

  # read only the first line into 'line' (doesn't consume rest of file)
  IFS= read -r line < "$file"
  first_char="${line:0:1}"

  if [[ "$first_char" == "[" ]]; then
    echo "Deleting: $file"
    rm -- "$file"
  fi
done
