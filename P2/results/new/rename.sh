for file in /Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P2/results/new/1d/defq/*; do
  mv "$file" "/Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P2/results/multi/defq/1d_$(basename "$file")"
done
