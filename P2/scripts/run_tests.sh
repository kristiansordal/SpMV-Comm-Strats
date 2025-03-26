#!/bin/bash
# run_acg_baseline.sh
#
# This script runs the acg-baseline script with varying numbers of tasks,
# starting from 2 and increasing by 2. It adjusts the number of nodes based
# on the number of tasks: 1 node for tasks ≤ 24, 2 nodes for tasks ≤ 24, 
# 3 nodes for tasks ≤ 72, and 4 nodes for tasks > 72.

partition=$1 # Set the partition to use
matrix=$2 # Change this to the matrix you want to test
acg_use_openmp=$3

# check if we need to build the binary with support for openmp or not
if [ "$acg_use_openmp" == "1" ]; then
    cmake -S baseline -B bin -DACG_USE_OPENMP=1
else
    cmake -S baseline -B bin -DACG_USE_OPENMP=0
fi
cmake --build bin

if  [ "$partition" == "rome16q" ]; then
    if [ "$acg_use_openmp" == "1" ]; then
        # laucnhes shared memory jobs
        for threads in $(seq 1 1 16); do
            sh launch.sh "$partition" "$matrix" "1" "1" "1" "$threads" "1"
        done
        for nodes in $(seq 2 1 8); do
            sh launch.sh "$partition" "$matrix" "$nodes" "$nodes" "1" "16" "1"
        done
    else
        #launches distributed memory jobs
        for threads in $(seq 1 1 16); do
            sh launch.sh "$partition" "$matrix" "1" "$threads" "$threads" "1" "0"
        done
        for nodes in $(seq 2 1 8); do
            tasks=$((nodes*16))
            sh launch.sh "$partition" "$matrix" "$nodes" "$tasks" "16" "1" "0"
        done
    fi
elif [ "$partition" == "fpgaq" ]; then
    if [ "$acg_use_openmp" == "1" ]; then
        # laucnhes shared memory jobs
        for threads in $(seq 2 2 96); do
            sh launch.sh "$partition" "$matrix" "1" "1" "1" "$threads" "1"
        done
        for nodes in $(seq 2 1 4); do
            sh launch.sh "$partition" "$matrix" "$nodes" "$nodes" "1" "96" "1"
        done
    else
        #launches distributed memory jobs
        for threads in $(seq 2 2 24); do
            sh launch.sh "$partition" "$matrix" "1" "$threads" "$threads" "1" "0"
        done
        for nodes in $(seq 2 1 4); do
            tasks=$((nodes*24))
            sh launch.sh "$partition" "$matrix" "$nodes" "$tasks" "24" "1" "0"
        done
    fi
fi
