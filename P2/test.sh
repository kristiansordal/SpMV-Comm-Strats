#!/bin/bash

PROGRAM="build/Debug/1a nlpkkt160.mtx"
NP=4

echo "Launching $NP MPI processes for: $PROGRAM"
mpirun -n $NP $PROGRAM &

MPI_PID=$!
sleep 2  # Give processes a moment to start

echo "Getting PIDs of MPI processes..."
PIDS=$(ps aux | grep "$PROGRAM" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
  echo "No processes found. Make sure your program is running."
  exit 1
fi

echo "Found PIDs:"
echo "$PIDS"

echo ""
echo "Running 'leaks' on each PID..."
for pid in $PIDS; do
  echo "🔍 Checking PID $pid"
  leaks $pid
  echo "----------------------------"
done

wait $MPI_PID
