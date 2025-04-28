import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict, DefaultDict, List
from matplotlib.ticker import LogLocator as LL
from matplotlib.patches import Patch
from sys import argv
import numpy as np
import re
import collections


def parse_file_name(file_name: str):
    tokens = file_name.split("/")[-1].split("_")
    comm, name, nodes, tasks, threads, mpi = 0, "", 0, 0, 0, 0
    if "mpi" in file_name:
        mpi = 1

    if "1a" in file_name:
        comm = 0
    elif "1b" in file_name:
        comm = 1
    elif "1c" in file_name:
        comm = 2
    elif "1d" in file_name:
        comm = 3

    for i, token in enumerate(tokens):
        if token == "nodes":
            name = "_".join(tokens[: i - 1])
            nodes = int(tokens[i - 1])
        elif token == "tasks":
            tasks = int(tokens[i - 1])
        elif token == "threads":
            threads = int(tokens[i - 1])
    return name, comm, nodes, tasks, threads, mpi


def parse_file_contents(file_name: str):
    ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg = 0, 0, 0, 0, 0, 0, 0
    with open(file_name, "r") as f:
        for line in f:
            tokens = line.split()
            if "Total time" in line:
                ttot = float(tokens[3][:-1])
            elif "Communication time" in line:
                tcomm = float(tokens[3][:-1])
            elif "Copmutation time" in line:
                tcomp = float(tokens[3][:-1])
            elif "GFLOPS" in line:
                gflops = float(tokens[2])
            elif "Comm min" in line:
                comm_min = float(tokens[3])
            elif "Comm max" in line:
                comm_max = float(tokens[3])
            elif "Comm avg" in line:
                comm_avg = float(tokens[3])
        return ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg


def parse_file(file_name):
    name, comm, nodes, tasks, threads, mpi = parse_file_name(file_name)
    ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg = parse_file_contents(file_name)
    return (
        name,
        comm,
        nodes,
        tasks,
        threads,
        mpi,
        ttot,
        tcomm,
        tcomp,
        gflops,
        comm_min,
        comm_max,
        comm_avg,
    )


def plot_gflops(directory_path):
    results = []

    # Parse all files
    for file in Path(directory_path).iterdir():
        if file.is_file() and "stdout" in file.name:
            data = parse_file(str(file))
            results.append(data)

    # Organize results
    # Key = (matrix_name, mpi_flag)
    matrix_mpi_to_results = collections.defaultdict(list)

    for (
        name,
        comm,
        nodes,
        tasks,
        threads,
        mpi,
        ttot,
        tcomm,
        tcomp,
        gflops,
        comm_min,
        comm_max,
        comm_avg,
    ) in results:
        x_value = (nodes, tasks, threads)
        matrix_mpi_to_results[(name, mpi)].append((comm, x_value, gflops))

    labels = {0: "1a", 1: "1b", 2: "1c", 3: "1d"}
    markers = ["o", "s", "D", "^"]

    # Now: one plot per (matrix, mpi_flag)
    for (matrix_name, mpi_flag), entries in matrix_mpi_to_results.items():
        # Organize entries by strategy
        strat_to_data = collections.defaultdict(list)
        for comm, x_value, gflops in entries:
            strat_to_data[comm].append((x_value, gflops))

        # Sort each strategy's data
        for comm in strat_to_data:
            strat_to_data[comm].sort()

        # Plot
        plt.figure(figsize=(10, 6))

        for comm, marker in zip(sorted(strat_to_data.keys()), markers):
            x_labels = [
                f"{nodes}n-{tasks}t-{threads}th"
                for (nodes, tasks, threads), _ in strat_to_data[comm]
            ]
            gflops = [g for _, g in strat_to_data[comm]]

            # Use scatter plot
            plt.scatter(
                x_labels, gflops, label=f"Strategy {labels.get(comm, comm)}", marker=marker, s=70
            )

        plt.xlabel("Configuration (nodes-tasks-threads)")
        plt.ylabel("GFLOPS")
        plt.title(f"Matrix: {matrix_name} (MPI={mpi_flag})")
        plt.xticks(rotation=45, ha="right")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


def main():
    base_path = Path(
        "/Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P2/results/single/milanq"
    )

    results = []

    # for file in base_path.iterdir():
    #     if file.is_file() and file.suffix == ".txt" and "stderr" not in file.name:
    #         parsed_entry = parse_file(str(file))
    #         results.append(parsed_entry)

    plot_gflops(base_path)


if __name__ == "__main__":
    main()
