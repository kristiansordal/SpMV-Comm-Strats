import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict, DefaultDict, List
from matplotlib.ticker import LogLocator as LL
from matplotlib.patches import Patch
from sys import argv
import numpy as np
import re


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


def plot_gflops_by_name_and_mpi(parsed_entries: List[tuple]):
    from matplotlib.ticker import LogLocator as LL

    from collections import defaultdict
    import matplotlib.pyplot as plt

    # Group entries by 'name'
    grouped: DefaultDict[str, List[tuple]] = defaultdict(list)
    for entry in parsed_entries:
        name = entry[0]
        grouped[name].append(entry)

    # Define colors for different communication strategies
    color_map = {
        0: "blue",
        1: "green",
        2: "orange",
        3: "red",
    }

    for name, entries in grouped.items():
        fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        fig.suptitle(name)

        for mpi_value, ax in zip([0, 1], axs):
            ax.set_title(f"MPI = {mpi_value}")
            ax.set_xlabel("Nodes")
            ax.set_ylabel("GFLOPS")
            ax.set_xscale("log", base=2)
            ax.xaxis.set_major_locator(LL(base=2, numticks=10))

            for comm in range(4):
                filtered = [e for e in entries if e[1] == comm and e[5] == mpi_value]
                if filtered:
                    nodes = [e[2] for e in filtered]
                    gflops = [e[9] for e in filtered]
                    ax.plot(nodes, gflops, marker="o", label=f"Comm {comm}", color=color_map[comm])

            ax.legend()

        plt.tight_layout()
        plt.show()
        # plt.subplots_adjust(top=0.88)
        # plt.savefig(f"{name}_gflops_plot.png")
        plt.close()


def main():
    base_path = argv[1]
    partition = argv[2]


if __name__ == "__main__":
    main()
