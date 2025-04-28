import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict, DefaultDict, List
from matplotlib.ticker import LogLocator as LL
from matplotlib.patches import Patch
from sys import argv
import numpy as np
import re


# the result of running SPMV on a single comm strat with a given configuration
class SingleResult:
    def __init__(
        self, nodes, tasks, threads, t, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg
    ):
        self.nodes = nodes
        self.tasks = tasks
        self.threads = threads
        self.t: float = t
        self.tcomm: float = tcomm
        self.tcomp: float = tcomp
        self.gflops: float = gflops
        self.comm_min: float = comm_min
        self.comm_max: float = comm_max
        self.comm_avg: float = comm_avg

    def __str__(self):
        return (
            f"nodes: {self.nodes}, tasks: {self.tasks}, threads: {self.threads}, "
            f"t: {self.t}, tcomm: {self.tcomm}, tcomp: {self.tcomp}, "
            f"gflops: {self.gflops}, comm_min: {self.comm_min}, comm_max: {self.comm_max}, comm_avg: {self.comm_avg}"
        )


def parse_file_name(file_name: str):
    tokens = file_name.split("/")[-1].split("_")
    name, nodes, tasks, threads, mpi = "", 0, 0, 0, 0
    if "mpi" in file_name:
        mpi = 1
    for i, token in enumerate(tokens):
        if token == "nodes":
            name = "_".join(tokens[: i - 1])
            nodes = int(tokens[i - 1])
        elif token == "tasks":
            tasks = int(tokens[i - 1])
        elif "threads" in token:
            threads = int(tokens[i - 1])
    return name, nodes, tasks, threads, mpi


def parse_file_name_single(file_name: str):
    tokens = file_name.split("/")[-1].split("_")
    comm_strat = tokens[0]
    name, nodes, tasks, threads, mpi = "", 0, 0, 0, 0
    # if "mpi" in file_name:
    #     mpi = 1
    for i, token in enumerate(tokens):
        if token == "nodes":
            name = "_".join(tokens[: i - 1])
            nodes = int(tokens[i - 1])
        elif token == "tasks":
            tasks = int(tokens[i - 1])
        elif "threads" in token:
            threads = int(tokens[i - 1])

    if tasks > 1:
        mpi = 1
    return comm_strat, name, nodes, tasks, threads, mpi


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


# comm_strat -> matrix -> result
def gather_results_from_directory(results, file_path: Path, pure_mpi):
    # Map from config string to (job_id, file path)

    config_to_best_file = {}

    # Adjusted regex to match your file format
    pattern = re.compile(r"^([a-zA-Z0-9]+)_([a-zA-Z0-9_]+)-(\d+)-stdout\.txt$")

    for file in file_path.iterdir():
        if file.is_file() and "stdout" in file.name:

            match = pattern.match(file.name)
            if not match:
                continue

            comm_strat, config_str, job_id_str = match.groups()
            print(comm_strat, config_str, job_id_str)
            job_id = int(job_id_str)

            valid_result = False
            for line in file.open():
                if "GFLOPS" in line:
                    valid_result = True

            if valid_result:
                if config_str not in config_to_best_file or (
                    job_id > config_to_best_file[config_str][0]
                    and comm_strat == config_to_best_file[config_str][2]
                ):
                    config_to_best_file[config_str] = (job_id, file, comm_strat)

    for _, file, comm_strat in config_to_best_file.values():
        # Extract the communication strategy and config from the filename
        # comm_strat = file.name.split("_")[0]  # This is now part of the file name
        # Parsing the rest of the information (name, nodes, tasks, threads, mpi)
        name, nodes, tasks, threads, mpi = parse_file_name(str(file))
        name = "_".join(name.split("_")[1:])  # Remove the communication strategy from name

        # Parse file contents (ttot, tcomm, tcomp, etc.)
        ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg = parse_file_contents(str(file))

        # Handle case when nodes == 1 and tasks == 1
        if nodes == 1 and tasks == 1:
            comm_min = 0
            comm_max = 0
            comm_avg = 0
            tcomm = 0

        # Add the result to the dictionary
        results[comm_strat][name][mpi].append(
            SingleResult(
                nodes, tasks, threads, ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg
            )
        )


def gather_results_single(directory, results):
    latest_files = defaultdict(lambda: ("", "", 0))

    for file in directory.iterdir():
        if not file.is_file() or "stderr" in file.name:
            continue

        config, job_id, _ = file.name.split("-")
        job_id = int(job_id)

        # print(config)
        print(job_id, latest_files[config])

        if job_id > latest_files[config][2]:
            latest_files[config] = (str(file), config, job_id)  # type: ignore

    for file, config, job_id in latest_files.values():
        print(config, job_id)
        comm_strat, name, nodes, tasks, threads, mpi = parse_file_name_single(str(file))
        ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg = parse_file_contents(str(file))
        results[comm_strat][name][mpi].append(
            SingleResult(
                nodes, tasks, threads, ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg
            )
        )


def plot_gflops_single(results):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    titles = ["Shared Memory", "Distributed Memory"]

    for idx, use_mpi in enumerate([False, True]):
        ax = axes[idx]
        for comm_strat in ["1a", "1b", "1c", "1d"]:
            all_threads = []
            all_gflops = []

            for matrix in results[comm_strat]:
                for mpi_flag in results[comm_strat][matrix]:
                    if (mpi_flag > 0) == use_mpi:
                        for res in results[comm_strat][matrix][mpi_flag]:
                            all_threads.append(res.threads if not mpi_flag else res.tasks)
                            all_gflops.append(res.gflops)

            if all_threads:
                # Sort by number of threads
                sorted_pairs = sorted(zip(all_threads, all_gflops))
                threads_sorted, gflops_sorted = zip(*sorted_pairs)
                ax.plot(threads_sorted, gflops_sorted, label=f"Comm {comm_strat}", marker="o")

        # Set x-axis to log base 2
        ax.set_xscale("log", base=2)

        # Set nice xticks (1, 2, 4, 8, 16, 32, 64)
        ticks = [1, 2, 4, 8, 16, 32, 64]
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])

        # Set titles and labels
        ax.set_title(titles[idx])
        ax.set_ylabel("GFLOPS")
        ax.set_xlabel("Threads" if not use_mpi else "Processes")
        ax.legend()
        ax.grid(True, which="both", ls="--")

    plt.tight_layout()
    plt.show()


def plot_compare_comm_strat(results):
    for matrix in results[next(iter(results))]:  # Iterate over matrix names
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        fig.suptitle(f"GFLOPS Comparison for {matrix}")

        mpi_labels = ["Single Process (mpi=0)", "Multi Process (mpi=1)"]
        colors = ["b", "g", "r", "c"]  # Colors for different comm_strats

        for mpi, ax in enumerate(axes):
            ax.set_title(mpi_labels[mpi])
            ax.set_xlabel("Number of Nodes")
            ax.set_ylabel("GFLOPS")

            for i, (comm_strat, matrices) in enumerate(results.items()):
                if matrix in matrices and mpi in matrices[matrix]:
                    # Sort the results by number of nodes in ascending order
                    sorted_results = sorted(matrices[matrix][mpi], key=lambda res: res.nodes)
                    nodes = [res.nodes for res in sorted_results]
                    gflops = [res.gflops for res in sorted_results]

                    ax.plot(
                        nodes,
                        gflops,
                        marker="o",
                        linestyle="-",
                        color=colors[i % len(colors)],
                        label=comm_strat,
                    )

            ax.legend()
            ax.grid(True, linestyle="--", alpha=0.6)

        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def plot_comm_load(results):
    # Get all communication strategies
    comm_strats = list(results.keys())

    # Get all unique node counts
    node_counts = set()
    for comm_strat in results.values():
        for matrix in comm_strat.values():
            for mpi in matrix.values():
                for res in mpi:
                    node_counts.add(res.nodes)
    node_counts = sorted(node_counts)

    # Prepare data structure: comm_strat -> mpi -> nodes -> [comm_avg values]
    data = {strat: {0: defaultdict(list), 1: defaultdict(list)} for strat in comm_strats}

    # Collect data
    for comm_strat, matrices in results.items():
        for matrix in matrices.values():
            for mpi, results_list in matrix.items():
                for res in results_list:
                    data[comm_strat][mpi][res.nodes].append(res.comm_avg)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot configuration
    bar_width = 0.15
    opacity = 0.8
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    mpi_labels = ["Single-socket", "Dual-socket"]

    # X-axis positions for groups
    index = np.arange(len(node_counts))

    # Plot bars for each communication strategy and MPI configuration
    for i, comm_strat in enumerate(comm_strats):
        for mpi in [0, 1]:
            # Calculate average communication for each node count
            avg_comms = [
                np.mean(data[comm_strat][mpi][nodes]) if data[comm_strat][mpi][nodes] else 0
                for nodes in node_counts
            ]

            # Position adjustment based on MPI and strategy
            pos = index + (i * 2 + mpi) * bar_width

            ax.bar(
                pos,
                avg_comms,
                bar_width,
                alpha=opacity,
                color=colors[i],
                label=f"{comm_strat} {mpi_labels[mpi]}",
            )

    ax.set_yscale("log")  # Set y-axis to logarithmic scale
    ax.yaxis.set_major_locator(LL(base=10, numticks=15))
    ax.yaxis.set_minor_locator(
        LL(base=10, subs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], numticks=12)
    )
    # Formatting
    ax.set_xlabel("Number of Nodes", fontsize=12)
    ax.set_ylabel("Communication Load (GB)", fontsize=12)
    ax.set_title("Communication Load Comparison Across Strategies", fontsize=14)
    ax.set_xticks(index + (len(comm_strats) * bar_width))
    ax.set_xticklabels([f"{n} nodes" for n in node_counts])
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_comm_and_comp_time(results):
    comm_strats = list(results.keys())

    for matrix in results[next(iter(results))]:  # Iterate over matrix names
        fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
        fig.suptitle(f"Communication and Computation Time for {matrix}", fontsize=16)

        node_counts = sorted(
            {
                res.nodes
                for comm_strat in results.values()
                for mat_name, mpi_data in comm_strat.items()
                if mat_name == matrix
                for res_list in mpi_data.values()
                for res in res_list
            }
        )

        bar_width = 0.15
        index = np.arange(len(node_counts))
        colors = {"comm": "skyblue", "comp": "sandybrown"}
        mpi_labels = ["Single MPI Process", "Dual MPI Processes"]

        for mpi in [0, 1]:
            ax = axes[mpi]
            ax.set_title(mpi_labels[mpi])
            ax.set_xlabel("Number of Nodes")
            if mpi == 0:
                ax.set_ylabel("Time (s)")
            ax.set_xticks(index + ((len(comm_strats) - 1) / 2) * bar_width)
            ax.set_xticklabels([f"{n}" for n in node_counts])
            ax.grid(True, linestyle="--", alpha=0.6)

            for i, comm_strat in enumerate(comm_strats):
                if matrix not in results[comm_strat] or mpi not in results[comm_strat][matrix]:
                    continue
                strat_results = results[comm_strat][matrix][mpi]

                # Map node count -> list of times
                time_by_node = defaultdict(list)
                for res in strat_results:
                    time_by_node[res.nodes].append(res)

                comm_times = []
                comp_times = []
                for node in node_counts:
                    entries = time_by_node[node]
                    if entries:
                        avg_comm = np.mean([r.tcomm for r in entries])
                        avg_comp = np.mean([r.tcomp for r in entries])
                    else:
                        avg_comm = 0
                        avg_comp = 0
                    comm_times.append(avg_comm)
                    comp_times.append(avg_comp)

                pos = index + i * bar_width
                ax.bar(pos, comm_times, bar_width, label=f"{comm_strat} comm", color=colors["comm"])
                ax.bar(
                    pos,
                    comp_times,
                    bar_width,
                    bottom=comm_times,
                    label=f"{comm_strat} comp",
                    color=colors["comp"],
                )

        handles = [
            Patch(facecolor=colors["comm"], label="Communication"),
            Patch(facecolor=colors["comp"], label="Computation"),
        ]
        labels = ["Communication", "Computation"]
        fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94))
        plt.tight_layout()
        plt.show()


def plot_comm_min_avg_max(results):
    comm_strats = list(results.keys())
    comm_strats = [c for c in comm_strats if c != "1a"]
    colors = {"min": "blue", "avg": "green", "max": "red"}
    markers = {"min": "o", "avg": "s", "max": "^"}
    labels = ["min", "avg", "max"]

    for matrix in results[next(iter(results))]:  # Loop over matrix names
        fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
        fig.suptitle(f"Communication Load for {matrix}", fontsize=16)

        for ax_index, mpi in enumerate([0, 1]):
            ax = axs[ax_index]
            mode = "MPI" if mpi == 1 else "Non-MPI"
            ax.set_title(mode)

            node_counts = sorted(
                {
                    res.nodes
                    for comm_strat in results.values()
                    for mat_name, mpi_data in comm_strat.items()
                    if mat_name == matrix
                    for res in mpi_data.get(mpi, [])
                }
            )

            xticks = []
            xticklabels = []

            for node_index, n in enumerate(node_counts):
                for entry_index, comm_strat in enumerate(comm_strats):
                    x_base = node_index * 4 + entry_index
                    matching_results = [
                        res for res in results[comm_strat][matrix].get(mpi, []) if res.nodes == n
                    ]
                    if matching_results:
                        comm_mins = [r.comm_min for r in matching_results]
                        comm_avgs = [r.comm_avg for r in matching_results]
                        comm_maxs = [r.comm_max for r in matching_results]

                        for val, label in zip([comm_mins, comm_avgs, comm_maxs], labels):
                            ax.plot(
                                [x_base] * len(val),
                                val,
                                marker=markers[label],
                                linestyle="None",
                                color=colors[label],
                                alpha=0.7,
                                label=label if x_base == 0 and ax_index == 0 else None,
                            )

                    xticks.append(x_base)
                    xticklabels.append(f"{n}\n{comm_strat}")

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=9)
            ax.set_xlabel("Node Configurations")
            ax.set_yscale("log")
            ax.grid(True, linestyle="--", alpha=0.5)
            if ax_index == 0:
                ax.set_ylabel("Communication Time (s)")
            ax.legend()

        plt.tight_layout()
        plt.show()


def main():
    base_path = argv[1]
    partition = argv[2]
    comm_strats = [argv[3]] if argv[3] != "all" else ["1a", "1b", "1c", "1d"]
    # Define nested defaultdicts
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # for comm_strat in comm_strats:
    path = Path(base_path + partition)
    gather_results_single(path, results)
    plot_gflops_single(results)
    # gather_results_from_directory(results, path, 0)
    # plot_comm_min_avg_max(results)
    # plot_comm_and_comp_time(results)
    # plot_compare_comm_strat(results)
    # plot_comm_load(results)
    # plot_comm_load2(results)


if __name__ == "__main__":
    main()
