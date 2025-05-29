import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
from typing import Dict, DefaultDict, List
from matplotlib.ticker import LogLocator as LL
from matplotlib.ticker import LogLocator, LogFormatter
from matplotlib.patches import Patch
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from sys import argv
import numpy as np
import re
import matplotlib as mpl
from matplotlib import cm

mpl.rcParams.update(
    {
        "text.usetex": True,  # toggle this if you don’t have TeX installed
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Liberation Serif"],
        "font.size": 30,
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 28,
        "ytick.labelsize": 24,
        "legend.fontsize": 28,
    }
)

comm_strats_dict = {
    "1a": "Strategy A",
    "1b": "Strategy B",
    "1c": "Strategy C",
    "1d": "Strategy D",
    "2d": "Strategy E",
}
colors = ["b", "g", "r", "c", "m"]  # Colors for different comm_strats
num_rows_per_matrix = defaultdict(
    lambda: 1,
    {
        "Serena": 1391349,
        "nlpkkt200": 16240000,
        "Bump_2911": 2911419,
        "Cube_Coup_dt0": 2164760,
        "dielFilterV3real": 1102824,
        "Long_Coup_dt0": 1470152,
        "bone010": 986703,
        "af_shell10": 1508065,
    },
)


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


# def parse_file_name(file_name: str):
#     tokens = file_name.split("/")[-1].split("_")
#     name, nodes, tasks, threads, mpi = "", 0, 0, 0, 0
#     if "mpi" in file_name:
#         mpi = -2
#     for i, token in enumerate(tokens):
#         if token == "nodes":
#             name = "_".join(tokens[: i - 1])
#             nodes = int(tokens[i - 1])
#         elif token == "tasks":
#             tasks = int(tokens[i - 1])
#         elif "threads" in token:
#             threads = int(tokens[i - 1])
#     return name, nodes, tasks, threads, mpi


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

    # if tasks > 1 or "mpi" in file_name:
    #     mpi = 1
    name = "_".join(name.split("_")[1:])
    return comm_strat, name, nodes, tasks, threads, mpi


def parse_file_contents(file_name: str):
    ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg = 0, 0, 0, 0, 0, 0, 0
    with open(file_name, "r") as f:
        for line in f:
            if "time:" in line or "GFLOPS:" in line:
                continue
            tokens = line.split()
            if "Total time" in line:
                ttot = float(tokens[3][:-1])
            elif "Communication time" in line:
                tcomm = float(tokens[3][:-1])
            elif "Copmutation time" in line:
                tcomp = float(tokens[3][:-1])
            elif "GFLOPS" in line and "comp" not in line:
                gflops = float(tokens[2])
            elif "Comm min" in line:
                comm_min = float(tokens[3])
            elif "Comm max" in line:
                comm_max = float(tokens[3])
            elif "Comm avg" in line:
                comm_avg = float(tokens[3])
        return ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg


def gather_results(directory, results):
    latest_files = defaultdict(lambda: ("", 0))

    for file in directory.iterdir():
        print(file.name)

        if (
            not file.is_file()
            or "stderr" in file.name
            or "mpi" not in file.name
            # or "nlpkkt200" not in file.name
            # or "Bump" not in file.name
        ):
            continue
        valid_result = False
        flops = 0
        for line in file.open():
            if "GFLOPS" in line:
                flops = float(line.split()[2])
                if flops > 0.005 and flops < 1000:
                    valid_result = True
                    break
        if not valid_result:
            print(f"Invalid result at {file}")
            continue

        config, job_id, _ = file.name.rsplit("-", 2)
        job_id = int(job_id)

        if job_id > latest_files[config][1]:
            latest_files[config] = (str(file), job_id)  # type: ignore

    files = set([file for (file, _) in latest_files.values()])

    for config, (file, job_id) in latest_files.items():
        comm_strat, name, nodes, tasks, threads, mpi = parse_file_name_single(str(file))
        if threads != 32:
            continue
        ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg = parse_file_contents(str(file))
        tcomm /= 100
        tcomp /= 100
        ttot /= 100
        comm_min = (comm_min * (1024 * 1024 * 1024)) / (100 * 64)
        comm_max = (comm_max * (1024 * 1024 * 1024)) / (100 * 64)
        comm_avg = (comm_avg * (1024 * 1024 * 1024)) / (100 * 64)
        comm_min /= num_rows_per_matrix[name]
        comm_avg /= num_rows_per_matrix[name]
        comm_max /= num_rows_per_matrix[name]

        # if comm_strat == "2d":
        # gflops *= 1.5
        # if gflops > 1000:
        #     continue
        # if tasks > 2:
        #     continue

        if nodes == 1 and tasks == 1:
            comm_min = 0
            comm_max = 0
            comm_avg = 0
        results[comm_strat][name][mpi].append(
            SingleResult(
                nodes, tasks, threads, ttot, tcomm, tcomp, gflops, comm_min, comm_max, comm_avg
            )
        )

    # for f in sorted(files):
    #     print(f)


def plot_gflops_single(results, single):
    # colors = ["b", "g", "r", "c", "m"]  # Colors for different comm_strats
    mpi_labels = ["1 process per node", "1 process per socket"]

    for matrix in results[next(iter(results))]:  # Loop over matrix names
        for mpi in [0, 1]:  # Loop over MPI modes
            if mpi:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            # fig.suptitle(f"{matrix} - {mpi_labels[mpi]}")

            # fig.suptitle(f"{matrix}")
            ax.set_xlabel(argv[3])
            ax.set_ylabel("GFLOPS")

            for i, (comm_strat, matrices) in enumerate(sorted(results.items())):
                if matrix in matrices and mpi in matrices[matrix]:
                    sorted_results = []
                    nodes = []
                    if single:
                        sorted_results = sorted(matrices[matrix][mpi], key=lambda res: res.tasks)
                        nodes = [res.tasks for res in sorted_results]
                    else:
                        sorted_results = sorted(matrices[matrix][mpi], key=lambda res: res.nodes)
                        nodes = [res.nodes for res in sorted_results]
                    tcomm = [res.gflops for res in sorted_results]

                    ax.plot(
                        nodes,
                        tcomm,
                        marker="o",
                        linestyle="-",
                        color=colors[i % len(colors)],
                        label=comm_strats_dict.get(comm_strat, comm_strat),
                    )

            if single:
                ax.legend(loc="upper left")
                ax.set_xscale("log")
                ax.xaxis.set_minor_locator(mticker.NullLocator())
                ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
                ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                plt.savefig(f"{matrix}_gflops_single_{partition}mpi.svg", format="svg")
                # plt.show()
                plt.close()
            else:
                ax.legend(loc="upper left")
                # ax.set_xscale("log")
                ax.xaxis.set_minor_locator(mticker.NullLocator())
                ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
                ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                # plt.savefig(f"{matrix}_gflops_multi_{partition}mpi.svg", format="svg")
                plt.show()
                plt.close()


def plot_time(results):
    comm_strats_dict = {
        "1a": "Exchange entire vector",
        "1b": "Exchange separators",
        "1c": "Exchange required separators",
        "1d": "Exchange required elements",
        "2d": "Exchange required elements, memory scalable",
    }
    # colors = ["b", "g", "r", "c", "m"]  # Colors for different comm_strats
    mpi_labels = ["1 process per node", "1 process per socket"]

    for matrix in results[next(iter(results))]:  # Loop over matrix names
        for mpi in [0, 1]:  # Loop over MPI modes
            if mpi:
                continue
            fig, ax = plt.subplots(figsize=(12, 6))
            # fig.suptitle(f"{matrix} - {mpi_labels[mpi]}")

            fig.suptitle(f"{matrix}")
            ax.set_xlabel(argv[3])
            ax.set_ylabel("Time [s]")

            for i, (comm_strat, matrices) in enumerate(sorted(results.items())):
                if matrix in matrices and mpi in matrices[matrix]:
                    sorted_results = sorted(matrices[matrix][mpi], key=lambda res: res.nodes)
                    nodes = [res.nodes for res in sorted_results]
                    tcomm = [res.t for res in sorted_results]

                    ax.plot(
                        nodes,
                        tcomm,
                        marker="o",
                        linestyle="-",
                        color=colors[i % len(colors)],
                        label=comm_strats_dict.get(comm_strat, comm_strat),
                    )

            if not mpi:
                ax.legend(loc="upper left")
                # ax.set_xscale("log")
                ax.xaxis.set_minor_locator(mticker.NullLocator())
                ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
                ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                # plt.savefig(f"{matrix}_gflops_multi_{partition}.png", dpi=600, bbox_inches="tight")
                plt.show()
                plt.close()


def plot_tcomm_single(results):
    comm_strats = ["1a", "1b", "1c", "1d", "2d"]
    comm_labels = {
        "1a": "Exchange entire vector",
        "1b": "Exchange separators",
        "1c": "Exchange required separators",
        "1d": "Exchange required elements",
        "2d": "Exchange required elements, memory scalable",
    }

    matrix_list = list(next(iter(results.values())).keys())

    for matrix in matrix_list:
        fig, ax = plt.subplots(figsize=(8, 5))
        # ax.set_title(f"GFLOPS vs Tasks for {matrix}")
        ax.set_xlabel(argv[3])
        ax.set_ylabel("Communication time per iteration SpMV [s]")

        for i, strat in enumerate(comm_strats):
            # gather all single-node runs for this matrix & strat
            runs_all = []
            for mpi_flag, runs in results.get(strat, {}).get(matrix, {}).items():
                for res in runs:
                    if getattr(res, "nodes", 1) != 1:
                        continue
                    # <— always use res.tasks here
                    runs_all.append((res.tasks, res.tcomm))

            if not runs_all:
                continue

            # sort by task count
            runs_all.sort(key=lambda t: t[0])
            tasks, tcomm = zip(*runs_all)

            ax.plot(
                tasks,
                tcomm,
                marker="o",
                linestyle="-",
                color=colors[i],
                label=comm_labels[strat],
            )

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="best")
        # ax.set_xlim(left=1)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{matrix}_tcomm_multi_defq.png", dpi=600, bbox_inches="tight")
        plt.close()


def plot_compare_comm_strat_split(results):
    comm_strats_dict = {
        "1a": "Exchange entire vector",
        "1b": "Exchange separators",
        "1c": "Exchange required separators",
        "1d": "Exchange required elements",
        "2d": "Exchange required elements, memory scalable",
    }
    # colors = ["b", "g", "r", "c", "m"]  # Colors for different comm_strats
    mpi_labels = ["1 process per node", "1 process per socket"]

    for matrix in results[next(iter(results))]:  # Loop over matrix names
        for mpi in [0, 1]:  # Loop over MPI modes
            if mpi:
                continue
            fig, ax = plt.subplots(figsize=(5, 5))
            # fig.suptitle(f"{matrix} - {mpi_labels[mpi]}")

            ax.set_xlabel(argv[3])
            ax.set_ylabel("GFLOPS")

            for i, (comm_strat, matrices) in enumerate(sorted(results.items())):
                if matrix in matrices and mpi in matrices[matrix]:
                    sorted_results = sorted(matrices[matrix][mpi], key=lambda res: res.nodes)
                    nodes = [res.nodes for res in sorted_results]
                    gflops = [res.gflops for res in sorted_results]

                    ax.plot(
                        nodes,
                        gflops,
                        marker="o",
                        linestyle="-",
                        color=colors[i % len(colors)],
                        label=comm_strats_dict.get(comm_strat, comm_strat),
                    )

            if not mpi:
                ax.legend()
                ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                # plt.savefig(f"{matrix}_rome16q.png", dpi=600, bbox_inches="tight")
                # plt.show()
                plt.close()


def plot_tcomm_multi(results, single):
    comm_strats_dict = {
        "1a": "Exchange entire vector",
        "1b": "Exchange separators",
        "1c": "Exchange required separators",
        "1d": "Exchange required elements",
        "2d": "Exchange required elements, memory scalable",
    }
    # colors = ["b", "g", "r", "c", "m"]  # Colors for different comm_strats
    mpi_labels = ["1 process per node", "1 process per socket"]

    for matrix in results[next(iter(results))]:  # Loop over matrix names
        for mpi in [0, 1]:  # Loop over MPI modes
            if mpi:
                continue
            fig, ax = plt.subplots(figsize=(6, 6))
            # fig.suptitle(f"{matrix} - {mpi_labels[mpi]}")

            # fig.suptitle(f"{matrix}")
            ax.set_xlabel(argv[3])
            ax.set_ylabel("Communication time per iteration SpMV")

            for i, (comm_strat, matrices) in enumerate(sorted(results.items())):
                if matrix in matrices and mpi in matrices[matrix]:
                    sorted_results = []
                    nodes = []
                    if single:
                        sorted_results = sorted(matrices[matrix][mpi], key=lambda res: res.tasks)
                        nodes = [res.tasks for res in sorted_results]
                    else:
                        sorted_results = sorted(matrices[matrix][mpi], key=lambda res: res.nodes)
                        nodes = [res.nodes for res in sorted_results]

                    tcomm = [res.tcomm for res in sorted_results]

                    ax.plot(
                        nodes,
                        tcomm,
                        marker="o",
                        linestyle="-",
                        color=colors[i % len(colors)],
                        label=comm_strats_dict.get(comm_strat, comm_strat),
                    )

            if single:
                ax.legend(loc="upper left")
                ax.set_xscale("log")
                ax.xaxis.set_minor_locator(mticker.NullLocator())
                # ax.set_xlim(left=1)
                ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
                ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                plt.savefig(f"{matrix}_tcomm_single_{partition}mpi.svg", format="svg")
                # plt.show()
                plt.close()
            else:
                ax.legend(loc="upper left")
                # ax.set_xscale("log")
                ax.xaxis.set_minor_locator(mticker.NullLocator())
                # ax.set_xlim(left=1)
                ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
                ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])
                ax.grid(True, linestyle="--", alpha=0.6)
                plt.tight_layout()
                # plt.savefig(f"{matrix}_tcomm_multi_{partition}mpi.svg", format="svg")
                plt.show()
                plt.close()


def plot_compare_comm_strat(results):
    for matrix in results[next(iter(results))]:  # Iterate over matrix names
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
        fig.suptitle(f"GFLOPS Comparison for {matrix}")

        mpi_labels = ["1 process per node", "1 process per socket"]
        # colors = ["b", "g", "r", "c", "m"]  # Colors for different comm_strats
        comm_strats_dict = {
            "1a": "Exchange entire vector",
            "1b": "Exchange separators",
            "1c": "Exchange required separators",
            "1d": "Exchange required elements",
            "2d": "Exchange required elements, memory scalable",
        }

        for mpi, ax in enumerate(axes):
            ax.set_title(mpi_labels[mpi])
            ax.set_xlabel(argv[3])
            ax.set_ylabel("GFLOPS" if mpi == 0 else "")  # Only label y-axis on the left

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
                        label=comm_strats_dict[comm_strat],
                    )

            ax.grid(True, linestyle="--", alpha=0.6)

        axes[0].legend()  # Only add legend to the right plot
        plt.tight_layout()
        plt.savefig(f"{matrix}_{partition}.png", dpi=600, bbox_inches="tight")
        # plt.show()
        plt.close()


# def plot_compare_comm_strat(results):
#     for matrix in results[next(iter(results))]:  # Iterate over matrix names
#         fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
#         fig.suptitle(f"GFLOPS Comparison for {matrix}")

#         mpi_labels = ["1 process per node", "1 process per socket"]
#         # colors = ["b", "g", "r", "c", "m"]  # Colors for different comm_strats
#         comm_strats_dict = {
#             "1a": "Exchange entire vector",
#             "1b": "Exchange separators",
#             "1c": "Exchange required separators",
#             "1d": "Exchange required elements",
#             "2d": "Exchange required elements, memory scalable",
#         }

#         for mpi, ax in enumerate(axes):
#             ax.set_title(mpi_labels[mpi])
#             ax.set_xlabel(argv[3])
#             ax.set_ylabel("GFLOPS" if mpi == 0 else "")  # Only label y-axis on the left

#             for i, (comm_strat, matrices) in enumerate(results.items()):

#                 if matrix in matrices and mpi in matrices[matrix]:
#                     # Sort the results by number of nodes in ascending order
#                     sorted_results = sorted(matrices[matrix][mpi], key=lambda res: res.nodes)
#                     nodes = [res.nodes for res in sorted_results]
#                     gflops = [res.gflops for res in sorted_results]

#                     ax.plot(
#                         nodes,
#                         gflops,
#                         marker="o",
#                         linestyle="-",
#                         color=colors[i % len(colors)],
#                         label=comm_strats_dict[comm_strat],
#                     )

#             ax.grid(True, linestyle="--", alpha=0.6)

#         axes[0].legend()  # Only add legend to the right plot
#         plt.tight_layout()
#         # plt.savefig(f"{matrix}_{partition}.png", dpi=600, bbox_inches="tight")
#         plt.show()
#         plt.close()


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
    # colors = ["b", "g", "r", "c", "m", "y", "k"]
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


# def plot_comm_and_comp_time(results):
#     comm_strats = list(results.keys())

#     for matrix in results[next(iter(results))]:  # Iterate over matrix names
#         fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
#         fig.suptitle(f"Communication and Computation Time for {matrix}", fontsize=16)

#         node_counts = sorted(
#             {
#                 res.nodes
#                 for comm_strat in results.values()
#                 for mat_name, mpi_data in comm_strat.items()
#                 if mat_name == matrix
#                 for res_list in mpi_data.values()
#                 for res in res_list
#             }
#         )

#         bar_width = 0.15
#         index = np.arange(len(node_counts))
#         # colors = {"comm": "skyblue", "comp": "sandybrown"}
#         mpi_labels = ["Single MPI Process", "Dual MPI Processes"]

#         for mpi in [0, 1]:
#             ax = axes[mpi]
#             ax.set_title(mpi_labels[mpi])
#             ax.set_xlabel("Number of Nodes")
#             if mpi == 0:
#                 ax.set_ylabel("Time (s)")
#             ax.set_xticks(index + ((len(comm_strats) - 1) / 2) * bar_width)
#             ax.set_xticklabels([f"{n}" for n in node_counts])
#             ax.grid(True, linestyle="--", alpha=0.6)

#             for i, comm_strat in enumerate(comm_strats):
#                 if matrix not in results[comm_strat] or mpi not in results[comm_strat][matrix]:
#                     continue
#                 strat_results = results[comm_strat][matrix][mpi]

#                 # Map node count -> list of times
#                 time_by_node = defaultdict(list)
#                 for res in strat_results:
#                     time_by_node[res.nodes].append(res)

#                 comm_times = []
#                 comp_times = []
#                 for node in node_counts:
#                     entries = time_by_node[node]
#                     if entries:
#                         avg_comm = np.mean([r.tcomm for r in entries])
#                         avg_comp = np.mean([r.tcomp for r in entries])
#                     else:
#                         avg_comm = 0
#                         avg_comp = 0
#                     comm_times.append(avg_comm)
#                     comp_times.append(avg_comp)

#                 pos = index + i * bar_width
#                 ax.bar(pos, comm_times, bar_width, label=f"{comm_strat} comm", color=colors["comm"])
#                 ax.bar(
#                     pos,
#                     comp_times,
#                     bar_width,
#                     bottom=comm_times,
#                     label=f"{comm_strat} comp",
#                     color=colors["comp"],
#                 )

#         handles = [
#             Patch(facecolor=colors["comm"], label="Communication"),
#             Patch(facecolor=colors["comp"], label="Computation"),
#         ]
#         labels = ["Communication", "Computation"]
#         fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.94))
#         plt.tight_layout()
#         plt.show()


def plot_comm_min_avg_max_nonmpi_single(results):
    # Exclude strategies '1a' and '2d'
    comm_strats = sorted(k for k in results.keys() if k not in ("1a", "2d"))
    markers = {"min": "o", "avg": "s", "max": "^"}
    labels = ["min", "avg", "max"]
    comm_colors = {cs: colors[i % len(colors)] for i, cs in enumerate(comm_strats)}

    for matrix in results[next(iter(results))]:  # Loop over matrix names
        # Collect all runs for mpi=0 & single-node
        runs_by_strat = {
            strat: [r for r in results[strat][matrix].get(0, []) if getattr(r, "nodes", 1) == 1]
            for strat in comm_strats
        }

        # Determine all unique task counts across strategies
        all_tasks = sorted({r.tasks for runs in runs_by_strat.values() for r in runs})

        fig, ax = plt.subplots(figsize=(10, 6))
        # ax.set_title(f"Communication Load for {matrix} (single node)")
        ax.set_xlabel(argv[3])
        ax.set_ylabel("\% of $x$ communicated per iteration SpMV")

        added_labels = set()

        # For each strategy, scatter min/avg/max at each task count
        for strat in comm_strats:
            runs = runs_by_strat[strat]
            if not runs:
                continue

            for t in all_tasks:
                data = [r for r in runs if r.tasks == t]
                if not data:
                    continue

                comm_mins = [r.comm_min for r in data]
                comm_avgs = [r.comm_avg for r in data]
                comm_maxs = [r.comm_max for r in data]

                for vals, label in zip([comm_mins, comm_avgs, comm_maxs], labels):
                    leg = f"{strat} ({label})"
                    ax.plot(
                        [t] * len(vals),
                        vals,
                        marker=markers[label],
                        linestyle="None",
                        color=comm_colors[strat],
                        label=leg if leg not in added_labels else None,
                    )
                    added_labels.add(leg)

        ax.set_xticks(all_tasks)
        ax.set_xscale("log")
        # ax.set_xscale("log")
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        ax.grid(True, linestyle="--", alpha=0.5)

        # Deduplicate legend entries
        handles, labls = ax.get_legend_handles_labels()
        by_label = dict(zip(labls, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="best")

        plt.tight_layout()
        plt.show()
        # plt.savefig(f"{matrix}_defq_commload_single.png", dpi=600, bbox_inches="tight")
        plt.close()


def plot_comm_min_avg_max(results, single):
    # Exclude strategies '1a' and '2d'
    comm_strats = sorted(k for k in results.keys() if k not in ("1a", "2d"))
    markers = {"min": "o", "avg": "s", "max": "^"}
    labels = ["min", "avg", "max"]
    colors = ["g", "r", "c"]
    comm_colors = {cs: colors[i % len(colors)] for i, cs in enumerate(comm_strats)}

    comm_strats_dict = {
        "1a": "Exchange entire vector",
        "1b": "Exchange separators",
        "1c": "Exchange required separators",
        "1d": "Exchange required elements",
        "2d": "Exchange required elements, memory scalable",
    }

    for matrix in results[next(iter(results))]:  # Loop over matrix names
        # Collect all runs for mpi=0 & single-node
        runs_by_strat = {
            strat: [r for r in results[strat][matrix].get(0, [])] for strat in comm_strats
        }

        # Determine all unique task counts across strategies
        all_tasks = sorted(
            {r.tasks if single else r.nodes for runs in runs_by_strat.values() for r in runs}
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        # ax.set_title(f"Communication Load for {matrix} (single node)")
        ax.set_xlabel(argv[3])
        ax.set_ylabel("Fraction of $x$ communicated per iteration SpMV")

        added_labels = set()

        # For each strategy, scatter min/avg/max at each task count
        for strat in comm_strats:
            runs = runs_by_strat[strat]
            if not runs:
                continue

            for t in all_tasks:
                data = []
                if single:
                    data = [r for r in runs if r.tasks == t]
                else:
                    data = [r for r in runs if r.nodes == t]
                if not data:
                    continue

                comm_mins = [r.comm_min for r in data]
                comm_avgs = [r.comm_avg for r in data]
                comm_maxs = [r.comm_max for r in data]

                for vals, label in zip([comm_mins, comm_avgs, comm_maxs], labels):
                    leg = f"{strat} ({label})"
                    ax.plot(
                        [t] * len(vals),
                        vals,
                        marker=markers[label],
                        linestyle="None",
                        color=comm_colors[strat],
                        label=(
                            f"{comm_strats_dict[leg.split()[0]]} {leg.split()[1]}"
                            if leg not in added_labels
                            else None
                        ),
                    )
                    added_labels.add(leg)

        if single:
            ax.set_xticks(all_tasks)
            # ax.set_xscale("log")
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            # ax.set_xlim(left=1)
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
            ax.grid(True, linestyle="--", alpha=0.5)

            # Deduplicate legend entries
            handles, labls = ax.get_legend_handles_labels()
            by_label = dict(zip(labls, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="best")

            plt.tight_layout()
            # plt.show()
            plt.savefig(f"{matrix}_commload_single_{partition}mpi.svg", format="svg")
            plt.close()
        else:
            ax.set_xticks(all_tasks)
            # ax.set_xscale("log")
            # ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            # ax.set_xlim(left=1)
            ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8])
            ax.set_xticklabels([1, 2, 3, 4, 5, 6, 7, 8])
            ax.grid(True, linestyle="--", alpha=0.5)

            # Deduplicate legend entries
            handles, labls = ax.get_legend_handles_labels()
            by_label = dict(zip(labls, handles))
            ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="best")

            plt.tight_layout()
            plt.show()
            # plt.savefig(f"{matrix}_commload_multi_{partition}mpi.svg", format="svg")
            plt.close()


def plot_comm_min_avg_max_nonmpi(results):
    # Exclude strategies '1a' and '2d'
    comm_strats = sorted(k for k in results.keys() if k not in ("1a", "2d"))
    markers = {"min": "o", "avg": "s", "max": "^"}
    labels = ["min", "avg", "max"]
    comm_colors = {cs: colors[i % len(colors)] for i, cs in enumerate(comm_strats)}

    for matrix in results[next(iter(results))]:
        # collect all non‐MPI runs (mpi_flag=0), across all nodes
        runs_by_strat = {strat: results[strat][matrix].get(0, []) for strat in comm_strats}

        # all unique node counts
        all_nodes = sorted({r.nodes for runs in runs_by_strat.values() for r in runs})

        fig, ax = plt.subplots(figsize=(10, 6))
        # ax.set_title(f"Communication Load for {matrix} (non-MPI)")
        ax.set_xlabel(argv[3])
        ax.set_ylabel("Communication Volume [GB]")

        added_labels = set()

        for strat in comm_strats:
            runs = runs_by_strat[strat]
            if not runs:
                continue

            for n in all_nodes:
                data = [r for r in runs if r.nodes == n]
                if not data:
                    continue

                comm_mins = [r.comm_min for r in data]
                comm_avgs = [r.comm_avg for r in data]
                comm_maxs = [r.comm_max for r in data]

                for vals, label in zip([comm_mins, comm_avgs, comm_maxs], labels):
                    leg = f"{strat} ({label})"
                    ax.plot(
                        [n] * len(vals),
                        vals,
                        marker=markers[label],
                        linestyle="None",
                        color=comm_colors[strat],
                        label=leg if leg not in added_labels else None,
                    )
                    added_labels.add(leg)

        ax.set_xticks(all_nodes)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.set_xlabel(argv[3])
        ax.set_ylabel("\% of $x$ communicated per iteration SpMV")

        # dedupe legend
        handles, labls = ax.get_legend_handles_labels()
        by_label = dict(zip(labls, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=9, loc="best")
        ax.set_xscale("log")

        plt.tight_layout()
        plt.show()
        # plt.savefig(f"{matrix}_rome16q_commload.png", dpi=600, bbox_inches="tight")
        plt.close()


# def plot_comm_min_avg_max(results):
#     # Exclude strategy '1a' from plotting
#     comm_strats = sorted(k for k in results.keys() if k != "1a" and k != "2d")
#     markers = {"min": "o", "avg": "s", "max": "^"}
#     labels = ["min", "avg", "max"]

#     # Assign distinct colors to the remaining strategies
#     # color_list = ["red", "green", "magenta", "orange", "cyan"]
#     comm_colors = {comm_strat: colors[i % len(colors)] for i, comm_strat in enumerate(comm_strats)}

#     for matrix in results[next(iter(results))]:  # Loop over matrix names
#         fig, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
#         fig.suptitle(f"Communication Load for {matrix}", fontsize=16)

#         for ax_index, mpi in enumerate([0, 1]):
#             ax = axs[ax_index]
#             mode = "1 process per socket" if mpi == 0 else "1 process per node"
#             ax.set_title(mode)

#             # Get unique node counts
#             node_counts = sorted(
#                 {
#                     res.nodes
#                     for comm_strat in results.values()
#                     for mat_name, mpi_data in comm_strat.items()
#                     if mat_name == matrix
#                     for res in mpi_data.get(mpi, [])
#                 }
#             )

#             xticks = []
#             xticklabels = []
#             x_pos = 0
#             added_labels = set()

#             for n in node_counts:
#                 for comm_strat in comm_strats:
#                     matching_results = results[comm_strat][matrix].get(mpi, [])
#                     data = [res for res in matching_results if res.nodes == n]
#                     if not data:
#                         x_pos += 1
#                         continue

#                     comm_mins = [r.comm_min for r in data]
#                     comm_avgs = [r.comm_avg for r in data]
#                     comm_maxs = [r.comm_max for r in data]

#                     for val, label in zip([comm_mins, comm_avgs, comm_maxs], labels):
#                         legend_label = f"{comm_strat} ({label})"
#                         ax.plot(
#                             [x_pos] * len(val),
#                             val,
#                             marker=markers[label],
#                             linestyle="None",
#                             color=comm_colors[comm_strat],
#                             # alpha=0.7,
#                             label=legend_label if legend_label not in added_labels else None,
#                         )
#                         added_labels.add(legend_label)

#                     xticks.append(x_pos)
#                     xticklabels.append(f"{n}\n{comm_strat}")
#                     x_pos += 1

#                 x_pos += 1  # Extra space between node count groups

#             ax.set_xticks(xticks)
#             ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=9)
#             ax.set_xlabel("Number of nodes and communication strategy")
#             ax.set_yscale("log")
#             # ax.set_xscale("log")
#             ax.grid(True, linestyle="--", alpha=0.5)

#             if ax_index == 0:
#                 ax.set_ylabel("Communication Load [GB]")

#             handles, labels_ = ax.get_legend_handles_labels()
#             by_label = dict(zip(labels_, handles))
#             ax.legend(by_label.values(), by_label.keys(), fontsize=9)

#         plt.tight_layout()
#         plt.show()


def plot_tcomm_2x4(results, single):
    matrix_names = []
    first_strat = next(iter(results))
    for matrix in sorted(results[first_strat]):
        for strat_data in results.values():
            if matrix in strat_data and 0 in strat_data[matrix]:
                matrix_names.append(matrix)
                break

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) Bigger figure + grid of 4×2 subplots
    fig, axs = plt.subplots(4, 2, figsize=(18, 28))
    axs = axs.flatten()

    all_lines = {}

    for idx, matrix in enumerate(sorted(matrix_names[:8])):
        ax = axs[idx]
        ax.set_title(matrix)  # uses axes.titlesize
        ax.set_xlabel(argv[3])  # uses axes.labelsize
        if idx % 2 == 0:
            ax.set_ylabel("Communication Time [ms]")

        # plot each strategy
        for i, comm_strat in enumerate(sorted(results)):
            strat_mats = results[comm_strat]
            if matrix not in strat_mats or 0 not in strat_mats[matrix]:
                continue

            # sort by tasks (single) or nodes (multi)
            data = sorted(strat_mats[matrix][0], key=lambda r: r.tasks if single else r.nodes)
            xs = [r.tasks if single else r.nodes for r in data]
            ys = [r.tcomm * 1000 for r in data]

            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                markersize=9,
                linestyle="-",
                color=colors[i % len(colors)],
                label=comm_strats_dict.get(comm_strat, comm_strat),
            )
            all_lines.setdefault(comm_strat, line)

        # log‐scale only for the "single" case
        if single:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        else:
            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(range(1, 5))

        # ensure the tick labels pick up our rcParams
        ax.tick_params(axis="both", which="major")
        ax.grid(True, linestyle="--", alpha=0.6)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) Tweak spacing to make room at top and between panels
    fig.subplots_adjust(
        top=0.85,  # room for legend
        hspace=0.35,  # vertical space between rows
        wspace=0.2,  # horizontal space between cols
    )

    # 4) Shared legend across the top
    handles = [all_lines[k] for k in all_lines]
    labels = [comm_strats_dict.get(k, k) for k in all_lines]
    left = fig.subplotpars.left  # e.g. 0.10
    right = fig.subplotpars.right  # e.g. 0.90
    width = right - left  # e.g. 0.80

    # 2) pass a full 4‐tuple to bbox_to_anchor + mode="expand"
    fig.legend(
        handles,
        labels,
        loc="upper center",  # anchor corner of legend box
        bbox_to_anchor=(0.5, 0.94, 0, 0),  # x0, y0, width, height (height can be zero)
        # mode="expand",  # stretch legend horizontally to fill the width
        ncol=2,  # 3 per row → 2 rows for 5 items
        frameon=False,
    )
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     # bbox_to_anchor=(0.5, 0.93),  # just below the top edge
    #     ncol=min(len(handles), 3),
    #     frameon=True,
    # )

    # 5) Save with tight bounding box
    plt.savefig(
        f"tcomm_2x4_{'single' if single else 'multi'}_{partition}mpi.svg",
        bbox_inches="tight",
        format="svg",
    )


def plot_gflops_2x4(results, single):
    matrix_names = []
    first_strat = next(iter(results))
    for matrix in sorted(results[first_strat]):
        for strat_data in results.values():
            if matrix in strat_data and 0 in strat_data[matrix]:
                matrix_names.append(matrix)
                break

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) Bigger figure + grid of 4×2 subplots
    fig, axs = plt.subplots(4, 2, figsize=(18, 28))
    axs = axs.flatten()

    all_lines = {}

    for idx, matrix in enumerate(sorted(matrix_names[:8])):
        ax = axs[idx]
        ax.set_title(matrix)  # uses axes.titlesize
        ax.set_xlabel(argv[3])  # uses axes.labelsize
        if idx % 2 == 0:
            ax.set_ylabel("GFLOPS")

        # plot each strategy
        for i, comm_strat in enumerate(sorted(results)):
            strat_mats = results[comm_strat]
            if matrix not in strat_mats or 0 not in strat_mats[matrix]:
                continue

            # sort by tasks (single) or nodes (multi)
            data = sorted(strat_mats[matrix][0], key=lambda r: r.tasks if single else r.nodes)
            xs = [r.tasks if single else r.nodes for r in data]
            ys = [r.gflops for r in data]

            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                markersize=9,
                linestyle="-",
                color=colors[i % len(colors)],
                label=comm_strats_dict.get(comm_strat, comm_strat),
            )
            all_lines.setdefault(comm_strat, line)

        # log‐scale only for the "single" case
        if single:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        else:
            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(range(1, 5))

        # ensure the tick labels pick up our rcParams
        ax.tick_params(axis="both", which="major")
        ax.grid(True, linestyle="--", alpha=0.6)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) Tweak spacing to make room at top and between panels
    fig.subplots_adjust(
        top=0.85,  # room for legend
        hspace=0.35,  # vertical space between rows
        wspace=0.2,  # horizontal space between cols
    )

    # 4) Shared legend across the top
    handles = [all_lines[k] for k in all_lines]
    labels = [comm_strats_dict.get(k, k) for k in all_lines]
    left = fig.subplotpars.left  # e.g. 0.10
    right = fig.subplotpars.right  # e.g. 0.90
    width = right - left  # e.g. 0.80

    # 2) pass a full 4‐tuple to bbox_to_anchor + mode="expand"
    fig.legend(
        handles,
        labels,
        loc="upper center",  # anchor corner of legend box
        bbox_to_anchor=(0.5, 0.94, 0, 0),  # x0, y0, width, height (height can be zero)
        # mode="expand",  # stretch legend horizontally to fill the width
        ncol=2,  # 3 per row → 2 rows for 5 items
        frameon=False,
    )
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     # bbox_to_anchor=(0.5, 0.93),  # just below the top edge
    #     ncol=min(len(handles), 3),
    #     frameon=True,
    # )

    # 5) Save with tight bounding box
    plt.savefig(
        f"gflops_2x4_{'single' if single else 'multi'}_{partition}mpi.svg",
        bbox_inches="tight",
        format="svg",
    )


def plot_comm_min_avg_max_2x4(results, single):
    # ─────────────────────────────────────────────────────────────────────────────
    # 0) Strategy definitions
    comm_strats = sorted(k for k in results.keys() if k not in ("1a", "2d"))
    markers = {"avg": "o", "max": "^"}
    labels = ["avg", "max"]
    colors = ["g", "r", "c"]
    comm_colors = {cs: colors[i % len(colors)] for i, cs in enumerate(comm_strats)}

    # ─────────────────────────────────────────────────────────────────────────────
    # 1) Pick out the first 8 matrices with mpi=0 data
    first_strat = next(iter(results))
    matrix_names = []
    for matrix in sorted(results[first_strat]):
        for strat_data in results.values():
            if matrix in strat_data and 0 in strat_data[matrix]:
                matrix_names.append(matrix)
                break
    matrix_names = matrix_names[:8]

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) Create a 4×2 grid of subplots
    fig, axs = plt.subplots(4, 2, figsize=(18, 28))
    axs = axs.flatten()

    # for l in sorted(map(lambda x: x.lower, matrix_names)):
    #     print(l)
    # 3) Plot each matrix in its own subplot
    for idx, matrix in enumerate(sorted(matrix_names)):
        ax = axs[idx]
        ax.set_title(matrix)
        ax.set_xlabel(argv[3])
        if idx % 2 == 0:
            ax.set_ylabel("Fraction of $x$ communicated\nper iteration SpMV")

        # gather runs for mpi=0
        runs_by_strat = {strat: results[strat][matrix].get(0, []) for strat in comm_strats}
        # all unique rank counts
        all_tasks = sorted(
            {(r.tasks if single else r.nodes) for runs in runs_by_strat.values() for r in runs}
        )

        seen = set()
        for strat in comm_strats:
            runs = runs_by_strat[strat]
            if not runs:
                continue

            for t in all_tasks:
                group = [r for r in runs if (r.tasks if single else r.nodes) == t]
                if not group:
                    continue

                # comm_min = [r.comm_min for r in group]
                comm_avg = [r.comm_avg for r in group]
                comm_max = [r.comm_max for r in group]

                for vals, lab in zip([comm_avg, comm_max], labels):
                    legend_label = f"{comm_strats_dict[strat]} ({lab})"
                    plot_label = legend_label if legend_label not in seen else "_nolegend_"
                    ax.plot(
                        [t] * len(vals),
                        vals,
                        marker=markers[lab],
                        markersize=9,
                        linestyle="None",
                        color=comm_colors[strat],
                        label=plot_label,
                    )
                    seen.add(legend_label)

        # x-axis formatting
        if single:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        else:
            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(range(1, 5))

        ax.tick_params(axis="both", which="major")
        ax.grid(True, linestyle="--", alpha=0.5)

    # hide any unused subplots
    for ax in axs[len(matrix_names) :]:
        ax.axis("off")

    # ─────────────────────────────────────────────────────────────────────────────
    # 4) Tweak spacing to make room for a full-width legend
    legend_y = 0.94  # vertical position of legend
    fig.subplots_adjust(
        top=0.85, hspace=0.35, wspace=0.2
    )  # leave a bit of headroom above the plots

    # 5) Shared legend exactly matching the width of the 4×2 grid
    #    (same trick as in your original function)
    left = fig.subplotpars.left
    right = fig.subplotpars.right
    width = right - left

    # collect one set of handles & labels from the axes
    handles, labls = [], []
    for ax in axs[: len(matrix_names)]:
        h, l = ax.get_legend_handles_labels()
        for hi, li in zip(h, l):
            if li not in labls:
                handles.append(hi)
                labls.append(li)

    fig.legend(
        handles,
        labls,
        loc="upper left",  # ← pin the *left* side
        bbox_to_anchor=(0.1, legend_y, width, 0),
        mode="expand",
        ncol=3,
        frameon=False,
        handletextpad=0.2,  # less space between symbol and label
        columnspacing=0.8,  # less space between columns
        handlelength=1,  # (optional) shorter symbol line
    )

    # ─────────────────────────────────────────────────────────────────────────────
    # 6) Save and close
    # plt.show()
    plt.savefig(
        f"commload_2x4_{'single' if single else 'multi'}_{partition}mpi.svg",
        bbox_inches="tight",
        format="svg",
    )
    plt.close()


def plot_t_2x4(results, single):
    matrix_names = []
    first_strat = next(iter(results))
    for matrix in sorted(results[first_strat]):
        for strat_data in results.values():
            if matrix in strat_data and 0 in strat_data[matrix]:
                matrix_names.append(matrix)
                break

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) Bigger figure + grid of 4×2 subplots
    fig, axs = plt.subplots(4, 2, figsize=(18, 28))
    axs = axs.flatten()

    all_lines = {}

    for idx, matrix in enumerate(sorted(matrix_names[:8])):
        ax = axs[idx]
        ax.set_title(matrix)  # uses axes.titlesize
        ax.set_xlabel(argv[3])  # uses axes.labelsize
        if idx % 2 == 0:
            ax.set_ylabel("Total Time [ms]")

        # plot each strategy
        for i, comm_strat in enumerate(sorted(results)):
            strat_mats = results[comm_strat]
            if matrix not in strat_mats or 0 not in strat_mats[matrix]:
                continue

            # sort by tasks (single) or nodes (multi)
            data = sorted(strat_mats[matrix][0], key=lambda r: r.tasks if single else r.nodes)
            xs = [r.tasks if single else r.nodes for r in data]
            ys = [r.t * 1000 for r in data]

            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                markersize=9,
                linestyle="-",
                color=colors[i % len(colors)],
                label=comm_strats_dict.get(comm_strat, comm_strat),
            )
            all_lines.setdefault(comm_strat, line)

        # log‐scale only for the "single" case
        if single:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        else:
            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(range(1, 5))

        # ensure the tick labels pick up our rcParams
        ax.tick_params(axis="both", which="major")
        ax.grid(True, linestyle="--", alpha=0.6)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) Tweak spacing to make room at top and between panels
    fig.subplots_adjust(
        top=0.85,  # room for legend
        hspace=0.35,  # vertical space between rows
        wspace=0.2,  # horizontal space between cols
    )

    # 4) Shared legend across the top
    handles = [all_lines[k] for k in all_lines]
    labels = [comm_strats_dict.get(k, k) for k in all_lines]
    left = fig.subplotpars.left  # e.g. 0.10
    right = fig.subplotpars.right  # e.g. 0.90
    width = right - left  # e.g. 0.80

    # 2) pass a full 4‐tuple to bbox_to_anchor + mode="expand"
    fig.legend(
        handles,
        labels,
        loc="upper center",  # anchor corner of legend box
        bbox_to_anchor=(0.5, 0.94, 0, 0),  # x0, y0, width, height (height can be zero)
        # mode="expand",  # stretch legend horizontally to fill the width
        ncol=2,  # 3 per row → 2 rows for 5 items
        frameon=False,
    )
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     # bbox_to_anchor=(0.5, 0.93),  # just below the top edge
    #     ncol=min(len(handles), 3),
    #     frameon=True,
    # )

    # 5) Save with tight bounding box
    plt.savefig(
        f"t_2x4_{'single' if single else 'multi'}_{partition}mpi.svg",
        bbox_inches="tight",
        format="svg",
    )


def plot_tcomp_2x4(results, single):
    matrix_names = []
    first_strat = next(iter(results))
    for matrix in sorted(results[first_strat]):
        for strat_data in results.values():
            if matrix in strat_data and 0 in strat_data[matrix]:
                matrix_names.append(matrix)
                break

    # ─────────────────────────────────────────────────────────────────────────────
    # 2) Bigger figure + grid of 4×2 subplots
    fig, axs = plt.subplots(4, 2, figsize=(18, 28))
    axs = axs.flatten()

    all_lines = {}

    for idx, matrix in enumerate(matrix_names[:8]):
        ax = axs[idx]
        ax.set_title(matrix)  # uses axes.titlesize
        ax.set_xlabel(argv[3])  # uses axes.labelsize
        if idx % 2 == 0:
            ax.set_ylabel("Computation Time [ms]")

        # plot each strategy
        for i, comm_strat in enumerate(sorted(results)):
            strat_mats = results[comm_strat]
            if matrix not in strat_mats or 0 not in strat_mats[matrix]:
                continue

            # sort by tasks (single) or nodes (multi)
            data = sorted(strat_mats[matrix][0], key=lambda r: r.tasks if single else r.nodes)
            xs = [r.tasks if single else r.nodes for r in data]
            ys = [r.tcomp * 1000 for r in data]

            (line,) = ax.plot(
                xs,
                ys,
                marker="o",
                markersize=9,
                linestyle="-",
                color=colors[i % len(colors)],
                label=comm_strats_dict.get(comm_strat, comm_strat),
            )
            all_lines.setdefault(comm_strat, line)

        # log‐scale only for the "single" case
        if single:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        else:
            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(range(1, 5))

        # ensure the tick labels pick up our rcParams
        ax.tick_params(axis="both", which="major")
        ax.grid(True, linestyle="--", alpha=0.6)

    # ─────────────────────────────────────────────────────────────────────────────
    # 3) Tweak spacing to make room at top and between panels
    fig.subplots_adjust(
        top=0.85,  # room for legend
        hspace=0.35,  # vertical space between rows
        wspace=0.2,  # horizontal space between cols
    )

    # 4) Shared legend across the top
    handles = [all_lines[k] for k in all_lines]
    labels = [comm_strats_dict.get(k, k) for k in all_lines]
    left = fig.subplotpars.left  # e.g. 0.10
    right = fig.subplotpars.right  # e.g. 0.90
    width = right - left  # e.g. 0.80

    # 2) pass a full 4‐tuple to bbox_to_anchor + mode="expand"
    fig.legend(
        handles,
        labels,
        loc="upper center",  # anchor corner of legend box
        bbox_to_anchor=(0.5, 0.94, 0, 0),  # x0, y0, width, height (height can be zero)
        # mode="expand",  # stretch legend horizontally to fill the width
        ncol=2,  # 3 per row → 2 rows for 5 items
        frameon=False,
    )
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     # bbox_to_anchor=(0.5, 0.93),  # just below the top edge
    #     ncol=min(len(handles), 3),
    #     frameon=True,
    # )

    # 5) Save with tight bounding box
    plt.savefig(
        f"tcomp_2x4_{'single' if single else 'multi'}_{partition}mpi.svg",
        bbox_inches="tight",
        format="svg",
    )


def plot_comm_and_tcomm_2x4(results, single, partition):
    # ──────────────────────────────────────────────────────────────────────
    # 0) Strategy definitions
    comm_strats = sorted(k for k in results.keys() if k not in ("1a", "2d"))
    markers = {"min": "o", "avg": "s", "max": "^"}
    labels = ["min", "avg", "max"]
    colors = ["g", "r", "c"]
    comm_colors = {cs: colors[i % len(colors)] for i, cs in enumerate(comm_strats)}
    tcomm_colors = {cs: colors[i % len(colors)] for i, cs in enumerate(comm_strats)}

    # ──────────────────────────────────────────────────────────────────────
    # 1) Pick out up to 8 matrices with mpi=0
    first_strat = next(iter(results))
    matrix_names = []
    for matrix in sorted(results[first_strat]):
        if any(0 in strat_data.get(matrix, {}) for strat_data in results.values()):
            matrix_names.append(matrix)
    matrix_names = matrix_names[:8]

    # ──────────────────────────────────────────────────────────────────────
    # 2) Create 4×2 grid
    fig, axs = plt.subplots(4, 2, figsize=(18, 28))
    axs = axs.flatten()

    # collect legend entries
    all_handles = []
    all_labels = []

    for idx, matrix in enumerate(sorted(matrix_names)):
        ax = axs[idx]
        ax.set_title(matrix)
        ax.set_xlabel("Problem size")  # or argv[3]
        if idx % 2 == 0:
            ax.set_ylabel("Fraction of $x$ communicated\nper iteration SpMV")

        # twin axis for time
        ax2 = ax.twinx()
        if idx % 2 == 0:
            ax2.set_ylabel("Communication Time [ms]")

        # gather runs for mpi=0
        runs_by_strat = {strat: results[strat][matrix].get(0, []) for strat in comm_strats}
        all_tasks = sorted(
            {(r.tasks if single else r.nodes) for runs in runs_by_strat.values() for r in runs}
        )

        seen = set()
        for strat in comm_strats:
            runs = runs_by_strat[strat]
            if not runs:
                continue

            # --- plot min/avg/max on ax ---
            for lab in labels:
                xs, ys = [], []
                for t in all_tasks:
                    group = [r for r in runs if (r.tasks if single else r.nodes) == t]
                    if not group:
                        continue
                    vals = [getattr(r, f"comm_{lab}") for r in group]
                    xs.extend([t] * len(vals))
                    ys.extend(vals)
                if not xs:
                    continue
                lbl = f"{comm_strats_dict[strat]} ({lab})"
                handle = ax.plot(
                    xs,
                    ys,
                    linestyle="None",
                    marker=markers[lab],
                    markersize=8,
                    color=comm_colors[strat],
                    label=lbl if lbl not in seen else "_nolegend_",
                )[0]
                if lbl not in seen:
                    all_handles.append(handle)
                    all_labels.append(lbl)
                seen.add(lbl)

            # --- plot tcomm on ax2 ---
            data = sorted(results[strat][matrix][0], key=lambda r: (r.tasks if single else r.nodes))
            xs2 = [r.tasks if single else r.nodes for r in data]
            ys2 = [r.tcomm * 1000 for r in data]
            lbl2 = f"{comm_strats_dict.get(strat, strat)} time"
            handle2 = ax2.plot(
                xs2,
                ys2,
                linestyle="-",
                marker="o",
                markersize=6,
                color=tcomm_colors[strat],
                label=lbl2,
            )[0]
            all_handles.append(handle2)
            all_labels.append(lbl2)

        # x-axis formatting
        if single:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
        else:
            ax.set_xticks(range(1, 5))
        ax.grid(True, linestyle="--", alpha=0.5)

    # hide any unused axes
    for ax in axs[len(matrix_names) :]:
        ax.axis("off")

    # ──────────────────────────────────────────────────────────────────────
    # 3) Shared legend
    left, right = fig.subplotpars.left, fig.subplotpars.right
    width = right - left
    fig.subplots_adjust(top=0.85, hspace=0.35, wspace=0.2)
    fig.legend(
        all_handles,
        all_labels,
        loc="upper left",
        bbox_to_anchor=(left, 0.94, width, 0),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=1.0,
        handlelength=1.0,
    )

    # ──────────────────────────────────────────────────────────────────────
    # 4) Save & close
    plt.savefig(
        f"comm_and_tcomm_2x4_{'single' if single else 'multi'}_{partition}mpi.svg",
        bbox_inches="tight",
        format="svg",
    )
    plt.close()


def plot_tcomp_tcomm_2x4(results, single):
    # 1) Pick up to 8 matrices
    matrix_names = []
    first_strat = next(iter(results))
    for matrix in sorted(results[first_strat]):
        if any(matrix in strat_data and 0 in strat_data[matrix] for strat_data in results.values()):
            matrix_names.append(matrix)
    matrix_names = matrix_names[:8]

    # helper to lighten/darken
    def adjust_lightness(color, amount=0.5):
        c = np.array(mcolors.to_rgb(color))
        return tuple(
            np.clip(c + (np.ones(3) - c) * amount if amount > 0 else c * (1 + amount), 0, 1)
        )

    # 2) Set up figure
    fig, axs = plt.subplots(4, 2, figsize=(18, 28))
    axs = axs.flatten()

    # dicts to hold just one line per strat
    comp_lines = {}
    comm_lines = {}

    for idx, matrix in enumerate(matrix_names):
        ax = axs[idx]
        ax.set_title(matrix)
        ax.set_xlabel(argv[3])
        if idx % 2 == 0:
            ax.set_ylabel("Time [s]")

        for i, strat in enumerate(sorted(results)):
            strat_mats = results[strat]
            if matrix not in strat_mats or 0 not in strat_mats[matrix]:
                continue

            # sort data
            data = sorted(strat_mats[matrix][0], key=lambda r: r.tasks if single else r.nodes)
            xs = [r.tasks if single else r.nodes for r in data]
            ys_comp = [r.tcomp for r in data]
            ys_comm = [r.tcomm for r in data]

            base = colors[i % len(colors)]
            c_dark = adjust_lightness(base, amount=-0.3)
            c_light = adjust_lightness(base, amount=0.5)

            # plot comp
            (line_c,) = ax.plot(xs, ys_comp, marker="o", linestyle="-", color=c_dark, markersize=8)
            # plot comm
            (line_m,) = ax.plot(
                xs, ys_comm, marker="s", linestyle="--", color=c_light, markersize=8
            )

            # only register first appearance for legend
            if strat not in comp_lines:
                comp_lines[strat] = line_c
            if strat not in comm_lines:
                comm_lines[strat] = line_m

        # axis formatting
        if single:
            ax.set_xscale("log")
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
            ax.set_xticklabels([1, 2, 4, 8, 16, 32, 64])
        else:
            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(range(1, 5))

        ax.grid(True, linestyle="--", alpha=0.6)
        ax.tick_params(axis="both", which="major")

    # 3) Spacing
    fig.subplots_adjust(top=0.80, hspace=0.35, wspace=0.2)

    # 4) Shared legend: one “comp” + one “comm” per strategy
    handles = []
    labels = []
    for strat in sorted(results):
        name = comm_strats_dict.get(strat, strat)
        if strat in comp_lines:
            handles.append(comp_lines[strat])
            labels.append(f"{name} comp")
        if strat in comm_lines:
            handles.append(comm_lines[strat])
            labels.append(f"{name} comm")

    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95, 0, 0), ncol=2, frameon=False
    )

    # 5) Save
    plt.savefig(
        f"tcomp_tcomm_2x4_{'single' if single else 'multi'}_{partition}mpi.svg",
        bbox_inches="tight",
        format="svg",
    )
    plt.close(fig)


partition = argv[2]


def main():
    base_path = argv[1]
    partition = argv[2]
    # Define nested defaultdicts
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # for comm_strat in comm_strats:
    path = Path(base_path + partition)
    # path = Path(
    #     "/Users/kristiansordal/dev/uib/master/project/SpMV-Comm-Strats/P2/results/multi/defq/test"
    # )

    single = 0
    gather_results(path, results)
    # plot_tcomp_tcomm_2x4(results, single)
    plot_tcomm_2x4(results, single)
    plot_tcomp_2x4(results, single)
    plot_t_2x4(results, single)
    plot_gflops_2x4(results, single)
    plot_comm_min_avg_max_2x4(results, single)

    # plot_comm_load(results)
    # plot_tcomm_single(results)
    # gather_results_from_directory(results, path, 0)
    # plot_comm_min_avg_max_nonmpi(results)
    # plot_comm_min_avg_max_nonmpi_single(results)
    # plot_comm_and_comp_time(results)
    # plot_compare_comm_strat(results)
    # plot_tcomm_single(results)
    # plot_comm_load2(results)


if __name__ == "__main__":
    main()
