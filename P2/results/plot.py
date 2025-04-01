import matplotlib.pyplot as plt
from pathlib import Path
from sys import argv
import re


class Result:
    def __init__(self, name, nodes, tasks, threads, mpi):
        self.name = name
        self.nodes = nodes
        self.tasks = tasks
        self.threads = threads
        self.mpi = mpi
        self.t: float = 0
        self.tcomm: float = 0
        self.tcomp: float = 0
        self.gflops: float = 0
        self.comm_min: float = 0
        self.comm_max: float = 0
        self.comm_avg: float = 0

    def __str__(self):
        return f"Name: {self.name}\n Nodes: {self.nodes}\n Tasks: {self.tasks}\n Threads: {self.threads}\n MPI: {self.mpi}"


def print_res(result: list[Result]):
    for r in result:
        print(r.mpi)


def parse_file_name(file_name: str) -> Result:
    tokens = file_name.split("_")
    name, nodes, tasks, threads, mpi = "", 0, 0, 0, 0
    if "mpi" in file_name:
        mpi = 1
    for i, token in enumerate(tokens):
        if token == "nodes":
            name = "_".join(tokens[: i - 1])
            nodes = int(tokens[i - 1])
        elif token == "tasks":
            tasks = int(tokens[i - 1])
        elif token == "threads":
            threads = int(tokens[i - 1])
    return Result(name, nodes, tasks, threads, mpi)


def parse_file_contents(file_name: Path, result: Result) -> Result:
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split()
            if "Total time" in line:
                result.t = float(tokens[3][:-1])
            elif "Communication time" in line:
                result.tcomm = float(tokens[3][:-1])
            elif "Computation time" in line:
                result.tcomp = float(tokens[3][:-1])
            elif "GFLOPS" in line:
                result.gflops = float(tokens[2])
            elif "Comm min" in line:
                result.comm_min = float(tokens[3])
            elif "Comm max" in line:
                result.comm_max = float(tokens[3])
            elif "Comm avg" in line:
                result.comm_avg = float(tokens[3])
    return result


def parse_file(file: Path) -> Result:
    return parse_file_contents(file, parse_file_name(file.stem))


def get_files(path: Path):
    stdout_files = list(path.glob("*stdout*"))
    results = [parse_file(file) for file in stdout_files]
    single_proc_res = [r for r in results if r.mpi == 0]
    dual_proc_res = [r for r in results if r.mpi == 1]
    return single_proc_res, dual_proc_res


# compares the performance of communication strategy 1a-d
def plot_compare_comm_strats():
    pass


# compares the performance of using 1 MPI process per socket, vs using 1 MPI process per node
def plot_compare_num_sockets(single_proc_res, dual_proc_res):
    single_proc_res.sort(key=lambda r: r.nodes)
    dual_proc_res.sort(key=lambda r: r.nodes)

    nodes_single = [r.nodes for r in single_proc_res]
    time_single = [r.t for r in single_proc_res]
    gflops_single = [r.gflops for r in single_proc_res]

    nodes_dual = [r.nodes for r in dual_proc_res]
    time_dual = [r.t for r in dual_proc_res]
    gflops_dual = [r.gflops for r in dual_proc_res]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(nodes_single, time_single, label="Non-MPI", marker="o")
    axes[0].plot(nodes_dual, time_dual, label="MPI", marker="s")
    axes[0].set_xlabel("Number of Nodes")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Execution Time Comparison")
    axes[0].legend()

    axes[1].plot(nodes_single, gflops_single, label="Non-MPI", marker="o")
    axes[1].plot(nodes_dual, gflops_dual, label="MPI", marker="s")
    axes[1].set_xlabel("Number of Nodes")
    axes[1].set_ylabel("GFLOPS")
    axes[1].set_title("Performance Comparison")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# compares the performance of the different matrices
def plot_compare_matrices():
    pass


def main():
    path: Path = Path(argv[1])
    single_proc_res, dual_proc_res = get_files(path)
    plot_compare_num_sockets(single_proc_res, dual_proc_res)
    print_res(single_proc_res)
    print_res(dual_proc_res)


if __name__ == "__main__":
    main()
