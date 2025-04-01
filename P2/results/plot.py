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
        self.tcomm: float = 0
        self.tcomp: float = 0
        self.gflops: float = 0
        self.comm_min: float = 0
        self.comm_max: float = 0
        self.comm_avg: float = 0

    def __str__(self):
        return f"Name: {self.name}\n Nodes: {self.nodes}\n Tasks: {self.tasks}\n Threads: {self.threads}\n MPI: {self.mpi}"


def parse_file_name(file_name: str) -> Result:
    tokens = file_name.split("_")
    name, nodes, tasks, threads, mpi = "", 0, 0, 0, 0
    for i, token in enumerate(tokens):
        if token == "nodes":
            name = "_".join(tokens[: i - 1])
            nodes = int(tokens[i - 1])
        elif token == "tasks":
            tasks = int(tokens[i - 1])
        elif token == "threads":
            threads = int(tokens[i - 1])
        elif token == "mpi":
            mpi = 1
    return Result(name, nodes, tasks, threads, mpi)


def parse_file_contents(file_name: str, result: Result) -> Result:
    with open(file_name, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "Total Time" in line:
                tokens = line.split()
                result.tcomm = float(tokens[3])
                result.tcomp = float(tokens[5])
            elif "GFLOPS" in line:
                tokens = line.split()
                result.gflops = float(tokens[1])
    return result


def parse_file(file: Path) -> Result:
    res = parse_file_name(file.stem)
    res = parse_file_contents(file, res)


# compares the performance of communication strategy 1a-d
def plot_compare_comm_strats():
    pass


# compares the performance of using 1 MPI process per socket, vs using 1 MPI process per node
def plot_compare_num_sockets():
    pass


# compares the performance of the different matrices
def plot_compare_matrices():
    pass


def main():
    path: Path = Path(argv[1])


if __name__ == "__main__":
    main()
