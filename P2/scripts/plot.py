from matplotlib import pyplot as plt
from pathlib import Path


def read_stdout_files(path: Path):
    path = path.expanduser()
    files = sorted([f for f in path.glob("*") if "stdout" in f.name])
    files2 = sorted([f for f in path.glob("*") if "stderr" in f.name])

    # a, b = 0, 0
    # for file in files:
    #     with file.open() as f:
    #         content = f.read()
    #         b += 1
    #         if "GFLOPS" in content.strip():
    #             a += 1
    # print(a, b)
    # a, b = 0, 0
    # for file in files2:
    #     with file.open() as f:
    #         content = f.read()
    #         if


def main():
    path_rome1a = Path("~/dev/uib/master/project/SpMV-Comm-Strats/P2/results/new/1a/rome")
    path_rome1b = Path("~/dev/uib/master/project/SpMV-Comm-Strats/P2/results/new/1b/rome")

    read_stdout_files(path_rome1a)
    read_stdout_files(path_rome1b)


if __name__ == "__main__":
    main()
