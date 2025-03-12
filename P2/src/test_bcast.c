#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 10000000;
    int *data = malloc(sizeof(int) * n);

    if (rank == 0) {
        for (int i = 0; i < n; i++) {
            data[i] = i;
        }
    }

    MPI_Bcast(data, n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 1) {
        for (int i = 0; i < 10; i++) {
            printf("%d\n", data[i]);
        }
    }

    MPI_Finalize();

    return 0;
}
