#include "mtx.h"
#include "spmv.h"
#include <math.h>
#include <mpi.h> // MPI header file
#include <stdio.h>
#include <stdlib.h>

#define n_it 100

typedef double v4df __attribute__((vector_size(32)));

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    CSR g;
    double *input;
    int *p = malloc(sizeof(int) * size + 1);

    if (rank == 0) {
        g = parse_and_validate_mtx(argv[1]);
        input = malloc(sizeof(double) * g.num_rows);
        for (int i = 0; i < g.num_rows; i++)
            input[i] = ((double)rand() / (double)RAND_MAX) - 0.5;
        partition_graph(g, size, p, input);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    distribute_graph(&g, rank);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *Vo = malloc(sizeof(double) * g.num_rows);
    double *Vn = malloc(sizeof(double) * g.num_rows);

    // -----Initialization end-----

    // -----Main program start-----

    for (int i = 0; i < g.num_rows; i++)
        Vo[i] = 1.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double tcomm = 0.0, tcomp = 0.0;

    // MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        recvcounts[i] = p[i + 1] - p[i];                              // Each process sends its own chunk
        displs[i] = (i == 0) ? 0 : displs[i - 1] + recvcounts[i - 1]; // Compute displacements
    }
    for (int i = 0; i < 100; i++) {
        double tc1 = MPI_Wtime();
        spmv_part(g, p[rank], p[rank + 1], Vo, Vn);
        int sendcount = p[rank + 1] - p[rank];
        MPI_Allgatherv(Vo, sendcount, MPI_DOUBLE, Vn, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double tc2 = MPI_Wtime();
        double *tmp = Vo;
        Vo = Vn;
        Vn = tmp;

        double tc3 = MPI_Wtime();

        tcomm += tc3 - tc2;
        tcomp += tc2 - tc1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    // Compute L2 and GLOPS

    double l2 = 0.0;
    for (int j = 0; j < g.num_rows; j++)
        l2 += Vo[j] * Vo[j];

    l2 = sqrt(l2);

    double ops = (long long)g.num_cols * 2ll * 100ll;
    double time = t1 - t0;

    if (rank == 0) {
        printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n", time, tcomp, tcomm,
               (ops / time) / 1e9, (g.num_rows * 64.0 * 100.0 / tcomp) / 1e9,
               ((g.num_rows * (size - 1)) * 8.0 * size * 100.0 / tcomm) / 1e9, l2);
    }
    // }

    free(Vn);
    free(Vo);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}
