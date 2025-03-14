#include "mtx.h"
#include "p2.h"
#include "spmv.h"
// #include <immintrin.h>
#include <math.h>
#include <mpi.h> // MPI header file
#include <stdio.h>
#include <stdlib.h>

#define alpha 0.7
#define beta 0.1

typedef double v4df __attribute__((vector_size(32)));

// int cmpfunc(const void *a, const void *b) { return (*(double *)a - *(double *)b); }

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    CSR g;
    int *p = malloc(sizeof(int) * (size + 1));
    for (int i = 0; i < size + 1; i++) {
        p[i] = 0;
    }
    comm_lists c = init_comm_lists(size);

    double tcomm, tcomp, t0, t1;

    if (rank == 0) {
        g = parse_and_validate_mtx(argv[1]);
        partition_graph_1b(g, size, p, &c);
    }

    distribute_graph(&g, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(c.send_count, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *V = malloc(sizeof(double) * g.num_rows);
    double *Y = malloc(sizeof(double) * g.num_rows);
    double *x = malloc(sizeof(double) * g.num_rows);
    double *y = malloc(sizeof(double) * g.num_rows);

    // -----Initialization start-----
    MPI_Barrier(MPI_COMM_WORLD);
    double ts0 = MPI_Wtime();

    // ----- Main Program Start -----
    for (int i = 0; i < g.num_rows; i++) {
        x[i] = 2.0;
        y[i] = 2.0;
    }

    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        recvcounts[i] = p[i + 1] - p[i];
        displs[i] = p[i];
    }
    MPI_Barrier(MPI_COMM_WORLD);

    t0 = MPI_Wtime();
    for (int i = 0; i < 100; i++) {
        double tc1 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        MPI_Allgatherv(y + displs[rank], c.send_count[rank], MPI_DOUBLE, y, c.send_count, displs, MPI_DOUBLE,
                       MPI_COMM_WORLD);
        double *tmp = y;
        y = x;
        x = tmp;

        spmv_part(g, rank, p[rank], p[rank + 1], x, y);
        double tc2 = MPI_Wtime();

        MPI_Barrier(MPI_COMM_WORLD);
        double tc3 = MPI_Wtime();
        tcomm += tc3 - tc2;
        tcomp += tc2 - tc1;
    }

    MPI_Allgatherv(y + displs[rank], recvcounts[rank], MPI_DOUBLE, y, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

    double *tmp = x;
    x = y;
    y = tmp;

    t1 = MPI_Wtime();

    double ops = (long long)g.num_cols * 2ll * 100ll;
    double time = t1 - t0;
    double l2 = 0.0;

    if (rank == 0) {
        for (int j = 0; j < g.num_rows; j++)
            l2 += x[j] * x[j];
        l2 = sqrt(l2);
    }

    // Print results
    if (rank == 0) {
        printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n", time, tcomp, tcomm,
               (ops / (time * 1e9)),                                           // GFLOPS
               (g.num_rows * 64.0 * 100.0 / tcomp) / 1e9,                      // GBs mem
               ((g.num_rows * (size - 1)) * 8.0 * size * 100.0 / tcomm) / 1e9, // GBs comm
               l2);
    }

    free(y);
    free(x);
    free_graph(&g);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}
