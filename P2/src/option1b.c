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
    double *input;

    printf("init commlists\n");
    fflush(stdout);
    comm_lists c = init_comm_lists(size);
    printf("done init commlists\n");
    fflush(stdout);

    double tcomm, tcomp, t0, t1;

    if (rank == 0) {
        printf("Reading matrix\n");
        fflush(stdout);
        g = parse_and_validate_mtx(argv[1]);
        printf("done reading matrix\n");
        fflush(stdout);

        input = malloc(sizeof(double) * g.num_rows);
        for (int i = 0; i < g.num_rows; i++)
            input[i] = ((double)rand() / (double)RAND_MAX) - 0.5;

        partition_graph_and_reorder_separators(g, size, p, input, &c);
    }

    if (rank == 0) {
        printf("distributing\n");
        fflush(stdout);
    }
    distribute_graph(&g, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(c.send_count, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("done distributing\n");
        fflush(stdout);
    }

    double *V = malloc(sizeof(double) * g.num_rows);
    double *Y = malloc(sizeof(double) * g.num_rows);
    double *x = malloc(sizeof(double) * g.num_rows);
    double *y = malloc(sizeof(double) * g.num_rows);

    // -----Initialization start-----
    MPI_Barrier(MPI_COMM_WORLD);
    double ts0 = MPI_Wtime();

    // ----- Main Program Start -----
    for (int i = 0; i < g.num_rows; i++) {
        x[i] = 1.0;
        y[i] = 1.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("starting spmv\n");
        fflush(stdout);
    }
    t0 = MPI_Wtime();
    for (int iter = 0; iter < 100; iter++) {
        double tc1 = MPI_Wtime();
        spmv_part(g, rank, p[rank], p[rank + 1], x, y);
        MPI_Barrier(MPI_COMM_WORLD);
        double tc2 = MPI_Wtime();
        MPI_Allgatherv(y + p[rank], c.send_count[rank], MPI_DOUBLE, x + p[rank], c.send_count, p, MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double tc3 = MPI_Wtime();
        tcomm += tc3 - tc2;
        tcomp += tc2 - tc1;

        double *tmp = x;
        x = y;
        y = tmp;
    }
    t1 = MPI_Wtime();

    if (rank == 0) {
        printf("starting spmv\n");
        fflush(stdout);
    }

    // Compute L2 and GLOPS

    double ops = (long long)g.num_cols * 2ll * 100ll; // 4 multiplications and 4 additions
    double time = t1 - t0;

    t1 = MPI_Wtime();
    double l2 = 0.0;
    for (int j = 0; j < g.num_rows; j++)
        l2 += x[j] * x[j];

    l2 = sqrt(l2);

    // Print results
    if (rank == 0) {
        printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n", time, tcomp, tcomm,
               (ops / time) / 1e9,                                             // GFLOPS
               (g.num_rows * 64.0 * 100.0 / tcomp) / 1e9,                      // GBs mem
               ((g.num_rows * (size - 1)) * 8.0 * size * 100.0 / tcomm) / 1e9, // GBs comm
               l2);
    }

    free(y);
    free(x);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}
