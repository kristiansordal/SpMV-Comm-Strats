#include "mtx.h"
#include "spmv.h"
#include <math.h>
#include <mpi.h> // MPI header file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define n_it 100

typedef double v4df __attribute__((vector_size(32)));

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    CSR g;
    double tcomm, tcomp, t0, t1; // tc1, tc2, tc3;
    int *p = malloc(sizeof(int) * size + 1);

    if (rank == 0) {
        g = parse_and_validate_mtx(argv[1]);
        printf("partitioning graph\n");
        partition_graph(g, size, p);
    }

    if (rank == 0) {
        printf("distributing graph\n");
        fflush(stdout);
    }

    printf("Rank %d at barrier 1\n", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d past barrier 1\n", rank);
    fflush(stdout);
    distribute_graph(&g, rank);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *x = (double *)malloc(sizeof(double) * g.num_rows);
    double *y = (double *)malloc(sizeof(double) * g.num_rows);

    // -----Initialization end-----

    // -----Main program start-----
    for (int i = 0; i < g.num_rows; i++) {
        x[i] = 2.0;
        y[i] = 2.0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    tcomm = 0.0, tcomp = 0.0;

    int *recvcounts = malloc(size * sizeof(int));
    int sendcount = p[rank + 1] - p[rank];
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size + 1; i++)
        recvcounts[i] = p[i + 1] - p[i];

    for (int i = 0; i < size; i++)
        displs[i] = p[i];

    if (rank == 0) {
        printf("starting spmv\n");
        fflush(stdout);
    }
    t0 = MPI_Wtime();
    long double flops = 0;
    for (int i = 0; i < 100; i++) {
        double tc1 = MPI_Wtime();
        spmv_part_flops(g, rank, p[rank], p[rank + 1], x, y, &flops);
        MPI_Barrier(MPI_COMM_WORLD);
        double tc2 = MPI_Wtime();
        MPI_Allgatherv(y + displs[rank], sendcount, MPI_DOUBLE, y, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        double tc3 = MPI_Wtime();
        double *tmp = x;
        x = y;
        y = tmp;
        tcomm += tc3 - tc2;
        tcomp += tc2 - tc1;
    }
    t1 = MPI_Wtime();
    if (rank == 0) {
        printf("done spmv\n");
        fflush(stdout);
    }

    double l2 = 0.0;
    if (rank == 0) {
        for (int j = 0; j < g.num_rows; j++)
            l2 += x[j] * x[j];
        l2 = sqrt(l2);
    }

    long double max_comm_size = ((double)g.num_rows * 100.0 * 64.0) / (1024.0 * 1024.0 * 1024.0);
    long double min_comm_size = ((double)g.num_rows * 100.0 * 64.0) / (1024.0 * 1024.0 * 1024.0);
    long double avg_comm_size = ((double)g.num_rows * 100.0 * 64.0) / (1024.0 * 1024.0 * 1024.0);

    // Compute FLOPs and memory bandwidth
    double ops = (long long)g.num_cols * 2ll * 100ll; // 2 FLOPs per nonzero entry, 100 iterations

    double time = t1 - t0;
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Total time = %lfs\n", time);
        printf("Communication time = %lfs\n", tcomm);
        printf("Copmutation time = %lfs\n", tcomp);
        printf("NFLOPS = %f\n", ops);
        printf("GFLOPS = %lf\n", ops / (time * 1e9));
        printf("Comm min = %Lf GB\nComm max = %Lf GB\nComm avg = %Lf GB\n", min_comm_size, max_comm_size,
               avg_comm_size);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    free(y);
    free(x);
    free(p);
    free(recvcounts);
    free(displs);

    MPI_Finalize();
    return 0;
}
