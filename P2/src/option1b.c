#include "mtx.h"
#include "p2.h"
#include "spmv.h"
// #include <immintrin.h>
#include <math.h>
#include <mpi.h> // MPI header file
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    printf("%d\n", displs[rank]);

    t0 = MPI_Wtime();
    for (int i = 0; i < 100; i++) {
        double tc1 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        if (size == 1) {
            // in-place allgather: leaves y[] untouched
            MPI_Allgatherv(MPI_IN_PLACE,      // sendbuf
                           0,                 // sendcount
                           MPI_DATATYPE_NULL, // sendtype
                           y,                 // recvbuf
                           recvcounts,        // recvcounts[0] = g.num_rows
                           displs,            // displs[0]     = 0
                           MPI_DOUBLE, MPI_COMM_WORLD);
        } else {
            // your normal multi-rank call
            MPI_Allgatherv(y + displs[rank], // sendbuf
                           recvcounts[rank], // sendcount
                           MPI_DOUBLE,
                           y, // recvbuf
                           recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
        }
        double *tmp = y;
        y = x;
        x = tmp;
        double tc2 = MPI_Wtime();
        spmv_part(g, rank, p[rank], p[rank + 1], x, y);
        double tc3 = MPI_Wtime();
        tcomm += tc2 - tc1;
        tcomp += tc3 - tc2;
    }

    t1 = MPI_Wtime();

    // if (size == 1) {
    // memcpy(x, y, sizeof(double) * g.num_rows);
    // } else {
    MPI_Allgatherv(y + displs[rank], recvcounts[rank], MPI_DOUBLE, y, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    // }

    double *tmp = x;
    x = y;
    y = tmp;

    // compute max min and average communication load
    long double comm_size = ((double)c.send_count[rank] * (size - 1) * 100.0 * 64.0) / (1024.0 * 1024.0 * 1024.0);

    printf("c.send_count[%d]: %d %Lf\n", rank, c.send_count[rank], comm_size);

    long double max_comm_size = 0.0;
    long double min_comm_size = 0.0;
    long double avg_comm_size = 0.0;

    MPI_Reduce(&comm_size, &max_comm_size, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_size, &min_comm_size, 1, MPI_LONG_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_size, &avg_comm_size, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_comm_size /= size;

    double ops = (long long)g.num_cols * 2ll * 100ll;
    double time = t1 - t0;
    double l2 = 0.0;

    if (rank == 0) {
        for (int j = 0; j < g.num_rows; j++)
            l2 += x[j] * x[j];
        l2 = sqrt(l2);
    }

    // tcomm = 0;
    // Print results
    if (rank == 0) {

        printf("Total time = %lfs\n", time);
        printf("Communication time = %lfs\n", tcomm);
        printf("Copmutation time = %lfs\n", tcomp);
        printf("GFLOPS = %lf\n", ops / (time * 1e9));
        printf("NFLOPS = %lf\n", ops);
        printf("Comm min = %Lf GB\nComm max = %Lf GB\nComm avg = %Lf GB\n", min_comm_size, max_comm_size,
               avg_comm_size);
        // printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n", time, tcomp, tcomm,
        //        (ops / (time * 1e9)),                                           // GFLOPS
        //        (g.num_rows * 64.0 * 100.0 / tcomp) / 1e9,                      // GBs mem
        //        ((g.num_rows * (size - 1)) * 8.0 * size * 100.0 / tcomm) / 1e9, // GBs comm
        //        l2);
        // printf("Comm min = %Lf GB\nComm max = %Lf GB\nComm avg = %Lf GB\n", min_comm_size, max_comm_size,
        //        avg_comm_size);
    }

    free(y);
    free(x);
    free_graph(&g);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}
