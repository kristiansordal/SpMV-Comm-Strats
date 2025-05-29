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
        partition_graph(g, size, p);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    distribute_graph(&g, rank);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    find_sendlists(g, p, rank, size, c);
    find_receivelists(g, p, rank, size, c);

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

    MPI_Barrier(MPI_COMM_WORLD);

    int *recvcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        recvcounts[i] = p[i + 1] - p[i];
        displs[i] = p[i];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    long double flops = 0.0;

    t0 = MPI_Wtime();
    for (int i = 0; i < 100; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double tc1 = MPI_Wtime();
        exchange_required_separators(c, y, rank, size);
        double tc2 = MPI_Wtime();
        double *tmp = y;
        y = x;
        x = tmp;
        spmv_part(g, rank, p[rank], p[rank + 1], x, y);
        double tc3 = MPI_Wtime();
        tcomm += tc2 - tc1;
        tcomp += tc3 - tc2;
    }
    t1 = MPI_Wtime();

    MPI_Allgatherv(y + displs[rank], recvcounts[rank], MPI_DOUBLE, y, recvcounts, displs, MPI_DOUBLE, MPI_COMM_WORLD);
    double *tmp = x;
    x = y;
    y = tmp;

    long double comm_size = 0.0;

    // if (rank == 0) {
    //     for (int i = 0; i < size; i++) {
    //         for (int j = 0; j < size; j++) {
    //             printf("%d,%d,%d\n", i, j, c.send_items[i][j]);
    //         }
    //     }
    // }

    for (int i = 0; i < size; i++) {
        comm_size += c.send_count[i];
    }
    // printf("%d,%Lf\n", rank, comm_size);

    long double total_comm_size = 0.0;
    MPI_Allreduce(&comm_size, &total_comm_size, 1, MPI_LONG_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    // printf("Total communication size: %Lf\n", total_comm_size);

    comm_size = (comm_size * 64.0 * 100.0) / (1024.0 * 1024.0 * 1024.0);

    long double max_comm_size = 0.0;
    long double min_comm_size = 0.0;
    long double avg_comm_size = 0.0;
    long double total_flops = 0.0;

    MPI_Reduce(&comm_size, &max_comm_size, 1, MPI_LONG_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_size, &min_comm_size, 1, MPI_LONG_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&comm_size, &avg_comm_size, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_flops, &flops, 1, MPI_LONG_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    avg_comm_size /= size;

    double ops = (long long)g.num_cols * 2ll * 100ll;
    double time = t1 - t0;
    double l2 = 0.0;

    if (rank == 0) {
        for (int j = 0; j < g.num_rows; j++)
            l2 += x[j] * x[j];
        l2 = sqrt(l2);
    }

    // unknowns: 2164760
    // NFLOPS = 25441228800.000000
    // | V |= 2164760 | E |= 127206144

    // Print results
    if (rank == 0) {

        printf("Total time = %lfs\n", time);
        printf("Communication time = %lfs\n", tcomm);
        printf("Copmutation time = %lfs\n", tcomp);
        printf("GFLOPS = %lf\n", ops / (time * 1e9));
        printf("compGFLOPS = %Lf\n", total_flops / (time * 1e9));
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
