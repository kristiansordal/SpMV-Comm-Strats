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

    // if (rank == 0) {
    //     printf("Main program start\n");
    //     printf("NNZ: %d\n", g.num_cols);
    // }

    printf("Rank %d num rows: %d\n", rank, g.num_rows);
    printf("Rank %d: %d -> %d with %d\n", rank, p[rank], p[rank + 1], p[rank + 1] - p[rank]);
    fflush(stdout);

    for (int i = 0; i < g.num_rows; i++) {
        Vo[i] = input[i];
    }

    if (rank == 1) {
        printf("Gonna print Vo\n");
        fflush(stdout);
        for (int i = p[rank]; i < p[rank + 1]; i++) {
            printf("%d ", i);
        }
        printf("\n");
        fflush(stdout);
    }

    printf("Rank %d at barrier", rank);
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    for (size_t it = 0; it < 1; it++) {
        // Vo[0] = 0xffffff;

        double tcomm = 0.0, tcomp = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        for (int i = 0; i < 100; i++) {
            double tc1 = MPI_Wtime();
            // spmv_part(g, p[rank], p[rank + 1], Vo, Vn);
            int sendcount = p[rank + 1] - p[rank];
            MPI_Allgatherv(Vn, sendcount, MPI_DOUBLE, Vo, p, p, MPI_DOUBLE, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);
            double tc2 = MPI_Wtime();

            // MPI_Barrier(MPI_COMM_WORLD);
            double tc3 = MPI_Wtime();

            tcomm += tc3 - tc2;
            tcomp += tc2 - tc1;
        }

        // MPI_Barrier(MPI_COMM_WORLD);
        // double t1 = MPI_Wtime();

        // // Compute L2 and GLOPS

        // double l2 = 0.0;
        // for (int j = 0; j < g.num_rows; j++)
        //     l2 += Vo[j] * Vo[j];

        // l2 = sqrt(l2);

        // double ops = (long long)g.num_cols * 8ll * 100ll; // 4 multiplications and 4 additions
        // double time = t1 - t0;

        // if (rank == 0) {
        //     printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n", time, tcomp, tcomm,
        //            (ops / time) / 1e9, (g.num_rows * 64.0 * 100.0 / tcomp) / 1e9,
        //            ((g.num_rows * (size - 1)) * 8.0 * size * 100.0 / tcomm) / 1e9, l2);
        // }
    }

    free(Vn);
    free(Vo);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}
