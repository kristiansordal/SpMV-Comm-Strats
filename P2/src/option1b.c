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

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    CSR g;
    int *p = malloc(sizeof(int) * (size + 1));
    double *input;
    comm_lists c = init_comm_lists(size);

    double tcomm, tcomp, t0, t1;

    if (rank == 0) {
        g = parse_and_validate_mtx(argv[1]);

        input = malloc(sizeof(double) * g.num_rows);
        for (int i = 0; i < g.num_rows; i++)
            input[i] = ((double)rand() / (double)RAND_MAX) - 0.5;

        partition_graph(g, size, p, input, &c);
    }

    distribute_graph(&g, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(c.send_count, size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    double *V = malloc(sizeof(double) * g.num_rows);
    double *Y = malloc(sizeof(double) * g.num_rows);
    double *Vo = malloc(sizeof(double) * g.num_rows);
    double *Vn = malloc(sizeof(double) * g.num_rows);

    // -----Initialization start-----
    MPI_Barrier(MPI_COMM_WORLD);
    double ts0 = MPI_Wtime();

    // ----- Main Program Start -----
    for (int i = 0; i < g.num_rows; i++) {
        Vo[i] = 1.0;
        Vn[i] = 1.0;
    }

    if (rank == 0) {
        for (int i = 0; i < size; i++) {
            printf("%d, ", p[i]);
        }
        fflush(stdout);
    }

    t0 = MPI_Wtime();
    for (int iter = 0; iter < 10; iter++) {
        double tc1 = MPI_Wtime();
        spmv_part(g, p[rank], p[rank + 1], Vn, Vo);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allgatherv(Vn + p[rank], c.send_count[rank], MPI_DOUBLE, Vo, c.send_count, p, MPI_DOUBLE, MPI_COMM_WORLD);
        double tc2 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        double tc3 = MPI_Wtime();
        tcomm += tc3 - tc2;
        tcomp += tc2 - tc1;

        double *tmp = Vo;
        Vo = Vn;
        Vn = tmp;
    }
    t1 = MPI_Wtime();

    // Compute L2 and GLOPS

    double ops = (long long)g.num_rows * 2ll * 100ll; // 4 multiplications and 4 additions
    double time = t1 - t0;

    t1 = MPI_Wtime();
    double l2 = 0.0;
    for (int j = 0; j < g.num_rows; j++)
        l2 += Vo[j] * Vo[j];

    l2 = sqrt(l2);

    // Print results
    if (rank == 0) {
        printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n", time, tcomp, tcomm,
               (ops / time) / 1e9,                                             // GFLOPS
               (g.num_rows * 64.0 * 100.0 / tcomp) / 1e9,                      // GBs mem
               ((g.num_rows * (size - 1)) * 8.0 * size * 100.0 / tcomm) / 1e9, // GBs comm
               l2);
    }

    free(Vn);
    free(Vo);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}
