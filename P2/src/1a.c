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

    distribute_graph(&g, rank);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    // -----Initialization start-----
    MPI_Barrier(MPI_COMM_WORLD);
    double ts0 = MPI_Wtime();

    // Allocate local part of A and I
    double *A = aligned_alloc(32, sizeof(double) * g.num_rows * nonzero);
    int *I = aligned_alloc(16, sizeof(int) * g.num_rows * nonzero);

    int *sep = malloc(sizeof(int) * size);
    int *new_id = malloc(sizeof(int) * g.num_rows);

    double *Vo = aligned_alloc(32, sizeof(double) * g.num_rows);
    double *Vn = aligned_alloc(32, sizeof(double) * g.num_rows);

    MPI_Barrier(MPI_COMM_WORLD);
    double ts1 = MPI_Wtime();

    if (rank == 0)
        printf("%lfs for initialization\n", ts1 - ts0);

    // -----Initialization end-----

    // -----Main program start-----
    for (size_t it = 0; it < n_it; it++) {
        spmv_part(g, p[rank], p[rank + 1], Vo, Vn);
        MPI_Barrier(MPI_COMM_WORLD);
        int sendcount = p[rank + 1] - p[rank];
        MPI_Allgatherv(Vn + p[rank], sendcount, MPI_DOUBLE, Vo, p, p + 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        // #pragma omp parallel
        //         {
        //             for (int i = 0; i < 100; i++) {
        //                 double tc1 = MPI_Wtime();

        // #pragma omp for
        //                 for (size_t j = 0; j < rows; j += 4) {
        //                     v4df z = {0.0, 0.0, 0.0, 0.0};

        //                     // for (size_t k = 0; k < 4; k++)
        //                     //     z[k] = A[(j + k) * nonzero + 0] * Vo[I[(j + k) * nonzero + 0]] +
        //                     //            A[(j + k) * nonzero + 1] * Vo[I[(j + k) * nonzero + 1]] +
        //                     //            A[(j + k) * nonzero + 2] * Vo[I[(j + k) * nonzero + 2]] +
        //                     //            A[(j + k) * nonzero + 3] * Vo[I[(j + k) * nonzero + 3]];

        //                     // _mm256_stream_pd(Vn + rank * rows + j, z);
        //                 }

        // #pragma omp master
        //                 {
        //                     MPI_Barrier(MPI_COMM_WORLD);
        //                     double tc2 = MPI_Wtime();

        //                     MPI_Request sends[size], receives[size];

        //                     for (int j = 0; j < size; j++) {
        //                         if (j != rank && to_first[j] < to_last[j])
        //                             MPI_Isend(Vn + to_first[j], to_last[j] - to_first[j], MPI_DOUBLE, j, 0,
        //                             MPI_COMM_WORLD,
        //                                       &sends[j]);

        //                         if (j != rank && from_first[j] < from_last[j])
        //                             MPI_Irecv(Vn + from_first[j], from_last[j] - from_first[j], MPI_DOUBLE, j, 0,
        //                                       MPI_COMM_WORLD, &receives[j]);
        //                     }

        //                     for (int j = 0; j < size; j++) {
        //                         if (j != rank && to_first[j] < to_last[j])
        //                             MPI_Wait(&sends[j], MPI_STATUS_IGNORE);

        //                         if (j != rank && from_first[j] < from_last[j])
        //                             MPI_Wait(&receives[j], MPI_STATUS_IGNORE);
        //                     }

        //                     double *t = Vn;
        //                     Vn = Vo;
        //                     Vo = t;

        //                     MPI_Barrier(MPI_COMM_WORLD);
        //                     double tc3 = MPI_Wtime();

        //                     tcomm += tc3 - tc2;
        //                     tcomp += tc2 - tc1;
        //                 }
        // #pragma omp barrier
        //             }
        //         }

        //         MPI_Barrier(MPI_COMM_WORLD);
        //         double t1 = MPI_Wtime();

        //         MPI_Allgather(Vo + rank * rows, rows, MPI_DOUBLE, Vn, rows, MPI_DOUBLE, MPI_COMM_WORLD);

        // Compute L2 and GLOPS

        double l2 = 0.0;
        for (int j = 0; j < g.num_rows; j++)
            l2 += Vn[j] * Vn[j];

        l2 = sqrt(l2);

        double ops = (long long)g.num_rows * 8ll * 100ll; // 4 multiplications and 4 additions
        double time = t1 - t0;

        //         // Comm size
        //         int send_count = 0;
        //         for (int j = 0; j < size; j++) {
        //             if (j != rank && to_first[j] < to_last[j])
        //                 send_count += to_last[j] - to_first[j];
        //         }

        //         if (rank == 0) {
        //             int *send_counts = malloc(sizeof(int) * size);
        //             MPI_Gather(&send_count, 1, MPI_INT, send_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
        //             int total_comm = 0;
        //             for (int j = 0; j < size; j++)
        //                 total_comm += send_counts[j];

        printf("%lfs (%lfs, %lfs), %lf GFLOPS, %lf GBs mem, %lf GBs comm, L2 = %lf\n", time, tcomp, tcomm,
               (ops / time) / 1e9, (N * 64.0 * 100.0 / tcomp) / 1e9, (total_comm * 8.0 * 100.0 / tcomm) / 1e9, l2);

        //             free(send_counts);
        //         } else {
        //             MPI_Gather(&send_count, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);
        //         }
    }

    // if (rank == 0)
    //     free_mesh(&m);

    free(A);
    free(I);
    free(Vn);
    free(Vo);

    MPI_Finalize(); // End MPI, called by every processor
    return 0;
}
