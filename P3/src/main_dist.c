#include "mtx.h"
#include "spmv.h"

#include <math.h>
#include <mpi.h> // MPI header file
#include <omp.h>
#include <stdlib.h>

#define FULL 0
#define COMPUTE 1
#define COMMUNICATION 2

void run(graph g, int it, double *input, int rank, int size, comm_lists c, int *p, int opt) {
    double *V = malloc(sizeof(double) * g.N);
    double *Y = malloc(sizeof(double) * g.N);

    int *p_local;
#pragma omp parallel shared(p)
    {
        int nt = omp_get_num_threads();
        int tid = omp_get_thread_num();
        if (tid == 0) {
            p_local = malloc(sizeof(int) * (nt + 1));
            partition_graph_naive(g, p[rank], p[rank + 1], nt, p_local);
        }
    }

    if (rank == 0)
        printf("Time GFLOPs GBs GBs MB CS\n");

    for (int t = 0; t < it; t++) {
        if (rank == 0) {
#pragma omp parallel for
            for (int i = 0; i < g.N; i++)
                V[i] = input[i];
        }
        MPI_Bcast(V, g.N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);

        for (int i = 0; i < 100; i++) {
            if (opt != COMMUNICATION) {

#pragma omp parallel default(shared)
                {
                    int tid = omp_get_thread_num();
                    spmv_part(g, p_local[tid], p_local[tid + 1], V, Y);
                }
            }

            if (opt != COMPUTE)
                exchange_separators(c, Y, rank, size);

            double *tmp = Y;
            Y = V;
            V = tmp;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        // Gather results on rank 0
        if (rank == 0)
            for (int i = 1; i < size; i++)
                MPI_Recv(V + p[i], p[i + 1] - p[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        else
            MPI_Send(V + p[rank], p[rank + 1] - p[rank], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

        // Communication volume
        int commc = 0;
        for (int i = 0; i < size; i++)
            commc += c.send_count[i];

        int commc_total = 0;
        MPI_Reduce(&commc, &commc_total, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            double a = 1.0;
            for (int i = 0; i < g.N; i++)
                a = a < fabs(V[i]) ? fabs(V[i]) : a;

            double L2 = 0.0;
            for (int i = 0; i < g.N; i++)
                L2 += (V[i] / a) * (V[i] / a);

            double ops = (double)g.M * 2.0 * 100.0;
            double GFLOPS = (ops / (t1 - t0)) / 1e9;

            //                            g.E,  g.A                   g.V,   V,    Y
            double data = ((double)g.M * (4.0 + 8.0) + (double)g.N * (4.0 + 8.0 + 8.0)) * 100.0;
            double GBs = (data / (t1 - t0)) / 1e9;

            double comm_data = (double)commc_total * 8.0 * 100.0;
            double comm_GBs = (comm_data / (t1 - t0)) / 1e9;

            printf("%.2lf %.2lf %.2lf %.2lf %.4lf %.10lf\n", t1 - t0, GFLOPS, GBs, comm_GBs, (double)commc_total / 1e6,
                   L2);
        }
    }

    free(p_local);
    free(V);
    free(Y);
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);               // starts MPI, called by every processor
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // get current process id
    MPI_Comm_size(MPI_COMM_WORLD, &size); // get number of processes

    graph g;
    double *input;
    int *p = malloc(sizeof(int) * (size + 1));
    comm_lists c = init_comm_lists(size);

    if (rank == 0) {
        double t0 = omp_get_wtime();
        g = parse_and_validate_mtx(argv[1]);
        double t1 = omp_get_wtime();
        printf("%.2lfs parsing\n", t1 - t0);

        input = malloc(sizeof(double) * g.N);
        for (int i = 0; i < g.N; i++)
            input[i] = ((double)rand() / (double)RAND_MAX) - 0.5;

        t0 = omp_get_wtime();
        partition_graph(g, size, p, input);
        t1 = omp_get_wtime();

        printf("%.2lfs partitioning\n", t1 - t0);
    }

    distribute_graph(&g, rank);
    MPI_Bcast(p, size + 1, MPI_INT, 0, MPI_COMM_WORLD);

    find_sendlists(g, p, rank, size, c);
    find_receivelists(g, p, rank, size, c);

    run(g, 3, input, rank, size, c, p, FULL);
    run(g, 3, input, rank, size, c, p, COMPUTE);
    run(g, 3, input, rank, size, c, p, COMMUNICATION);

    if (rank == 0)
        free(input);
    free_graph(&g);
    free_comm_lists(&c, size);

    MPI_Finalize(); // End MPI, called by every processor

    return 0;
}
