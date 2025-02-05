#include "spmv.h"
#include <metis.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>

void spmv(CSR g, double *x, double *y) {
#pragma omp parallel for
    for (int u = 0; u < g.num_rows; u++) {
        double z = 0.0;
        for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
            int v = g.col_idx[i];
            z += x[v] * g.values[i];
        }
        y[u] = z;
    }
}

void spmv_part(CSR g, int row_ptr_start_idx, int row_ptr_end_idx, double *x, double *y) {
#pragma omp parallel for schedule(static)
    for (int u = row_ptr_start_idx; u < row_ptr_end_idx; u++) {
        double z = 0.0;
        for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
            int v = g.col_idx[i];
            z += x[v] * g.values[i];
        }
        y[u] = z;
    }
}

void partition_graph(CSR g, int num_partitions, int *partition_idx, double *x) {
    if (num_partitions == 1) {
        partition_idx[0] = 0;
        partition_idx[1] = g.num_rows;
        return;
    }

    int ncon = 1;
    int objval;
    real_t ubvec = 1.01;
    int *part = malloc(sizeof(int) * g.num_rows);
    int rc = METIS_PartGraphKway(&g.num_rows, &ncon, g.row_ptr, g.col_idx, NULL, NULL, NULL, &num_partitions, NULL,
                                 &ubvec, NULL, &objval, part);

    int *new_id = malloc(sizeof(int) * g.num_rows);
    int *old_id = malloc(sizeof(int) * g.num_rows);
    int id = 0;
    partition_idx[0] = 0;
    for (int r = 0; r < num_partitions; r++) {
        for (int i = 0; i < g.num_rows; i++) {
            if (part[i] == r) {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }
        partition_idx[r + 1] = id;
        printf("P: %d, %d\n", r, id);
    }
    partition_idx[num_partitions] = g.num_rows;

    int *new_V = malloc(sizeof(int) * (g.num_rows + 1));
    int *new_E = malloc(sizeof(int) * g.num_cols);
    double *new_A = malloc(sizeof(double) * g.num_cols);

    new_V[0] = 0;
    for (int i = 0; i < g.num_rows; i++) {
        int d = g.row_ptr[old_id[i] + 1] - g.row_ptr[old_id[i]];
        new_V[i + 1] = new_V[i] + d;
        memcpy(new_E + new_V[i], g.col_idx + g.row_ptr[old_id[i]], sizeof(int) * d);
        memcpy(new_A + new_V[i], g.values + g.row_ptr[old_id[i]], sizeof(double) * d);

        for (int j = new_V[i]; j < new_V[i + 1]; j++) {
            new_E[j] = new_id[new_E[j]];
        }
    }

    double *new_X = malloc(sizeof(double) * g.num_rows);
    for (int i = 0; i < g.num_rows; i++) {
        new_X[i] = x[old_id[i]];
    }

    memcpy(x, new_X, sizeof(double) * g.num_rows);
    memcpy(g.row_ptr, new_V, sizeof(int) * (g.num_rows + 1));
    memcpy(g.col_idx, new_E, sizeof(int) * g.num_cols);
    memcpy(g.values, new_A, sizeof(double) * g.num_cols);

    free(new_V);
    free(new_E);
    free(new_A);
    free(new_X);

    free(new_id);
    free(old_id);
    free(part);
}

void partition_graph_naive(CSR g, int s, int t, int k, int *p) {
    int edges_per = (g.row_ptr[t] - g.row_ptr[s]) / k;
    p[0] = s;
    int id = 1;
    for (int u = s; u < t; u++) {
        if ((g.row_ptr[u] - g.row_ptr[s]) >= edges_per * id)
            p[id++] = u;
    }
    while (id <= k)
        p[id++] = t;
    p[k] = t;
}

void distribute_graph(CSR *g, int rank) {
    MPI_Bcast(&g->num_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&g->num_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank != 0) {
        g->row_ptr = malloc(sizeof(int) * (g->num_rows + 1));
        g->col_idx = malloc(sizeof(int) * g->num_cols);
        g->values = malloc(sizeof(double) * g->num_cols);
    }

    MPI_Bcast(g->row_ptr, g->num_rows + 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->col_idx, g->num_cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(g->values, g->num_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

comm_lists init_comm_lists(int size) {
    comm_lists c = {.send_count = malloc(sizeof(int) * size),
                    .receive_count = malloc(sizeof(int) * size),
                    .send_items = malloc(sizeof(int *) * size),
                    .receive_items = malloc(sizeof(int *) * size),
                    .send_lists = malloc(sizeof(double *) * size),
                    .receive_lists = malloc(sizeof(double *) * size)};
    return c;
}

void free_comm_lists(comm_lists *c, int size) {
    for (int i = 0; i < size; i++) {
        free(c->send_items[i]);
        free(c->send_lists[i]);
        free(c->receive_items[i]);
        free(c->receive_lists[i]);
    }

    free(c->send_count);
    free(c->receive_count);

    free(c->send_items);
    free(c->receive_items);

    free(c->send_lists);
    free(c->receive_lists);
}

void find_sendlists(CSR g, int *p, int rank, int size, comm_lists c) {
    int *send_mark = malloc(sizeof(int) * g.num_rows);
    for (int r = 0; r < size; r++) {
        c.send_count[r] = 0;
        c.send_items[r] = NULL;
        c.send_lists[r] = NULL;
        if (r == rank)
            continue;

        // Set marks to zero
        for (int i = p[rank]; i < p[rank + 1]; i++)
            send_mark[i] = 0;

        // Find separators
        for (int u = p[r]; u < p[r + 1]; u++) {
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                int v = g.col_idx[i];
                if (v >= p[rank] && v < p[rank + 1])
                    send_mark[v] = 1;
            }
        }

        // Count separators
        for (int i = p[rank]; i < p[rank + 1]; i++)
            c.send_count[r] += send_mark[i];

        if (c.send_count[r] == 0)
            continue;

        // Store list of separators
        c.send_items[r] = malloc(sizeof(int) * c.send_count[r]);
        c.send_lists[r] = malloc(sizeof(double) * c.send_count[r]);

        int j = 0;
        for (int i = p[rank]; i < p[rank + 1]; i++)
            if (send_mark[i])
                c.send_items[r][j++] = i;
    }

    free(send_mark);
}

void find_receivelists(CSR g, int *p, int rank, int size, comm_lists c) {
    int *receive_mark = malloc(sizeof(int) * g.num_rows);
    for (int r = 0; r < size; r++) {
        c.receive_count[r] = 0;
        c.receive_items[r] = NULL;
        c.receive_lists[r] = NULL;
        if (r == rank)
            continue;

        // Set marks to zero
        for (int i = p[r]; i < p[r + 1]; i++)
            receive_mark[i] = 0;

        // Find separators
        for (int u = p[rank]; u < p[rank + 1]; u++) {
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                int v = g.col_idx[i];
                if (v >= p[r] && v < p[r + 1])
                    receive_mark[v] = 1;
            }
        }

        // Count separators
        for (int i = p[r]; i < p[r + 1]; i++)
            c.receive_count[r] += receive_mark[i];

        if (c.receive_count[r] == 0)
            continue;

        // Store list of separators
        c.receive_items[r] = malloc(sizeof(int) * c.receive_count[r]);
        c.receive_lists[r] = malloc(sizeof(double) * c.receive_count[r]);

        int j = 0;
        for (int i = p[r]; i < p[r + 1]; i++)
            if (receive_mark[i])
                c.receive_items[r][j++] = i;
    }

    free(receive_mark);
}

void exchange_separators(comm_lists c, double *y, int rank, int size) {
    MPI_Request sends[size], receives[size];
    // Start sends
    for (int r = 0; r < size; r++) {
        if (c.send_count[r] == 0)
            continue;
        for (int j = 0; j < c.send_count[r]; j++)
            c.send_lists[r][j] = y[c.send_items[r][j]];

        MPI_Isend(c.send_lists[r], c.send_count[r], MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &sends[r]);
    }
    // Start receives
    for (int r = 0; r < size; r++)
        if (c.receive_count[r] > 0)
            MPI_Irecv(c.receive_lists[r], c.receive_count[r], MPI_DOUBLE, r, 0, MPI_COMM_WORLD, &receives[r]);

    // Wait for receives and unpack
    for (int r = 0; r < size; r++) {
        if (c.receive_count[r] > 0)
            MPI_Wait(&receives[r], MPI_STATUS_IGNORE);

        for (int j = 0; j < c.receive_count[r]; j++)
            y[c.receive_items[r][j]] = c.receive_lists[r][j];
    }

    // Wait for sends (could go after spmv)
    for (int r = 0; r < size; r++)
        if (c.send_count[r] > 0)
            MPI_Wait(&sends[r], MPI_STATUS_IGNORE);
}
