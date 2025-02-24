#include "spmv.h"
#include "p2.h"
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

void determine_separators(CSR g, int num_partitions, int *partition_idx, int *partition_map, double *x) {
    int *new_row_ptr = malloc(sizeof(int) * (g.num_rows + 1));
    int *new_col_idx = malloc(sizeof(int) * g.num_cols);
    double *new_values = malloc(sizeof(double) * g.num_cols);

    int *sep_marker = malloc(sizeof(int) * g.num_rows);

    for (int i = 0; i < g.num_rows; i++)
        sep_marker[i] = 0;

    for (int i = 0; i < g.num_rows + 1; i++)
        new_row_ptr[i] = 0;

    new_row_ptr[g.num_rows] = g.nnz;

    for (int i = 0; i < g.num_cols; i++) {
        new_col_idx[i] = 0;
        new_values[i] = 0;
    }

    int *new_id = malloc(sizeof(int) * g.num_rows);
    int *old_id = malloc(sizeof(int) * g.num_rows);

    for (int r = 0; r < num_partitions; r++) {
        int idx = 0;
        for (int i = partition_idx[r]; i < partition_idx[r + 1]; i++) {
            if (sep_marker[i]) {
                old_id[idx] = i;
                new_id[i] = idx++;
                // new_row_ptr[i] = partition_idx[r] + idx;
                // for (int u = g.row_ptr[i]; u < g.row_ptr[i + 1]; u++) {
                //     new_col_idx[partition_idx[r] + idx] = g.col_idx[u];
                //     new_values[partition_idx[r] + idx++] = g.values[u];
                // }
            }
        }

        for (int i = partition_idx[r]; i < partition_idx[r + 1]; i++) {
            if (!sep_marker[i]) {
                new_row_ptr[i] = partition_idx[r] + idx;
                for (int u = g.row_ptr[i]; u < g.row_ptr[i + 1]; u++) {
                    new_col_idx[partition_idx[r] + idx] = g.col_idx[u];
                    new_values[partition_idx[r] + idx++] = g.values[u];
                }
            }
        }
    }

    // int *new_id = malloc(sizeof(int) * g.num_rows);
    // int *old_id = malloc(sizeof(int) * g.num_rows);
    // int id = 0;
}

void partition_graph(CSR g, int num_partitions, int *partition_idx, double *x, comm_lists *c) {
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

    int *sep_marker = calloc(g.num_rows, sizeof(int));

    for (int i = 0; i < g.num_rows; i++) {
        for (int j = g.row_ptr[i]; j < g.row_ptr[i + 1]; j++) {
            if (part[i] != part[g.col_idx[j]]) {
                sep_marker[i] = 1; // Mark separator nodes
                c->send_count[part[i]]++;
                break;
            }
        }
    }

    int *new_id = malloc(sizeof(int) * g.num_rows);
    int *old_id = malloc(sizeof(int) * g.num_rows);
    int id = 0;
    partition_idx[0] = 0;
    for (int r = 0; r < num_partitions; r++) {

        for (int i = 0; i < g.num_rows; i++) {
            if (part[i] == r && sep_marker[i]) {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }
        for (int i = 0; i < g.num_rows; i++) {
            if (part[i] == r && !sep_marker[i]) {
                old_id[id] = i;
                new_id[i] = id++;
            }
        }
        partition_idx[r + 1] = id;
    }

    int *new_V = malloc(sizeof(int) * (g.num_rows + 1));
    int *new_E = malloc(sizeof(int) * g.num_cols);
    double *new_A = malloc(sizeof(double) * g.num_cols);

    // reorder according to the partition - makes sense
    // need to do something similar for separators
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
    g.nnz = g.num_cols;

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
    c.send_mark = malloc(sizeof(int) * g.num_rows);
    c.send_count_total = 0;
    // int send_count_total = 0;
    for (int r = 0; r < size; r++) {
        c.send_count[r] = 0;
        c.send_items[r] = NULL;
        c.send_lists[r] = NULL;
        if (r == rank)
            continue;

        // Set marks to zero
        for (int i = p[rank]; i < p[rank + 1]; i++)
            c.send_mark[i] = 0;

        // Find separators
        for (int u = p[r]; u < p[r + 1]; u++) {
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                int v = g.col_idx[i];
                if (v >= p[rank] && v < p[rank + 1]) {
                    c.send_mark[v] = 1;
                    c.send_count_total++;
                }
            }
        }
        // Count separators
        for (int i = p[rank]; i < p[rank + 1]; i++)
            c.send_count[r] += c.send_mark[i];

        if (c.send_count[r] == 0)
            continue;

        // Store list of separators
        c.send_items[r] = malloc(sizeof(int) * c.send_count[r]);
        c.send_lists[r] = malloc(sizeof(double) * c.send_count[r]);

        // for (int i = 0; i < c.send_count[r]; i++) {
        //     printf("%d, c.send_items[%d][%d] = %d\n", rank, r, i, c.send_items[r][i]);
        // }

        int j = 0;
        for (int i = p[rank]; i < p[rank + 1]; i++)
            if (c.send_mark[i])
                c.send_items[r][j] = i;
    }

    c.send_items_flat = malloc(sizeof(int) * c.send_count_total);
    c.send_lists_flat = malloc(sizeof(double) * c.send_count_total);

    int idx = 0;
    for (int i = 0; i < g.num_rows; i++) {
        if (c.send_mark[i]) {
            c.send_items_flat[idx] = i;
            c.send_lists_flat[idx++] = i;
        }
    }

    // free(c.send_mark);
}

void find_receivelists(CSR g, int *p, int rank, int size, comm_lists c) {
    c.receive_mark = malloc(sizeof(int) * g.num_rows);
    c.receive_count_total = 0;
    for (int r = 0; r < size; r++) {
        c.receive_count[r] = 0;
        c.receive_items[r] = NULL;
        c.receive_lists[r] = NULL;
        if (r == rank)
            continue;

        // Set marks to zero
        for (int i = p[r]; i < p[r + 1]; i++)
            c.receive_mark[i] = 0;

        // Find separators
        for (int u = p[rank]; u < p[rank + 1]; u++) {
            for (int i = g.row_ptr[u]; i < g.row_ptr[u + 1]; i++) {
                int v = g.col_idx[i];
                if (v >= p[r] && v < p[r + 1])
                    c.receive_mark[v] = 1;
            }
        }

        // Count separators
        for (int i = p[r]; i < p[r + 1]; i++)
            c.receive_count[r] += c.receive_mark[i];

        if (c.receive_count[r] == 0)
            continue;

        // Store list of separators
        c.receive_items[r] = malloc(sizeof(int) * c.receive_count[r]);
        c.receive_lists[r] = malloc(sizeof(double) * c.receive_count[r]);

        int j = 0;
        for (int i = p[r]; i < p[r + 1]; i++)
            if (c.receive_mark[i])
                c.receive_items[r][j++] = i;
    }

    c.receive_items_flat = malloc(sizeof(int) * c.receive_count_total);
    c.receive_lists_flat = malloc(sizeof(double) * c.receive_count_total);

    int idx = 0;
    for (int i = 0; i < g.num_rows; i++) {
        if (c.receive_mark[i]) {
            c.receive_items_flat[idx] = i;
            c.receive_lists_flat[idx++] = i;
        }
    }

    // free(c.receive_mark);
}

void reorder_separators(CSR g, int *p, int rank, int size, comm_lists c) {
    int n_local = p[rank + 1] - p[rank];
    int nnz_local = g.row_ptr[p[rank + 1]] - g.row_ptr[p[rank]];

    int *new_row_ptr_local = malloc(sizeof(int) * (n_local + 1));
    int *new_col_idx_local = malloc(sizeof(int) * nnz_local);
    double *new_values_local = malloc(sizeof(double) * nnz_local);

    int row_idx = 0;
    int col_idx = 0;
    new_row_ptr_local[0] = 0;

    for (int i = p[rank]; i < p[rank + 1]; i++) {
        if (c.send_mark[i]) {
            new_row_ptr_local[row_idx + 1] = new_row_ptr_local[row_idx];
            for (int j = g.row_ptr[i]; j < g.row_ptr[i + 1]; j++) {
                new_col_idx_local[col_idx] = g.col_idx[j];
                new_values_local[col_idx] = g.values[j];
                col_idx++;
                new_row_ptr_local[row_idx + 1]++;
            }
            row_idx++;
        }
    }

    for (int i = p[rank]; i < p[rank + 1]; i++) {
        if (!c.send_mark[i]) {
            new_row_ptr_local[row_idx + 1] = new_row_ptr_local[row_idx];
            for (int j = g.row_ptr[i]; j < g.row_ptr[i + 1]; j++) {
                new_col_idx_local[col_idx] = g.col_idx[j];
                new_values_local[col_idx] = g.values[j];
                col_idx++;
                new_row_ptr_local[row_idx + 1]++;
            }
            row_idx++;
        }
    }

    free(g.row_ptr);
    free(g.col_idx);
    free(g.values);

    g.row_ptr = new_row_ptr_local;
    g.col_idx = new_col_idx_local;
    g.values = new_values_local;
}

void gather_reordered_csr(CSR *g, int rank, int size, int *p, comm_lists c) {
    int local_nnz = g->row_ptr[p[rank + 1]] - g->row_ptr[p[rank]]; // Local nonzeros
    int local_rows = p[rank + 1] - p[rank];                        // Local rows

    // Step 1: Gather all nonzero counts and row counts
    int *all_nnz_counts = NULL, *displs_nnz = NULL;
    int *all_row_counts = NULL, *displs_row = NULL;

    if (rank == 0) {
        all_nnz_counts = malloc(sizeof(int) * size);
        displs_nnz = malloc(sizeof(int) * size);
        all_row_counts = malloc(sizeof(int) * size);
        displs_row = malloc(sizeof(int) * size);
    }

    MPI_Gather(&local_nnz, 1, MPI_INT, all_nnz_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_rows, 1, MPI_INT, all_row_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute displacements
    if (rank == 0) {
        displs_nnz[0] = 0;
        displs_row[0] = 0;
        for (int i = 1; i < size; i++) {
            displs_nnz[i] = displs_nnz[i - 1] + all_nnz_counts[i - 1];
            displs_row[i] = displs_row[i - 1] + all_row_counts[i - 1];
        }
    }

    // Step 2: Allocate buffers for global storage on rank 0
    int *global_col_idx = NULL;
    double *global_values = NULL;
    int *global_row_ptr = NULL;

    if (rank == 0) {
        int total_nnz = displs_nnz[size - 1] + all_nnz_counts[size - 1];
        int total_rows = displs_row[size - 1] + all_row_counts[size - 1];

        global_col_idx = malloc(sizeof(int) * total_nnz);
        global_values = malloc(sizeof(double) * total_nnz);
        global_row_ptr = malloc(sizeof(int) * (total_rows + 1));
    }

    // Step 3: Gather col_idx and values
    MPI_Gatherv(g->col_idx + g->row_ptr[p[rank]], local_nnz, MPI_INT, global_col_idx, all_nnz_counts, displs_nnz,
                MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Gatherv(g->values + g->row_ptr[p[rank]], local_nnz, MPI_DOUBLE, global_values, all_nnz_counts, displs_nnz,
                MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 4: Gather row_ptr
    MPI_Gatherv(g->row_ptr + p[rank], local_rows, MPI_INT, global_row_ptr, all_row_counts, displs_row, MPI_INT, 0,
                MPI_COMM_WORLD);

    // Step 5: Finalize & Cleanup
    if (rank == 0) {
        // Assign gathered arrays to global CSR structure (if needed)
        g->col_idx = global_col_idx;
        g->values = global_values;
        g->row_ptr = global_row_ptr;
    }

    free(all_nnz_counts);
    free(displs_nnz);
    free(all_row_counts);
    free(displs_row);
}

// void reorder_separators(CSR g, int size, int rows, int *sep, int *old_id, int *new_id) { return; }
void exchange_separators(comm_lists c, double *Vn, int rank, int size) {
    int total_send = 0, total_recv = 0;

    // Compute total send and receive counts
    for (int i = 0; i < size; i++) {
        total_send += c.send_count[i];
        total_recv += c.receive_count[i];
    }

    // Allocate send and receive buffers
    double *send_buffer = malloc(sizeof(double) * total_send);
    double *recv_buffer = malloc(sizeof(double) * total_recv);

    // Pack send buffer
    int send_offset = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < c.send_count[i]; j++) {
            send_buffer[send_offset++] = Vn[c.send_items[i][j]];
        }
    }

    // Compute displacement arrays for MPI_Alltoallv
    int *sdispls = malloc(sizeof(int) * size);
    int *rdispls = malloc(sizeof(int) * size);
    sdispls[0] = 0;
    rdispls[0] = 0;

    for (int i = 1; i < size; i++) {
        sdispls[i] = sdispls[i - 1] + c.send_count[i - 1];
        rdispls[i] = rdispls[i - 1] + c.receive_count[i - 1];
    }

    // Exchange separator values using MPI_Alltoallv
    MPI_Alltoallv(send_buffer, c.send_count, sdispls, MPI_DOUBLE, recv_buffer, c.receive_count, rdispls, MPI_DOUBLE,
                  MPI_COMM_WORLD);

    // Unpack received values into Vn
    int recv_offset = 0;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < c.receive_count[i]; j++) {
            Vn[c.receive_items[i][j]] = recv_buffer[recv_offset++];
        }
    }

    free(send_buffer);
    free(recv_buffer);
    free(sdispls);
    free(rdispls);
}
