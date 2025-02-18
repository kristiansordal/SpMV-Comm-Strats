#include "p2.h"
#include "mtx.h"
#include <stdalign.h>
#include <stdlib.h>

mesh init_mesh_4(int scale, double alpha, double beta) {
    int N = 1 << scale;

    mesh m;
    m.N = N * N;

    m.A = aligned_alloc(32, sizeof(double) * m.N * 4);
    m.I = aligned_alloc(16, sizeof(int) * m.N * 4);

    for (size_t i = 0; i < m.N; i++) {
        m.A[i * 4] = beta;
        m.A[i * 4 + 1] = beta;
        m.A[i * 4 + 2] = beta;
        m.A[i * 4 + 3] = alpha;

        m.I[i * 4] = i - 1;
        m.I[i * 4 + 1] = i + 1;
        m.I[i * 4 + 2] = (i & 1) ? i - N - 1 : i + N + 1;
        m.I[i * 4 + 3] = i;

        if ((i % N) == 0) // First element in row
        {
            m.I[i * 4] = i;
            m.A[i * 4] = 0.0;
        }

        if ((!(i & 1) && i >= N * N - N) || // Last row
            ((i & 1) && i < N))             // First row
        {
            m.I[i * 4 + 2] = i;
            m.A[i * 4 + 2] = 0.0;
        }

        if ((i % N) == N - 1) // Last element in row
        {
            m.I[i * 4 + 1] = i;
            m.A[i * 4 + 1] = 0.0;
        }
    }

    return m;
}

/* Reorders separators by putting them at indices ranging from p[rank]-> p[rank] + sep[rank]
 * CSR g: graph
 * int size: number of ranks
 * int *p: partition vector
 * int *num_separator: number of separators for each rank
 * int *old_id: old id of each vertex
 * int *new_id: new id of each vertex
 */
void reorder_separators(CSR g, int size, int rows, int *p, int *num_separators) {
    int *sep_marker = calloc(g.num_rows, sizeof(int));
    int *new_id = malloc(g.num_rows * sizeof(int));
    int total_rows = g.num_rows;

    // First pass: count separators
    for (size_t rank = 0; rank < size; rank++) {
        num_separators[rank] = 0;
        for (size_t i = p[rank]; i < p[rank + 1]; i++) {
            sep_marker[i] = 0;
            for (int j = g.row_ptr[i]; j < g.row_ptr[i + 1]; j++) {
                int u = g.col_idx[j];
                if (u < p[rank] || u >= p[rank + 1]) { // If neighbor is outside rank
                    num_separators[rank]++;
                    sep_marker[i] = 1;
                    break;
                }
            }
        }
    }

    // Compute new row order
    int *new_row_ptr = malloc((total_rows + 1) * sizeof(int));
    int *new_col_idx = malloc(g.num_cols * sizeof(int));
    double *new_values = malloc(g.num_cols * sizeof(double));

    int row_idx = 0, col_idx = 0;
    for (size_t rank = 0; rank < size; rank++) {
        int sep_count = 0, non_sep_count = 0;

        // First, place separators at the start of the rank’s range
        for (size_t i = p[rank]; i < p[rank + 1]; i++) {
            if (sep_marker[i]) {
                new_id[i] = p[rank] + sep_count++;
            } else {
                new_id[i] = p[rank] + num_separators[rank] + non_sep_count++;
            }
        }

        // Copy row data in new order
        for (size_t i = p[rank]; i < p[rank + 1]; i++) {
            int new_pos = new_id[i];
            new_row_ptr[new_pos] = col_idx;
            for (size_t j = g.row_ptr[i]; j < g.row_ptr[i + 1]; j++) {
                new_col_idx[col_idx] = g.col_idx[j];
                new_values[col_idx] = g.values[j];
                col_idx++;
            }
        }
    }
    new_row_ptr[total_rows] = col_idx; // Final row pointer

    // Free old arrays
    free(g.row_ptr);
    free(g.col_idx);
    free(g.values);

    // Assign new arrays
    g.row_ptr = new_row_ptr;
    g.col_idx = new_col_idx;
    g.values = new_values;

    free(sep_marker);
}

void free_mesh(mesh *m) {
    m->N = 0;
    free(m->A);
    free(m->I);
}

void step_ref(mesh m, double *Vold, double *Vnew) {
    for (size_t i = 0; i < m.N; i++) {
        Vnew[i] = 0.0;
        for (size_t j = 0; j < nonzero; j++)
            Vnew[i] += m.A[i * nonzero + j] * Vold[m.I[i * nonzero + j]];
    }
}

void step_par(mesh m, double *Vold, double *Vnew) {
#pragma omp parallel for
    for (size_t i = 0.0; i < m.N; i++) {
        Vnew[i] = 0.0;
        for (size_t j = 0; j < nonzero; j++)
            Vnew[i] += m.A[i * nonzero + j] * Vold[m.I[i * nonzero + j]];

        // Vnew[i] = m.A[i * nonzero + 0] * Vold[m.I[i * nonzero + 0]] +
        //           m.A[i * nonzero + 1] * Vold[m.I[i * nonzero + 1]] +
        //           m.A[i * nonzero + 2] * Vold[m.I[i * nonzero + 2]] +
        //           m.A[i * nonzero + 3] * Vold[m.I[i * nonzero + 3]];
    }
}
