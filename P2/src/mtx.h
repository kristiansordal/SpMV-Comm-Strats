#pragma once
#include <stdio.h>

typedef struct {
    int num_rows, num_cols;
    int *row_ptr, *col_idx;
    double *values;
} CSR;

CSR parse_and_validate_mtx(const char *path);

CSR parse_mtx(FILE *f);

void free_graph(CSR *g);

void sort_edges(CSR g);

void normalize_graph(CSR g);

int validate_graph(CSR g);
