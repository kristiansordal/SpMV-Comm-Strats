#pragma once

#include "mtx.h"
#define nonzero 4

typedef struct {
    int N;
    double *A;
    int *I;
} mesh;

mesh init_mesh_4(int scale, double alpha, double beta);

// void reorder_separators(CSR g, int size, int rows, int *partition_idx, int *num_separators);

void free_mesh(mesh *m);

void step_ref(mesh m, double *Vold, double *Vnew);

void step_par(mesh m, double *Vold, double *Vnew);
