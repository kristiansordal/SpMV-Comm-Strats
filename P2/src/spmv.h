#pragma once
#include "mtx.h"

void spmv(CSR g, double *x, double *y);

void spmv_part(CSR g, int s, int t, double *x, double *y);

void partition_graph(CSR g, int k, int *p, double *x);

void partition_graph_naive(CSR g, int s, int t, int k, int *p);

void distribute_graph(CSR *g, int rank);

typedef struct {
    int *send_count, *receive_count;
    int **send_items, **receive_items;
    double **send_lists, **receive_lists;
} comm_lists;

comm_lists init_comm_lists(int size);

void free_comm_lists(comm_lists *c, int size);

void find_sendlists(CSR g, int *p, int rank, int size, comm_lists c);

void find_receivelists(CSR g, int *p, int rank, int size, comm_lists c);

void exchange_separators(comm_lists c, double *y, int rank, int size);
