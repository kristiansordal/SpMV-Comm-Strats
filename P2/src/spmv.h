#pragma once
#include "mtx.h"
typedef struct {
    int send_count_total, receive_count_total;
    int *send_mark, *receive_mark;
    int *send_count, *receive_count;
    int *send_items_flat, *receive_items_flat;
    double *send_lists_flat, *receive_lists_flat;
    int **send_items, **receive_items;
    double **send_lists, **receive_lists;
} comm_lists;

void spmv(CSR g, double *x, double *y);

void spmv_part(CSR g, int rank, int s, int t, double *x, double *y);

void partition_graph_1b(CSR g, int k, int *p, double *x, comm_lists *c);

void partition_graph_1c(CSR g, int k, int *p, double *x, comm_lists *c);

void find_receivelists(CSR g, int *p, int rank, int size, comm_lists c);

void find_sendlists(CSR g, int *p, int rank, int size, comm_lists c);

void partition_graph(CSR g, int num_partitions, int *partition_idx, double *x);

void partition_graph_naive(CSR g, int s, int t, int k, int *p);

void distribute_graph(CSR *g, int rank);

comm_lists init_comm_lists(int size);

void free_comm_lists(comm_lists *c, int size);

void find_sendlists(CSR g, int *p, int rank, int size, comm_lists c);

void find_receivelists(CSR g, int *p, int rank, int size, comm_lists c);

void reorder_separators(CSR g, int num_partitions, int *partition_idx, double *x, comm_lists *c);

void exchange_separators(comm_lists c, double *y, int rank, int size);
