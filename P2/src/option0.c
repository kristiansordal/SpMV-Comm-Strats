#include "mtx.h"
#include "spmv.h"
#include <stdlib.h>
int main(int argc, char **argv) {
    CSR g;
    g = parse_and_validate_mtx(argv[1]);

    double *x = malloc(sizeof(double) * g.num_rows);
    double *y = malloc(sizeof(double) * g.num_rows);

    for (int i = 0; i < g.num_rows; i++) {
        x[i] = 1.0;
        y[i] = 1.0;
    }

    for (int i = 0; i < 5; i++) {
        spmv(g, x, y);
        double *tmp = x;
        x = y;
        y = tmp;
    }

    double l2 = 0.0;
    for (int i = 0; i < g.num_rows; i++)
        l2 += x[i] * x[i];

    printf("L2 norm: %f\n", l2);

    free(x);
    free(y);
    return 0;
}
