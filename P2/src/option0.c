#include "mtx.h"
#include "spmv.h"
#include <math.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    CSR g;
    g = parse_and_validate_mtx(argv[1]);

    double *x = malloc(sizeof(double) * g.num_rows);
    double *y = malloc(sizeof(double) * g.num_rows);

    for (int i = 0; i < g.num_rows; i++) {
        x[i] = 2.0;
        y[i] = 2.0;
    }

    for (int i = 0; i < 2; i++) {
        spmv(g, x, y);
        double *tmp = x;
        x = y;
        y = tmp;
    }

    double l2 = 0.0;
    for (int i = 0; i < g.num_rows; i++)
        l2 += x[i] * x[i];
    l2 = sqrt(l2);

    printf("L2 norm: %f\n", l2);

    free(x);
    free(y);
    return 0;
}
