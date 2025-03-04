#include "mtx.h"
#include "spmv.h"
#include <math.h>
#include <stdlib.h>
int main(int argc, char **argv) {
    CSR g;
    g = parse_and_validate_mtx(argv[1]);

    printf("values: ");
    for (int i = 0; i < g.num_cols; i++) {
        printf("%f ", g.values[i]);
    }
    printf("\n");

    for (int i = 0; i < g.num_rows + 1; i++) {
        printf("%d -> ", g.row_ptr[i]);
        for (int j = g.row_ptr[i]; j < g.row_ptr[i + 1]; j++) {
            printf("%d ", g.col_idx[j]);
        }
        printf("\n");
    }

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

        // for (int i = 0; i < g.num_rows; i++) {
        //     printf("%f ", x[i]);
        // }
        // printf("\n");
    }

    for (int i = 0; i < g.num_rows; i++) {
        printf("%f\n", x[i]);
    }
    printf("\n");

    double l2 = 0.0;
    for (int i = 0; i < g.num_rows; i++)
        l2 += x[i] * x[i];
    l2 = sqrt(l2);

    printf("L2 norm: %f\n", l2);

    free(x);
    free(y);
    return 0;
}
