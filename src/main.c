#include <stdio.h>
#include <stdlib.h>
#include "matops.h"

int main() {
    int size = 3;
    float A[] = {1.0, 2.0, 3.0};
    float B[] = {4.0, 5.0, 6.0};

    float *dot_result = dot_product(A, B, size);
    printf("Dot Product: %.2f\n", *dot_result);
    free(dot_result);

    return 0;
}
