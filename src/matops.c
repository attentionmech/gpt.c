#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matops.h"


float* dot_product(float *A, float *B, int size) {
    float *result = (float*)malloc(sizeof(float));
    if (result == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    *result = 0;
    for (int i = 0; i < size; ++i) {
        *result += A[i] * B[i];
    }
    return result;
}
