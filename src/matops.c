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

float** matmul(float **A, float **B, int m, int n, int p) {

    float **C = (float**)malloc(m * sizeof(float*));
    for (int i = 0; i < m; i++) {
        C[i] = (float*)malloc(p * sizeof(float));
    }

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < p; ++j) {
            C[i][j] = 0;
            for (int k = 0; k < n; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}