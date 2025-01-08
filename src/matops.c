#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matops.h"

float** add(float **A, float **B, int rows, int cols) {
    float **C = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        C[i] = (float*)malloc(cols * sizeof(float));
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] + B[i][j];
        }
    }
    return C;
}

float** transpose(float **A, int rows, int cols) {
    float **AT = (float**)malloc(cols * sizeof(float*));
    for (int i = 0; i < cols; i++) {
        AT[i] = (float*)malloc(rows * sizeof(float));
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

float** multiply(float **A, float **B, int rows, int cols) {
    float **C = (float**)malloc(rows * sizeof(float*));
    for (int i = 0; i < rows; i++) {
        C[i] = (float*)malloc(cols * sizeof(float));
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            C[i][j] = A[i][j] * B[i][j];
        }
    }
    return C;
}

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

