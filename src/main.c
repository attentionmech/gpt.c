#include <stdio.h>
#include <stdlib.h>
#include "matops.h"

int main() {


    float **A = (float**)malloc(2 * sizeof(float*));
    float **B = (float**)malloc(2 * sizeof(float*));

    for (int i = 0; i < 2; i++) {
        A[i] = (float*)malloc(2 * sizeof(float));
        B[i] = (float*)malloc(2 * sizeof(float));
    }

    // Initialize A and B
    A[0][0] = 1.0; A[0][1] = 2.0;
    A[1][0] = 3.0; A[1][1] = 4.0;

    B[0][0] = 5.0; B[0][1] = 6.0;
    B[1][0] = 7.0; B[1][1] = 8.0;

    // Test matrix multiplication
    float **C = multiply(A, B, 2, 2);

    printf("Matrix multiplication result:\n");
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%.2f ", C[i][j]);
        }
        printf("\n");
    }

    float **AT = transpose(A,2,2);
    
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%.2f ", AT[i][j]);
        }
        printf("\n");
    }

    float **ADD = addition(A,B, 2,2);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            printf("%.2f ", ADD[i][j]);
        }
        printf("\n");
    }


    // Free memory
    for (int i = 0; i < 2; i++) {
        free(A[i]);
        free(B[i]);
        free(C[i]);
    }
    free(A);
    free(B);
    free(C);


    return 0;
}
