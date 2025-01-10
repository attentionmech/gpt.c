#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "matops.h"
#include "gradops.h"

Value **addition(Value **A, Value **B, int rows, int cols)
{
    Value **C = (Value **)malloc(rows * sizeof(Value *));
    for (int i = 0; i < rows; i++)
    {
        C[i] = (Value *)malloc(cols * sizeof(Value));
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {

            C[i][j] = *add(&A[i][j], &B[i][j]);
        }
    }
    return C;
}

Value **transpose(Value **A, int rows, int cols)
{
    Value **AT = (Value **)malloc(cols * sizeof(Value *));
    for (int i = 0; i < cols; i++)
    {
        AT[i] = (Value *)malloc(rows * sizeof(Value));
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {

            AT[j][i] = A[i][j];
        }
    }
    return AT;
}

Value **multiply(Value **A, Value **B, int rows, int cols)
{
    Value **C = (Value **)malloc(rows * sizeof(Value *));
    for (int i = 0; i < rows; i++)
    {
        C[i] = (Value *)malloc(cols * sizeof(Value));
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {

            C[i][j] = *mul(&A[i][j], &B[i][j]);
        }
    }
    return C;
}

Value *dot_product(Value *A, Value *B, int size)
{
    Value *result = (Value *)malloc(sizeof(Value));
    result->data = 0;
    result->grad = 0;

    for (int i = 0; i < size; ++i)
    {
        result->data += A[i].data * B[i].data;
    }
    return result;
}

Value **matmul(Value **A, Value **B, int m, int n, int p)
{
    Value **C = (Value **)malloc(m * sizeof(Value *));
    for (int i = 0; i < m; i++)
    {
        C[i] = (Value *)malloc(p * sizeof(Value));
    }

    for (int i = 0; i < m; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            C[i][j].data = 0;
            for (int k = 0; k < n; ++k)
            {
                C[i][j] = *add(&C[i][j], mul(&A[i][k], &B[k][j]));
            }
        }
    }
    return C;
}
