#ifndef MATOPS_H
#define MATOPS_H

#include "gradops.h"

Value *dot_product(Value *A, Value *B, int size);
Value **matmul(Value **A, Value **B, int m, int n, int p);
Value **multiply(Value **A, Value **B, int rows, int cols);
Value **transpose(Value **A, int rows, int cols);
Value **addition(Value **A, Value **B, int rows, int cols);

#endif