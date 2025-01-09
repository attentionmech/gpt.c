#include <stdlib.h>

typedef struct Value
{
    double data;
    double grad;
    struct Value **_prev;
    size_t num_prev;
    char *op;
    void (*_backward)(struct Value *);
} Value;

void print_counter();

Value *create_value(double data, size_t num_prev, struct Value **prev, const char *op);

Value *add(Value *self, Value *other, Value *out, int update);
Value *sub(Value *self, Value *other, Value *out, int update);
Value *power(Value *self, double exponent, Value *out, int update);
Value *mul(Value *self, Value *other, Value *out, int update);
Value *relu(Value *self, Value* result, int update);

void add_backward(Value *out);
void mul_backward(Value *out);
void relu_backward(Value *out);
void sub_backward(Value *out);
void power_backward(Value *out, double exponent);
void generate_graphviz(Value *v);

void backward(Value *v);