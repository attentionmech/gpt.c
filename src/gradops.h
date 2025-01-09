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

Value *create_value(double data, size_t num_prev, struct Value **prev, const char *op, int temp);

Value *add(Value *self, Value *other);
Value *sub(Value *self, Value *other);
Value *power(Value *self, double exponent);
Value *mul(Value *self, Value *other);
Value *relu(Value *self);
Value *sigmoid(Value *self);

void generate_graphviz(Value *v);
void backward(Value *v);