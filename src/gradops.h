#include <stdlib.h>

typedef struct Value {
    double data;  
    double grad;  
    struct Value** _prev;  
    size_t num_prev;  
    char* op; 
    void (*_backward)(struct Value*); 
} Value;

Value* create_value(double data, size_t num_prev, struct Value** prev, const char* op);

Value* add(Value* self, Value* other);
Value* mul(Value* self, Value* other);
Value* relu(Value* self);

void add_backward(Value* out);
void mul_backward(Value* out);
void relu_backward(Value* out);