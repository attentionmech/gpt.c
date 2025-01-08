#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct Value {
    double data;  
    double grad;  
    struct Value** _prev;  
    size_t num_prev;  
    char* op; 
    void (*_backward)(struct Value*); 
} Value;

void add_backward(Value* out);
void mul_backward(Value* out);


Value* create_value(double data, size_t num_prev, struct Value** prev, const char* op) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0;
    v->_prev = prev;
    v->num_prev = num_prev;
    v->op = op ? strdup(op) : NULL;
    v->_backward = NULL;
    return v;
}


Value* add(Value* self, Value* other) {
    Value* out = create_value(self->data + other->data, 2, (Value*[]){self, other}, "+");
    out->_backward = add_backward;
    return out;
}

Value* mul(Value* self, Value* other) {
    Value* out = create_value(self->data * other->data, 2, (Value*[]){self, other}, "*");
    out->_backward = mul_backward;
    return out;
}

void add_backward(Value* out) {
    out->_prev[0]->grad += out->grad;
    out->_prev[1]->grad += out->grad;
}

void mul_backward(Value* out) {
    out->_prev[0]->grad += out->_prev[1]->data * out->grad;
    out->_prev[1]->grad += out->_prev[0]->data * out->grad;
}
