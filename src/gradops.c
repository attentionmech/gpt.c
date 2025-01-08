#define MAX_PARAMS 10000

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
    size_t id;
} Value;

size_t node_counter = 0;

void add_backward(Value* out);
void sub_backward(Value* out);
void power_backward(Value* out);
void mul_backward(Value* out);
void relu_backward(Value* out);
void noop_backward(Value* out);
void mse_loss_backward(Value* loss);


Value* create_value(double data, size_t num_prev, struct Value** prev, const char* op) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0;
    v->_prev = prev;
    v->num_prev = num_prev;
    v->op = op ? strdup(op) : NULL;
    v->_backward = noop_backward;
    v->id = node_counter++;
    return v;
}

void noop_backward(Value* out) {
    // nop
}

Value* add(Value* self, Value* other) {
    Value** prev_nodes = (Value**)malloc(2 * sizeof(Value*));
    if (prev_nodes == NULL) {
        fprintf(stderr, "Memory allocation failed for prev_nodes\n");
        return NULL;
    }

    prev_nodes[0] = self;
    prev_nodes[1] = other;
    Value* out = create_value(self->data + other->data, 2, prev_nodes, "+");
    out->_backward = add_backward;
    return out;
}


Value* mul(Value* self, Value* other) {
    Value** prev_nodes = (Value**)malloc(2 * sizeof(Value*));
    if (prev_nodes == NULL) {
        fprintf(stderr, "Memory allocation failed for prev_nodes\n");
        return NULL;
    }
    prev_nodes[0] = self;
    prev_nodes[1] = other;
    Value* out = create_value(self->data * other->data, 2, prev_nodes, "*");
    out->_backward = mul_backward;
    return out;
}

Value* relu(Value* self) {
    Value** prev_nodes = (Value**)malloc(sizeof(Value*));
    if (prev_nodes == NULL) {
        fprintf(stderr, "Memory allocation failed for prev_nodes\n");
        return NULL;
    }

    prev_nodes[0] = self;
    Value* out = create_value(self->data > 0 ? self->data : 0, 1, prev_nodes, "ReLU");
    out->_backward = relu_backward;
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

void relu_backward(Value* out) {
    out->_prev[0]->grad += (out->data > 0) * out->grad;
}

Value* sub(Value* a, Value* b) {
    double result = a->data - b->data;

    Value** prev_nodes = (Value**)malloc(2 * sizeof(Value*));
    prev_nodes[0] = a;
    prev_nodes[1] = b;

    Value* result_val = create_value(result, 2, prev_nodes, "sub");
    result_val->_backward = sub_backward;

    return result_val;
}

void sub_backward(Value* out) {
    Value* a = out->_prev[0];
    Value* b = out->_prev[1];

    a->grad += out->grad;
    b->grad -= out->grad;
}


Value* power(Value* a, double exponent) {
    double result = pow(a->data, exponent);

    
    Value** prev_nodes = (Value**)malloc(2 * sizeof(Value*));
    prev_nodes[0] = a;
    prev_nodes[1] = create_value(exponent, 0, NULL, "exponent-val");

    Value* result_val = create_value(result, 2, prev_nodes, "power");
    result_val->_backward = power_backward;

    return result_val;
}

void power_backward(Value* out) {
    Value* a = out->_prev[0];
    double exponent = (out->_prev[1])->data;
    a->grad += out->grad * (exponent-1) * a->data; 
}


void build_topo(Value* v, Value** topo, int* idx, char* visited) {
    if (!visited[v->id] && v!= NULL) {
        visited[v->id] = 1;
        for (size_t i = 0; i < v->num_prev; i++) {
            build_topo(v->_prev[i], topo, idx, visited);
        }
        topo[*idx] = v;
        (*idx)++;
    }
}

void backward(Value* v) {
    v->grad = 1.0;
    
    char visited[MAX_PARAMS] = {0};
    Value* topo[MAX_PARAMS];
    int idx = 0;
    
    build_topo(v, topo, &idx, visited);

    for (int i = idx; i > 0; i--) {
        topo[i - 1]->_backward(topo[i - 1]);
    }
}

// int main() {
//     Value* v1 = create_value(2.0, 0, NULL, "input");
//     Value* v0 = create_value(2.0, 0, NULL, "input");
//     Value* v3 = create_value(4.0,0,NULL,"input");


//     Value* v2 = mul(v0, v1);
//     Value* v4 = add(v3, v2);

//     backward(v4);

//     return 0;
// }
