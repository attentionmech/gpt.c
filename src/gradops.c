#define MAX_PARAMS 1000000
#define MAX_TEMP_PARAMS 10000000
#define TEMP_COUNTER_LOW_BOUND  MAX_TEMP_PARAMS+100

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "memgr.h"
#include "chunked_array.h"

size_t node_counter = 0;
long temp_node_counter = TEMP_COUNTER_LOW_BOUND;

void add_backward(Value *out);
void sub_backward(Value *out);
void power_backward(Value *out);
void mul_backward(Value *out);
void relu_backward(Value *out);
void noop_backward(Value *out);
void sigmoid_backward(Value *out);

void print_graphviz(Value *v, FILE *f, char *visited, char *temp_visited);

Value *create_value(double data, size_t num_prev, struct Value **prev, const char *op, int temp)
{
    // printf("create_value called with data: %f, num_prev: %zu, op: %s, temp: %d\n", data, num_prev, op, temp);
    Value *v = (Value *)malloc(sizeof(Value));
    v->data = data;
    v->grad = 0;
    v->_prev = prev;
    v->num_prev = num_prev;
    v->op = op ? strdup(op) : NULL;
    v->_backward = noop_backward;
    if (temp == 0)
    {
        v->id = node_counter++;
    }
    else
    {
        v->id = temp_node_counter++;
    }
    v->exponent = 0;

    if(temp){
        mgr_track_value(v);
    }

    return v;
}

void noop_backward(Value *out)
{
    // nop
}

Value *add(Value *self, Value *other)
{

    Value **prev_nodes;

    prev_nodes = (Value **)malloc(2 * sizeof(Value *));

    if (prev_nodes == NULL)
    {
        fprintf(stderr, "Memory allocation failed for prev_nodes in add\n");
        return NULL;
    }

    if (self == NULL || other == NULL)
    {
        fprintf(stderr, "Error: NULL pointer passed to add function.\n");
        return NULL;
    }

    prev_nodes[0] = self;
    prev_nodes[1] = other;

    Value *out;

    out = create_value(self->data + other->data, 2, prev_nodes, "+", 1);

    out->_backward = add_backward;
    return out;
}

Value *mul(Value *self, Value *other)
{

    Value **prev_nodes;

    prev_nodes = (Value **)malloc(2 * sizeof(Value *));

    if (prev_nodes == NULL)
    {
        fprintf(stderr, "Memory allocation failed for prev_nodes in mul\n");
        return NULL;
    }
    if (self == NULL || other == NULL)
    {
        fprintf(stderr, "Error: NULL pointer passed to mul function.\n");
    }

    prev_nodes[0] = self;
    prev_nodes[1] = other;
    Value *out;

    out = create_value(self->data * other->data, 2, prev_nodes, "*", 1);

    out->_backward = mul_backward;

    return out;
}

Value *relu(Value *self)
{
    Value **prev_nodes;

    prev_nodes = (Value **)malloc(sizeof(Value *));

    if (prev_nodes == NULL)
    {
        fprintf(stderr, "Memory allocation failed for prev_nodes in relu\n");
        return NULL;
    }
    if (self == NULL)
    {
        fprintf(stderr, "Error: NULL pointer passed to relu function.\n");
        return NULL;
    }

    prev_nodes[0] = self;
    Value *out;

    out = create_value(self->data > 0 ? self->data : 0, 1, prev_nodes, "act", 1);
    out->_backward = relu_backward;
    return out;
}

Value *power(Value *a, double exponent)
{
    Value **prev_nodes;

    prev_nodes = (Value **)malloc(1 * sizeof(Value *));

    if (prev_nodes == NULL)
    {
        fprintf(stderr, "Memory allocation failed for prev_nodes in power\n");
        return NULL;
    }
    if (a == NULL)
    {
        fprintf(stderr, "Error: NULL pointer passed to power function.\n");
        return NULL;
    }

    prev_nodes[0] = a;

    double result_data = pow(a->data, exponent);

    Value *out;

    out = create_value(result_data, 1, prev_nodes, "power", 1);
    out->_backward = power_backward;
    out->exponent = exponent;
    return out;
}

Value *sub(Value *a, Value *b)
{
    Value **prev_nodes;

    prev_nodes = (Value **)malloc(2 * sizeof(Value *));

    if (prev_nodes == NULL)
    {
        fprintf(stderr, "Memory allocation failed for prev_nodes in sub\n");
        return NULL;
    }
    if (a == NULL || b == NULL)
    {
        fprintf(stderr, "Error: NULL pointer passed to sub function.\n");
        return NULL;
    }

    prev_nodes[0] = a;
    prev_nodes[1] = b;
    double result_data = a->data - b->data;

    Value *out;

    out = create_value(result_data, 2, prev_nodes, "sub", 1);

    out->_backward = sub_backward;
    return out;
}

Value *sigmoid(Value *self)
{
    Value **prev_nodes;

    prev_nodes = (Value **)malloc(sizeof(Value *));

    if (prev_nodes == NULL)
    {
        fprintf(stderr, "Memory allocation failed for prev_nodes in sigmoid\n");
        return NULL;
    }

    if (self == NULL)
    {
        fprintf(stderr, "Error: NULL pointer passed to sigmoid function.\n");
        return NULL;
    }

    prev_nodes[0] = self;
    Value *out;
    double sigmoid_data = 1 / (1 + exp(-self->data));

    out = create_value(sigmoid_data, 1, prev_nodes, "act", 1);

    out->_backward = sigmoid_backward;
    return out;
}

void sigmoid_backward(Value *out)
{
    Value *a = out->_prev[0];
    a->grad += out->grad * a->data * (1 - a->data);
}

void add_backward(Value *out)
{
    out->_prev[0]->grad += out->grad;
    out->_prev[1]->grad += out->grad;
}

void mul_backward(Value *out)
{
    out->_prev[0]->grad += out->_prev[1]->data * out->grad;
    out->_prev[1]->grad += out->_prev[0]->data * out->grad;
}

void relu_backward(Value *out)
{
    out->_prev[0]->grad += (out->data > 0) * out->grad;
}

void sub_backward(Value *out)
{
    Value *a = out->_prev[0];
    Value *b = out->_prev[1];

    a->grad += out->grad;
    b->grad -= out->grad;
}

void power_backward(Value *out)
{
    Value *a = out->_prev[0];
    double exponent = out->exponent;
    a->grad += out->grad * (exponent - 1) * a->data;
}



void build_topo(Value *v, ChunkedArray *visited, ChunkedArray *temp_visited, ChunkedArray *topo, int *idx)
{
    if (v->id < TEMP_COUNTER_LOW_BOUND)
    {
        if (!chunked_array_get(visited, v->id))
        {
            chunked_array_add(visited, v->id, v);
            for (size_t i = 0; i < v->num_prev; i++)
            {
                build_topo(v->_prev[i], visited, temp_visited, topo, idx);
            }
            chunked_array_add(topo, *idx, (Value*)v);
            (*idx)++;
        }
    }
    else
    {
        if (!chunked_array_get(temp_visited, v->id - TEMP_COUNTER_LOW_BOUND))
        {
            chunked_array_add(temp_visited, v->id - TEMP_COUNTER_LOW_BOUND, v);
            for (size_t i = 0; i < v->num_prev; i++)
            {
                build_topo(v->_prev[i], visited, temp_visited, topo, idx);
            }
            chunked_array_add(topo, *idx, (Value*)v);
            (*idx)++;
        }
    }
}

void backward(Value *v)
{
    v->grad = 1.0;

    int max_params = MAX_PARAMS;
    int max_temp_params = MAX_TEMP_PARAMS;

    ChunkedArray *visited = chunked_array_init(max_params, 1000);
    ChunkedArray *temp_visited = chunked_array_init(max_temp_params, 1000);
    ChunkedArray *topo = chunked_array_init(max_params + max_temp_params, 1000);

    int idx = 0;
    build_topo(v, visited, temp_visited, topo, &idx);

    for (int i = idx; i > 0; i--) {
        Value *topo_value = chunked_array_get(topo, i - 1);
        topo_value->_backward(topo_value);
    }

    chunked_array_cleanup(visited);
    chunked_array_cleanup(temp_visited);
    chunked_array_cleanup(topo);
}

void print_counter()
{
    printf("Counter: %zu TempCounter: %zu\n\n", node_counter, temp_node_counter-TEMP_COUNTER_LOW_BOUND);
}

void reset_temp_counter()
{
    temp_node_counter = TEMP_COUNTER_LOW_BOUND;
}



//utils for visualisation; move them somewhere else when done

void print_graphviz(Value *v, FILE *f, char *visited, char *temp_visited)
{
    if (v->id < TEMP_COUNTER_LOW_BOUND && visited[v->id])
        return;

    if (v->id >= TEMP_COUNTER_LOW_BOUND && temp_visited[v->id - TEMP_COUNTER_LOW_BOUND])
        return;

    if (v->id < TEMP_COUNTER_LOW_BOUND)
    {
        visited[v->id] = 1;
    }
    else
    {
        temp_visited[v->id - TEMP_COUNTER_LOW_BOUND] = 1;
    }

    fprintf(f, "  node%zu [label=\"%.2f (%s)\"];\n", v->id, v->data, v->op);

    for (size_t i = 0; i < v->num_prev; i++)
    {
        Value *prev_node = v->_prev[i];
        fprintf(f, "  node%zu -> node%zu;\n", prev_node->id, v->id); // directed edge
        print_graphviz(prev_node, f, visited, temp_visited);
    }
}

void generate_graphviz(Value *v)
{
    FILE *f = fopen("graphviz_output.dot", "w");
    if (!f)
    {
        fprintf(stderr, "Error: Unable to open file for writing Graphviz output.\n");
        return;
    }

    fprintf(f, "digraph G {\n");

    char visited[MAX_PARAMS] = {0};
    char temp_visited[MAX_TEMP_PARAMS] = {0};

    print_graphviz(v, f, visited, temp_visited);

    fprintf(f, "}\n");

    fclose(f);
    printf("Graphviz DOT file generated as 'graphviz_output.dot'. Use Graphviz to render the graph.\n");
}
