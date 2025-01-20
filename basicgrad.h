#ifndef BASICGRAD_H
#define BASICGRAD_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>


#define MAX_SLOTS 10000000
#define BATCH_SIZE 100
#define MAX_DEPENDENCY 1000

typedef enum
{
    ADD,
    MULTIPLY,
    SUB,
    POW2, // square
    NEG,
    EXP,
    DIV,
    LOG,

    SIGMOID,
    RELU,
    GELU,
    LEAKY_RELU,

    PARAMETER, // nodes which are just values (leaf nodes)
} OperationType;

typedef struct
{
    double *value;
    double *gradient;
    OperationType operation;
    int *dependencies;
    int num_dependencies;
    int learnable_param;
    int visited;
} Slot;

Slot slots[MAX_SLOTS];


double get_slot_value(int slot, int b_index);
void set_slot_value(int slot, int b_index, double v);
double *compute_graph(int slot);
int *create_softmax_layer(int *input_slots, int num_outputs);
int *create_feedforward_network(int *layer_sizes, int num_layers);
int create_value_slot(int learnable_param);
int create_cross_entropy_loss(int *target_slots, int *softmax_slots, int num_outputs);
int zerograd();
void compute_grad(int slot);

#endif