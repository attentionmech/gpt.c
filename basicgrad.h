#ifndef BASICGRAD_H
#define BASICGRAD_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>


#define MAX_SLOTS 10000000

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
    int *shape;             
    int *strides;           
    int num_dimensions;
    int size;     
    OperationType operation; 
    int *dependencies;      
    int num_dependencies;   
    int learnable_param;    
    int visited;
} Slot;


Slot slots[MAX_SLOTS];


int create_value_slot(int learnable_param, int *shape, int num_dimensions);
int create_operation_slot(OperationType op, int *dep, int num_dependencies, int *shape, int num_dimensions);

double get_slot_value(int slot, int b_index);
void set_slot_value(int slot, int b_index, double v);

void compute_grad(int slot);
double *compute_graph(int slot);

int zerograd();

#endif