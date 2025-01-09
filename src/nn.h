#include "gradops.h"

typedef struct Neuron
{
    Value **weights;
    Value *bias;
    size_t num_inputs;
    int use_relu;
} Neuron;

Neuron *create_neuron(size_t num_inputs, int use_relu);
Value *forward_neuron(Neuron *n, Value **inputs);

typedef struct Layer
{
    Neuron **neurons;
    size_t num_neurons;
    size_t num_inputs;
} Layer;

Layer *create_layer(size_t num_inputs, size_t num_neurons, int use_relu);
Value **forward_layer(Layer *layer, Value **inputs);

typedef struct MLP
{
    Layer **layers;
    size_t num_layers;
} MLP;

MLP *create_mlp(size_t *layer_sizes, size_t num_layers);
Value **forward_mlp(MLP *mlp, Value **inputs);

void zero_gradients(MLP *mlp);
void free_mlp(MLP *mlp);

void print_weights_and_biases(MLP *mlp);
void reset_temp_counter();