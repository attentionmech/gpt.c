#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "matops.h"
#include "gradops.h"

typedef struct Neuron
{
    Value **weights;
    Value *bias;
    size_t num_inputs;
    int use_relu;
    Value *add_out;
    Value *mul_out;
} Neuron;

typedef struct Layer
{
    Neuron **neurons;
    size_t num_neurons;
    size_t num_inputs;
    Value **outputs;

} Layer;

typedef struct MLP
{
    Layer **layers;
    size_t num_layers;
} MLP;

Neuron *create_neuron(size_t num_inputs, int use_relu)
{
    Neuron *n = (Neuron *)malloc(sizeof(Neuron));
    n->num_inputs = num_inputs;
    n->use_relu = use_relu;

    n->weights = (Value **)malloc(num_inputs * sizeof(Value *));
    for (size_t i = 0; i < num_inputs; i++)
    {
        n->weights[i] = create_value(rand() / (double)RAND_MAX, 0, NULL, "weight");
    }
    n->bias = create_value(0, 0, NULL, "bias");
    n->add_out = create_value(0, 0, NULL, "add_out");
    n->mul_out = create_value(0, 0, NULL, "mul_out");

    return n;
}

Layer *create_layer(size_t num_inputs, size_t num_neurons, int use_relu)
{
    Layer *layer = (Layer *)malloc(sizeof(Layer));
    layer->num_inputs = num_inputs;
    layer->num_neurons = num_neurons;
    layer->neurons = (Neuron **)malloc(num_neurons * sizeof(Neuron *));
    layer->outputs = (Value **)malloc(num_neurons * sizeof(Value *));

    for (size_t i = 0; i < num_neurons; i++)
    {
        layer->neurons[i] = create_neuron(num_inputs, use_relu);
    }

    return layer;
}

MLP *create_mlp(size_t *layer_sizes, size_t num_layers)
{
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    mlp->num_layers = num_layers;
    mlp->layers = (Layer **)malloc(num_layers * sizeof(Layer *));

    for (size_t i = 0; i < num_layers; i++)
    {
        int use_relu = (i != num_layers - 1);
        size_t num_inputs = (i == 0) ? layer_sizes[i] : layer_sizes[i - 1];
        mlp->layers[i] = create_layer(num_inputs, layer_sizes[i], use_relu);
    }

    return mlp;
}

Value *forward_neuron(Neuron *n, Value **inputs)
{
    Value *output = n->bias;
    Value *add_out = n->add_out;
    Value *mul_out = n->mul_out;
    for (size_t i = 0; i < n->num_inputs; i++)
    {
        output = add(output, mul(inputs[i], n->weights[i], mul_out, 1), add_out, 1);
    }
    return n->use_relu ? relu(output) : output;
}

Value **forward_layer(Layer *layer, Value **inputs)
{
    for (size_t i = 0; i < layer->num_neurons; i++)
    {
        layer->outputs[i] = forward_neuron(layer->neurons[i], inputs);
    }
    return layer->outputs;
}

Value **forward_mlp(MLP *mlp, Value **inputs)
{
    Value **outputs = inputs;
    for (size_t i = 0; i < mlp->num_layers; i++)
    {
        outputs = forward_layer(mlp->layers[i], outputs);
    }
    return outputs;
}
