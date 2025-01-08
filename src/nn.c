#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "matops.h"
#include "gradops.h"

typedef struct Neuron {
    Value** weights;
    Value* bias;
    size_t num_inputs;
    int use_relu;
} Neuron;

typedef struct Layer {
    Neuron** neurons;
    size_t num_neurons;
    size_t num_inputs;
} Layer;

typedef struct MLP {
    Layer** layers;
    size_t num_layers;
} MLP;

Neuron* create_neuron(size_t num_inputs, int use_relu) {
    Neuron* n = (Neuron*)malloc(sizeof(Neuron));
    n->num_inputs = num_inputs;
    n->use_relu = use_relu;
    
    n->weights = (Value**)malloc(num_inputs * sizeof(Value*));
    for (size_t i = 0; i < num_inputs; i++) {
        n->weights[i] = create_value(rand() / (double)RAND_MAX, 0, NULL, "weight");
    }
    n->bias = create_value(0, 0, NULL, "bias");

    return n;
}

Layer* create_layer(size_t num_inputs, size_t num_neurons, int use_relu) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->num_inputs = num_inputs;
    layer->num_neurons = num_neurons;
    layer->neurons = (Neuron**)malloc(num_neurons * sizeof(Neuron*));
    
    for (size_t i = 0; i < num_neurons; i++) {
        layer->neurons[i] = create_neuron(num_inputs, use_relu);
    }
    
    return layer;
}


MLP* create_mlp(size_t* layer_sizes, size_t num_layers) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->num_layers = num_layers;
    mlp->layers = (Layer**)malloc(num_layers * sizeof(Layer*));

    for (size_t i = 0; i < num_layers; i++) {
        int use_relu = (i != num_layers - 1);
        size_t num_inputs = (i == 0) ? layer_sizes[i] : layer_sizes[i - 1];
        mlp->layers[i] = create_layer(num_inputs, layer_sizes[i], use_relu);
    }

    return mlp;
}

Value* forward_neuron(Neuron* n, Value** inputs) {
    Value* output = n->bias;
    for (size_t i = 0; i < n->num_inputs; i++) {
        output = add(output, mul(inputs[i], n->weights[i]));
    }
    return n->use_relu ? relu(output) : output;
}

Value** forward_layer(Layer* layer, Value** inputs) {
    Value** outputs = (Value**)malloc(layer->num_neurons * sizeof(Value*));
    for (size_t i = 0; i < layer->num_neurons; i++) {
        outputs[i] = forward_neuron(layer->neurons[i], inputs);
    }
    return outputs;
}

Value** forward_mlp(MLP* mlp, Value** inputs) {
    Value** outputs = inputs;
    for (size_t i = 0; i < mlp->num_layers; i++) {
        outputs = forward_layer(mlp->layers[i], outputs);
    }
    return outputs;
}



