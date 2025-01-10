#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "matops.h"
#include "gradops.h"
#include "train.h"
#include "memgr.h"

Neuron *create_neuron(size_t num_inputs, int use_relu)
{
    Neuron *n = (Neuron *)malloc(sizeof(Neuron));
    n->num_inputs = num_inputs;
    n->use_relu = use_relu;

    n->weights = (Value **)malloc(num_inputs * sizeof(Value *));
    double stddev = sqrt(2.0 / num_inputs); // This is for ReLU activation
    for (size_t i = 0; i < num_inputs; i++)
    {
        n->weights[i] = create_value(((rand() / (double)RAND_MAX) * 2 - 1) * stddev, 0, NULL, "weight", 0);
    }
    n->bias = create_value(rand() / (double)RAND_MAX, 0, NULL, "bias", 0);

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
    mgr_init();
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
    Value *temp = NULL;

    for (size_t i = 0; i < n->num_inputs; i++)
    {


        Value *mul_out = mul(inputs[i], n->weights[i]);
        if (i > 0)
        {
            temp = add(temp, mul_out);
        }
        else
        {
            temp = mul_out;
        }
    }
    Value *acc_out = add(n->bias, temp);
    Value *activation_result = n->use_relu ? relu(acc_out) : sigmoid(acc_out);
    return activation_result;
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

void zero_gradients(MLP *mlp)
{
    for (size_t i = 0; i < mlp->num_layers; i++)
    {
        for (size_t j = 0; j < mlp->layers[i]->num_neurons; j++)
        {
            Neuron *neuron = mlp->layers[i]->neurons[j];
            neuron->bias->grad = 0;
            for (size_t k = 0; k < neuron->num_inputs; k++)
            {
                neuron->weights[k]->grad = 0;
            }
        }
    }
    // so there is a static part of computation graph i.e. your network weights and biases
    //  and then there is operations you do on them while training/inference
    //  given that operations have to happen every loop of training,
    //  we allow for a temp parameter concept which can be reset so that
    //  there can be reuse of them
    reset_temp_counter();

    // temp cleanup
    mgr_cleanup();
}

void free_mlp(MLP *mlp)
{
    for (size_t i = 0; i < mlp->num_layers; i++)
    {
        Layer *layer = mlp->layers[i];
        for (size_t j = 0; j < layer->num_neurons; j++)
        {
            Neuron *neuron = layer->neurons[j];
            for (size_t k = 0; k < neuron->num_inputs; k++)
            {
                free(neuron->weights[k]->op);
                free(neuron->weights[k]);
            }
            free(neuron->weights);
            free(neuron->bias->op);
            free(neuron->bias);
            free(neuron);
        }
        free(layer->neurons);
        free(layer->outputs);
        free(layer);
    }
    free(mlp->layers);
    free(mlp);
}

void print_weights_and_biases(MLP *mlp)
{
    for (size_t layer_idx = 0; layer_idx < mlp->num_layers; layer_idx++)
    {
        printf("Layer %zu:\n", layer_idx + 1);
        Layer *layer = mlp->layers[layer_idx];
        for (size_t neuron_idx = 0; neuron_idx < layer->num_neurons; neuron_idx++)
        {
            Neuron *neuron = layer->neurons[neuron_idx];
            printf("  Neuron %zu:\n", neuron_idx + 1);
            printf("    Bias: %f\n", neuron->bias->data);
            printf("    Weights:");
            for (size_t weight_idx = 0; weight_idx < neuron->num_inputs; weight_idx++)
            {
                printf(" %f", neuron->weights[weight_idx]->data);
            }
            printf("\n");
        }
    }
}