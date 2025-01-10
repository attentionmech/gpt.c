#include "nn.h"

void update_weights(MLP *mlp, float learning_rate)
{
    for (size_t layer_idx = 0; layer_idx < mlp->num_layers; layer_idx++)
    {
        Layer *layer = mlp->layers[layer_idx];
        for (size_t neuron_idx = 0; neuron_idx < layer->num_neurons; neuron_idx++)
        {
            Neuron *neuron = layer->neurons[neuron_idx];
            for (size_t weight_idx = 0; weight_idx < neuron->num_inputs; weight_idx++)
            {
                neuron->weights[weight_idx]->data -= learning_rate * neuron->weights[weight_idx]->grad;
            }
            neuron->bias->data -= learning_rate * neuron->bias->grad;
        }
    }
}


void update_weights_with_momentum(MLP *mlp, float learning_rate, float momentum)
{
    for (size_t layer_idx = 0; layer_idx < mlp->num_layers; layer_idx++)
    {
        Layer *layer = mlp->layers[layer_idx];
        for (size_t neuron_idx = 0; neuron_idx < layer->num_neurons; neuron_idx++)
        {
            Neuron *neuron = layer->neurons[neuron_idx];
            for (size_t weight_idx = 0; weight_idx < neuron->num_inputs; weight_idx++)
            {
                // Store previous weight update
                float weight_update = -learning_rate * neuron->weights[weight_idx]->grad;
                neuron->weights[weight_idx]->data += momentum * neuron->weights[weight_idx]->prev_update + weight_update;
                neuron->weights[weight_idx]->prev_update = weight_update;
            }
            // Update bias with momentum
            float bias_update = -learning_rate * neuron->bias->grad;
            neuron->bias->data += momentum * neuron->bias->prev_update + bias_update;
            neuron->bias->prev_update = bias_update;
        }
    }
}
