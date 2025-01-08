#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

void print_full_network_state(MLP* mlp, Value** inputs, Value** outputs, size_t num_inputs, size_t num_outputs) {
    printf("\nInputs with Gradients:\n");
    for (size_t i = 0; i < num_inputs; i++) {
        printf("Input %zu: Value = %f, Gradient = %f\n", i, inputs[i]->data, inputs[i]->grad);
    }

    printf("\nWeights, Biases, and Gradients:\n");
    for (size_t layer_idx = 0; layer_idx < mlp->num_layers; layer_idx++) {
        Layer* layer = mlp->layers[layer_idx];
        printf("Layer %zu:\n", layer_idx);

        for (size_t neuron_idx = 0; neuron_idx < layer->num_neurons; neuron_idx++) {
            Neuron* neuron = layer->neurons[neuron_idx];
            printf("  Neuron %zu:\n", neuron_idx);

            for (size_t weight_idx = 0; weight_idx < neuron->num_inputs; weight_idx++) {
                printf("    Weight %zu: Value = %f, Gradient = %f\n",
                       weight_idx, neuron->weights[weight_idx]->data,
                       neuron->weights[weight_idx]->grad);
            }

            printf("    Bias: Value = %f, Gradient = %f\n",
                   neuron->bias->data, neuron->bias->grad);
        }
    }

    printf("\nOutputs with Gradients:\n");
    for (size_t i = 0; i < num_outputs; i++) {
        printf("Output %zu: Value = %f, Gradient = %f\n", i, outputs[i]->data, outputs[i]->grad);
    }
}

int main() {
    size_t layer_sizes[] = {3, 5, 2};
    MLP* mlp = create_mlp(layer_sizes, 3);

    Value* inputs[3] = {
        create_value(1.0, 0, NULL, "input"),
        create_value(2.0, 0, NULL, "input"),
        create_value(3.0, 0, NULL, "input")
    };

    Value** outputs = forward_mlp(mlp, inputs);

    Value* loss = add(outputs[0], outputs[1]);

    backward(loss);

    print_full_network_state(mlp, inputs, outputs, 3, 2);

    free(mlp);
    return 0;
}
