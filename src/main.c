#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

int main()
{

    Value *xor_inputs[4][2] = {
        {create_value(0.0, 0, NULL, "input"), create_value(0.0, 0, NULL, "input")},
        {create_value(0.0, 0, NULL, "input"), create_value(1.0, 0, NULL, "input")},
        {create_value(1.0, 0, NULL, "input"), create_value(0.0, 0, NULL, "input")},
        {create_value(1.0, 0, NULL, "input"), create_value(1.0, 0, NULL, "input")}};

    Value *xor_targets[4] = {
        create_value(0.0, 0, NULL, "target"),
        create_value(1.0, 0, NULL, "target"),
        create_value(1.0, 0, NULL, "target"),
        create_value(0.0, 0, NULL, "target")};

    size_t layer_sizes[] = {2, 3, 1};
    MLP *mlp = create_mlp(layer_sizes, 3);

    size_t epochs = 10;
    float learning_rate = 0.001;

    Value* sub_out = create_value(0,0,NULL,"sub_out");
    Value* power_out = create_value(0,0,NULL,"power_out");

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        float total_loss = 0.0;

        for (size_t i = 0; i < 4; i++)
        {

            Value **inputs = xor_inputs[i];
            Value *target = xor_targets[i];

            Value **outputs = forward_mlp(mlp, inputs);

            printf("output: %f ", outputs[0]->data);
            
            Value *loss = power(sub(outputs[0], target,sub_out, 1), 2, power_out, 1);

            total_loss += loss->data;
            backward(loss);

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

        printf("Epoch %zu: Loss = %f\n", epoch + 1, total_loss / 4);
    }

    printf("\nTesting the trained network:\n");
    for (size_t i = 0; i < 4; i++)
    {
        Value **inputs = xor_inputs[i];
        Value *target = xor_targets[i];
        Value **outputs = forward_mlp(mlp, inputs);
        printf("Input: (%f, %f) -> Output: %f (Expected: %f)\n",
               inputs[0]->data, inputs[1]->data, outputs[0]->data, target->data);
    }

    free(mlp);
    return 0;
}
