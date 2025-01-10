#include <stdio.h>
#include <stdlib.h>
#include "nn.h"
#include "train.h"

int main()
{

    Value *xor_inputs[4][2] = {
        {create_value(0.0, 0, NULL, "input", 0), create_value(0.0, 0, NULL, "input", 0)},
        {create_value(0.0, 0, NULL, "input", 0), create_value(1.0, 0, NULL, "input", 0)},
        {create_value(1.0, 0, NULL, "input", 0), create_value(0.0, 0, NULL, "input", 0)},
        {create_value(1.0, 0, NULL, "input", 0), create_value(1.0, 0, NULL, "input", 0)}};

    Value *xor_targets[4] = {
        create_value(0.0, 0, NULL, "target", 0),
        create_value(1.0, 0, NULL, "target", 0),
        create_value(1.0, 0, NULL, "target", 0),
        create_value(0.0, 0, NULL, "target", 0)};

    size_t layer_sizes[] = {2, 32, 1};
    MLP *mlp = create_mlp(layer_sizes, 3);

    size_t epochs = 100000;
    float learning_rate = 0.001;

    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        float total_loss = 0.0;
        zero_gradients(mlp);

        for (size_t i = 0; i < 4; i++)
        {
            Value **inputs = xor_inputs[i];
            Value *target = xor_targets[i];
            Value **outputs = forward_mlp(mlp, inputs);
            Value *loss = power(sub(outputs[0], target), 2);
            total_loss += loss->data;
            backward(loss);
        }
        update_weights_with_momentum(mlp, learning_rate,0.1);
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

    free_mlp(mlp);
    return 0;
}

