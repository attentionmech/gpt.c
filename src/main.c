#include <stdio.h>
#include <stdlib.h>
#include "nn.h"

int main() {
    size_t layer_sizes[] = {3, 5, 2};
    MLP* mlp = create_mlp(layer_sizes, 3);

    Value* inputs[3] = {create_value(1.0, 0, NULL, "input"), create_value(2.0, 0, NULL, "input"), create_value(3.0, 0, NULL, "input")};

    Value** outputs = forward_mlp(mlp, inputs);

    for (size_t i = 0; i < 2; i++) {
        printf("Output %zu: %f\n", i, outputs[i]->data);
    }

    free(mlp);
    return 0;
}