#ifndef TRAIN_H
#define TRAIN_H

#include "nn.h"

void update_weights(MLP *mlp, float learning_rate);
void update_weights_with_momentum(MLP *mlp, float learning_rate, float momentum);


#endif