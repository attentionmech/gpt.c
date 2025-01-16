#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_SLOTS 100

typedef enum
{
    ADD,
    MULTIPLY,
    SIGMOID,
    MSE,
    PARAMETER
} OperationType;

typedef struct
{
    double value;
    double gradient;
    OperationType operation;
    int dependencies[2];
    int num_dependencies;
    int learnable_param;
} Slot;

Slot slots[MAX_SLOTS];
int slot_count = 0;

double random_init()
{
    return (rand() / (double)RAND_MAX) * 2.0 - 1.0;
}

int increment_slot()
{

    if (slot_count >= MAX_SLOTS)
    {
        fprintf(stderr, "Error: Exceeded maximum number of slots (%d)\n", MAX_SLOTS);
        exit(EXIT_FAILURE);
    }

    return slot_count++;
}

int create_value_slot(double value, int learnable_param)
{
    slots[slot_count].value = value;
    slots[slot_count].gradient = 0.0;
    slots[slot_count].operation = PARAMETER;
    slots[slot_count].num_dependencies = 0;
    slots[slot_count].learnable_param = learnable_param;

    if (learnable_param == 1)
    {
        slots[slot_count].value = random_init();
    }

    return increment_slot();
}

int create_operation_slot(OperationType op, int dep1, int dep2)
{
    slots[slot_count].operation = op;
    slots[slot_count].dependencies[0] = dep1;
    slots[slot_count].dependencies[1] = dep2;
    slots[slot_count].num_dependencies = 2;
    slots[slot_count].gradient = 0.0;
    slots[slot_count].learnable_param = 0;
    return increment_slot();
}

void set_value_slot(int slot, double value)
{
    slots[slot].value = value;
}

double get_value_slot(int slot)
{
    return slots[slot].value;
}

double compute_graph(int slot)
{
    Slot *s = &slots[slot];
    if (s->num_dependencies > 0)
    {
        double dep1_value = (s->num_dependencies > 0) ? compute_graph(s->dependencies[0]) : 0.0;
        double dep2_value = (s->num_dependencies > 1) ? compute_graph(s->dependencies[1]) : 0.0;

        switch (s->operation)
        {
        case ADD:
            s->value = dep1_value + dep2_value;
            break;
        case MULTIPLY:
            s->value = dep1_value * dep2_value;
            break;
        case SIGMOID:
            s->value = 1.0 / (1.0 + exp(-dep1_value));
            break;
        case MSE:
            s->value = (dep1_value - dep2_value) * (dep1_value - dep2_value);
            break;
        default:
            break;
        }
    }
    return s->value;
}

void compute_grad(int slot)
{
    Slot *s = &slots[slot];
    if (s->num_dependencies > 0)
    {
        double dep1_value = (s->num_dependencies > 0) ? get_value_slot(s->dependencies[0]) : 0.0;
        double dep2_value = (s->num_dependencies > 1) ? get_value_slot(s->dependencies[1]) : 0.0;

        switch (s->operation)
        {
        case ADD:
            slots[s->dependencies[0]].gradient += s->gradient;
            slots[s->dependencies[1]].gradient += s->gradient;
            break;
        case MULTIPLY:
            slots[s->dependencies[0]].gradient += s->gradient * dep2_value;
            slots[s->dependencies[1]].gradient += s->gradient * dep1_value;
            break;
        case SIGMOID:
        {
            double sigmoid_derivative = s->value * (1.0 - s->value);
            slots[s->dependencies[0]].gradient += s->gradient * sigmoid_derivative;
        }
        break;
        case MSE:
            slots[s->dependencies[0]].gradient += 2.0 * s->gradient * (dep1_value - dep2_value);
            break;
        default:
            break;
        }

        for (int i = 0; i < s->num_dependencies; i++)
        {
            compute_grad(s->dependencies[i]);
        }
    }
}

double calculate_loss(double predictions[], int labels[], int num_samples)
{
    double loss = 0.0;
    for (int i = 0; i < num_samples; i++)
    {
        loss += predictions[i];
    }
    return loss / num_samples;
}

int create_feedforward_network(int *layer_sizes, int num_layers)
{
    int *prev_layer_slots = NULL;
    int *curr_layer_slots = NULL;
    int curr_slot = 0;

    for (int layer = 0; layer < num_layers; layer++)
    {
        curr_layer_slots = malloc(layer_sizes[layer] * sizeof(int));

        if (layer == 0)
        {
            for (int i = 0; i < layer_sizes[layer]; i++)
            {
                curr_layer_slots[i] = create_value_slot(0.0, 0);
            }
        }
        else
        {
            for (int neuron = 0; neuron < layer_sizes[layer]; neuron++)
            {
                int sum = -1;
                for (int prev = 0; prev < layer_sizes[layer - 1]; prev++)
                {
                    int weight = create_value_slot(0.1, 1);
                    int mul = create_operation_slot(MULTIPLY, prev_layer_slots[prev], weight);

                    if (sum == -1)
                    {
                        sum = mul;
                    }
                    else
                    {
                        sum = create_operation_slot(ADD, sum, mul);
                    }
                }

                int bias = create_value_slot(0.0, 1);
                int biased = create_operation_slot(ADD, sum, bias);

                curr_layer_slots[neuron] = create_operation_slot(SIGMOID, biased, 0);
            }
        }

        free(prev_layer_slots);
        prev_layer_slots = curr_layer_slots;
    }

    curr_slot = curr_layer_slots[0];

    free(curr_layer_slots);

    return curr_slot;
}

void train(double inputs[][2], int labels[], int num_samples, double learning_rate)
{
    int layer_sizes[] = {2, 2, 1};
    int num_layers = 3;

    int final_output = create_feedforward_network(layer_sizes, num_layers);
    // the input neurons are first created, hence 0,1 etc can be used from there

    int target_slot = create_value_slot(0.0, 0);
    int loss_slot = create_operation_slot(MSE, final_output, target_slot);

    srand(time(NULL));

    for (int epoch = 0; epoch < 1000000; epoch++)
    {
        double total_loss = 0.0;

        for (int i = 0; i < num_samples; i++)
        {

            for (int j = 0; j < slot_count; j++)
            {
                slots[j].gradient = 0.0;
            }

            set_value_slot(0, inputs[i][0]);
            set_value_slot(1, inputs[i][1]);
            set_value_slot(target_slot, labels[i]);

            compute_graph(loss_slot);
            total_loss += get_value_slot(loss_slot);

            slots[loss_slot].gradient = 1.0;
            compute_grad(loss_slot);

            for (int j = 0; j < slot_count; j++)
            {
                if (slots[j].learnable_param == 1)
                {
                    set_value_slot(j, get_value_slot(j) - learning_rate * slots[j].gradient);
                }
            }
        }

        if (epoch % 100 == 0)
        {
            printf("Epoch %d Average Loss: %f\n", epoch + 1, total_loss / num_samples);
        }
    }

    printf("\nFinal Results:\n");
    for (int i = 0; i < num_samples; i++)
    {
        set_value_slot(0, inputs[i][0]);
        set_value_slot(1, inputs[i][1]);
        compute_graph(final_output);
        printf("Input: (%.1f, %.1f) => Target: %d, Predicted: %.4f\n",
               inputs[i][0], inputs[i][1], labels[i], get_value_slot(final_output));
    }
}

int main()
{
    double inputs[4][2] = {
        {0.0, 0.0},
        {0.0, 1.0},
        {1.0, 0.0},
        {1.0, 1.0}};
    int labels[4] = {0, 1, 1, 0};

    double learning_rate = 0.001;

    train(inputs, labels, 4, learning_rate);

    return 0;
}
