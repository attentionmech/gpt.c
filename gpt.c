#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>


#define MAX_SLOTS 1000000
#define IMAGE_SIZE 784

typedef enum
{
    ADD,
    MULTIPLY,
    SUB,
    POW2,
    SIGMOID,
    RELU,
    PARAMETER,
} OperationType;

typedef struct
{
    double value;
    double gradient;
    OperationType operation;
    int *dependencies;
    int num_dependencies;
    int learnable_param;
    int visited;
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

int create_operation_slot(OperationType op, int* dep, int num_dependencies)
{
    slots[slot_count].operation = op;
    slots[slot_count].dependencies = dep;
    slots[slot_count].num_dependencies = num_dependencies;

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


double _sum(double *list, int length)
{
    double total = 0.0;
    for (int i = 0; i < length; i++)
    {
        total += list[i];
    }
    return total;
}

double _mul(double *list, int length)
{
    double total = 1.0;
    for (int i = 0; i < length; i++)
    {
        total *= list[i];
    }
    return total;
}

double compute_graph(int slot)
{

    Slot *s = &slots[slot];

    if (s->visited){
        return s->value;
    }

    if (s->num_dependencies > 0)
    {
        double dep_value[s->num_dependencies];
        for (int j=0; j< s->num_dependencies; j++){
            dep_value[j] = compute_graph(s->dependencies[j]);
        }


        switch (s->operation)
        {
        case ADD:
            s->value = _sum(dep_value, s->num_dependencies);
            break;
        case MULTIPLY:
            s->value = _mul(dep_value, s->num_dependencies);
            break;


         case SUB:
            s->value = dep_value[0] - dep_value[1];  // a - b
            break;
         case POW2:
            s->value = dep_value[0] * dep_value[0];  // a^2
            break;
        case SIGMOID:
            s->value = 1.0 / (1.0 + exp(-dep_value[0]));
            break;
        case RELU:
            s->value = fmax(0.0, dep_value[0]);
            break;
        default:
            break;
        }
    }

    s->visited = 1;

    return s->value;
}

void compute_grad(int slot)
{
    printf("Computing gradient for slot %d with operation %d\n", slot, slots[slot].operation);
    Slot *s = &slots[slot];

    if (s->num_dependencies > 0)
    {
        double dep_value[s->num_dependencies];
        for (int j = 0; j < s->num_dependencies; j++)
        {
            dep_value[j] = get_value_slot(s->dependencies[j]);
        }

        switch (s->operation)
        {
        case ADD:
            for (int i = 0; i < s->num_dependencies; i++)
            {
                slots[s->dependencies[i]].gradient += s->gradient;
            }
            break;

        case MULTIPLY:
            for (int i = 0; i < s->num_dependencies; i++)
            {
                double product = 1.0;
                for (int j = 0; j < s->num_dependencies; j++)
                {
                    if (i != j)
                        product *= dep_value[j];
                }
                slots[s->dependencies[i]].gradient += s->gradient * product;
            }
            break;
        
         case SUB:
            slots[s->dependencies[0]].gradient += s->gradient;      
            slots[s->dependencies[1]].gradient -= s->gradient; 
            break;
        
         case POW2:
            slots[s->dependencies[0]].gradient += s->gradient * 2.0 * dep_value[0];  
            break;


        case SIGMOID:
            slots[s->dependencies[0]].gradient += s->gradient * s->value * (1.0 - s->value);
            break;

        case RELU:
            if (dep_value[0] > 0)
            {
                slots[s->dependencies[0]].gradient += s->gradient;
            }
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


int* create_feedforward_network(int *layer_sizes, int num_layers)
{
    int *prev_layer_slots = NULL;
    int *curr_layer_slots = NULL;
    int curr_slot = 0;

    for (int layer = 0; layer < num_layers; layer++)
    {
        curr_layer_slots = malloc(layer_sizes[layer] * sizeof(int));

        if (layer == 0)
        {
            // Input layer (no batch norm required here)
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
                    int mul = create_operation_slot(MULTIPLY, (int[]){prev_layer_slots[prev], weight}, 2);

                    if (sum == -1)
                    {
                        sum = mul;
                    }
                    else
                    {
                        sum = create_operation_slot(ADD, (int[]){sum, mul}, 2);
                    }
                }

                int bias = create_value_slot(0.0, 1);
                int biased = create_operation_slot(ADD, (int[]){sum, bias},2);

                curr_layer_slots[neuron] = create_operation_slot(RELU, (int[]){biased}, 1);
            }
        }

        free(prev_layer_slots);
        prev_layer_slots = curr_layer_slots;
    }

    return curr_layer_slots;
}


void train(double inputs[][IMAGE_SIZE], int labels[], int num_samples, double learning_rate)

{

    int num_inputs = IMAGE_SIZE;
    int num_outputs = 10;


    int layer_sizes[] = {num_inputs,  num_outputs}; // 784 inputs, 128 hidden, 10 output neurons
    int num_layers = 2;

    int* output_slots = create_feedforward_network(layer_sizes, num_layers);
    int target_slots[num_outputs];
    int diff_slots[num_outputs];

    for(int i=0; i< 10; i++){
        target_slots[i] = create_value_slot(0.0, 0);
        int temp = create_operation_slot(SUB, (int[]){target_slots[i], output_slots[i]},2);
        diff_slots[i] = create_operation_slot(POW2, (int[]){temp}, 1);
    }

    int loss_slot = create_operation_slot(ADD, diff_slots,num_outputs);

    srand(time(NULL));


    for (int epoch = 0; epoch < 1000000; epoch++) // Reduced number of epochs
    {
        double total_loss = 0.0;

        for (int i = 0; i < num_samples; i++)
        {
            for (int j = 0; j < slot_count; j++)
            {
                slots[j].gradient = 0.0;
                slots[j].visited = 0;
            }

            // set sample
            for (int k = 0; k < num_inputs; k++)
            {
                set_value_slot(k, inputs[i][k]);
            }

            // one hot
            for (int l =1; l <= num_outputs; l++){
                if (l == labels[i]){
                    set_value_slot(target_slots[l-1], 1);
                } else {
                    set_value_slot(target_slots[l-1], 0);
                }
            }

            compute_graph(loss_slot);

            
            total_loss += get_value_slot(loss_slot);


            slots[loss_slot].gradient = 1.0;
            // compute_grad(loss_slot);

            for (int j = 0; j < slot_count; j++)
            {
                if (slots[j].learnable_param == 1)
                {
                    set_value_slot(j, get_value_slot(j) - learning_rate * slots[j].gradient);
                }
            }
        }

        printf("Epoch %d, Loss: %f\n", epoch, total_loss / num_samples);
    }
}

int main()
{
    FILE *file = fopen("dataset/mnist.txt", "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    int num_samples = 100;
    double inputs[num_samples][IMAGE_SIZE];
    int labels[num_samples];

    
    int i=0;


     while (fscanf(file, "%d", &labels[i]) != EOF && i < num_samples) {
        
        for (int j = 0; j < IMAGE_SIZE; j++) {

            if (fscanf(file, "%lf", &inputs[i][j]) != 1) {
                break;
            }
        }
        i++;
        
    }

    fclose(file);

    double learning_rate = 0.001;

    train(inputs, labels, num_samples, learning_rate);

    return 0;
}
