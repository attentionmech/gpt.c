#include "basicgrad.h"

int topo[MAX_SLOTS];
double *dependency_buffer[BATCH_SIZE];
int slot_counter = 0;

const char *get_operation_name(OperationType op)
{
    switch (op)
    {
    case ADD:
        return "ADD";
    case MULTIPLY:
        return "MULTIPLY";
    case SUB:
        return "SUB";
    case POW2:
        return "POW2";
    case SIGMOID:
        return "SIGMOID";
    case RELU:
        return "RELU";
    case PARAMETER:
        return "PARAM";
    case EXP:
        return "EXP";
    case DIV:
        return "DIV";
    case LOG:
        return "LOG";
    case NEG:
        return "NEG";
    case LEAKY_RELU:
        return "LEAKY_RELU";
    case GELU:
        return "GELU";
    default:
        return "UNKNOWN";
    }
}

void export_graph_to_dot(const char *filename)
{
    FILE *file = fopen(filename, "w");
    if (!file)
    {
        fprintf(stderr, "Error opening file for writing Graphviz DOT file.\n");
        return;
    }

    fprintf(file, "digraph ComputationalGraph {\n");
    fprintf(file, "    node [shape=circle, style=filled, fillcolor=lightblue];\n");

    for (int i = 0; i < slot_counter; i++)
    {
        Slot *s = &slots[i];
        // Node definition with value, gradient, and operation type
        fprintf(file, "    slot_%d [label=\"%d\\nVal: %.2f\\nGrad: %.2f\\nOp: %s\"];\n",
                i, i, s->value[0], s->gradient[0], get_operation_name(s->operation));

        // Edge definitions based on dependencies
        for (int j = 0; j < s->num_dependencies; j++)
        {
            fprintf(file, "    slot_%d -> slot_%d;\n", s->dependencies[j], i);
        }
    }

    fprintf(file, "}\n");
    fclose(file);
    printf("Graph exported to %s. Use 'dot -Tpng %s -o graph.png' to generate an image.\n", filename, filename);
}

// --------------------------------------------

double random_init()
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
    return z0;
}

int increment_slot()
{
    if (slot_counter >= MAX_SLOTS)
    {
        fprintf(stderr, "Error: Exceeded maximum number of slots (%d)\n", MAX_SLOTS);
        exit(EXIT_FAILURE);
    }
    return slot_counter++;
}

int create_value_slot(int learnable_param)
{
    slots[slot_counter].value = (double *)malloc(BATCH_SIZE * sizeof(double));
    slots[slot_counter].gradient =
        (double *)malloc(BATCH_SIZE * sizeof(double));
    slots[slot_counter].operation = PARAMETER;
    slots[slot_counter].num_dependencies = 0;
    slots[slot_counter].learnable_param = learnable_param;

    return increment_slot();
}

int create_operation_slot(OperationType op, int *dep, int num_dependencies)
{
    slots[slot_counter].operation = op;
    slots[slot_counter].dependencies = dep;
    slots[slot_counter].num_dependencies = num_dependencies;
    slots[slot_counter].value = (double *)malloc(BATCH_SIZE * sizeof(double));
    slots[slot_counter].gradient = (double *)malloc(BATCH_SIZE * sizeof(double));
    slots[slot_counter].learnable_param = 0;
    return increment_slot();
}

void set_slot_value(int slot, int b_index, double v)
{
    slots[slot].value[b_index] = v;
}

double get_slot_value(int slot, int b_index)
{
    if (b_index >= BATCH_SIZE)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }
    return slots[slot].value[b_index];
}

double _sum(double **list, int b, int length)
{
    double total = 0;
    for (int i = 0; i < length; i++)
    {
        total += list[i][b];
    }
    return total;
}

double _mul(double **list, int b, int length)
{
    double product;

    product = 1.0;
    for (int i = 0; i < length; i++)
    {
        product *= list[i][b];
    }

    return product;
}

double *compute_graph(int slot)
{
    Slot *s = &slots[slot];

    if (s->visited)
    {
        return s->value;
    }

    if (s->num_dependencies > 0)
    {
        double *dep_value[s->num_dependencies];
        for (int j = 0; j < s->num_dependencies; j++)
        {
            dep_value[j] = compute_graph(s->dependencies[j]);
        }

        switch (s->operation)
        {
        case ADD:
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                s->value[b] = _sum(dep_value, b, s->num_dependencies);
            }
            break;
        case MULTIPLY:
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                s->value[b] = _mul(dep_value, b, s->num_dependencies);
            }
            break;

        case EXP:

            for (int b = 0; b < BATCH_SIZE; b++)
            {
                s->value[b] = exp(dep_value[0][b]);
            }
            break;
        case DIV:
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                s->value[b] = dep_value[0][b] / dep_value[1][b];
            }

            break;

        case NEG:
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                s->value[b] = -dep_value[0][b];
            }
            break;

        case LOG:
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                s->value[b] = log(dep_value[0][b]);
            }
            break;

        case SUB:
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                s->value[b] = dep_value[0][b] - dep_value[0][b];
            }
            break;
        case POW2:
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                s->value[b] = dep_value[0][b] * dep_value[0][b];
            }
            break;
        case SIGMOID:
            for (int b = 0; b < BATCH_SIZE; b++)
            {

                s->value[b] = 1.0 / (1.0 + exp(-dep_value[0][b]));
            }
            break;
        case RELU:
            for (int b = 0; b < BATCH_SIZE; b++)
            {

                s->value[b] = fmax(0.0, dep_value[0][b]);
            }
            break;
        case LEAKY_RELU:
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                double alpha = 0.01;
                s->value[b] = (dep_value[0][b] > 0) ? dep_value[0][b] : alpha * dep_value[0][b];
            }
            break;

        case GELU:
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                double x = dep_value[0][b];
                s->value[b] = 0.5 * x * (1.0 + tanh(sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x)));
            }
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

    for (int b = 0; b < BATCH_SIZE; b++)
    {
        if (dependency_buffer[b] == NULL)
        {
            dependency_buffer[b] = (double *)malloc(MAX_DEPENDENCY * sizeof(double));
        }
    }

    for (int curr = slot; curr >= 0; curr--)
    {
        Slot *s = &slots[curr];

        if (s->num_dependencies > 0)
        {

            for (int j = 0; j < s->num_dependencies; j++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    dependency_buffer[b][j] = get_slot_value(s->dependencies[j], b);
                }
            }

            switch (s->operation)
            {
            case ADD:
                for (int i = 0; i < s->num_dependencies; i++)
                {
                    for (int b = 0; b < BATCH_SIZE; b++)
                    {
                        slots[s->dependencies[i]].gradient[b] += s->gradient[b];
                    }
                }
                break;

            case MULTIPLY:
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    double product = 1.0;
                    for (int j = 0; j < s->num_dependencies; j++)
                    {
                        product *= dependency_buffer[b][j];
                    }
                    for (int i = 0; i < s->num_dependencies; i++)
                    {
                        slots[s->dependencies[i]].gradient[b] += s->gradient[b] * (product / dependency_buffer[b][i]);
                    }
                }
                break;

            case SUB:
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b];
                    slots[s->dependencies[1]].gradient[b] -= s->gradient[b];
                }
                break;

            case POW2:
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * 2.0 * dependency_buffer[b][0];
                }
                break;

            case SIGMOID:
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * s->value[b] * (1.0 - s->value[b]);
                }
                break;

            case RELU:
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    if (dependency_buffer[b][0] > 0)
                    {
                        slots[s->dependencies[0]].gradient[b] += s->gradient[b];
                    }
                }
                break;

            case LEAKY_RELU:
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    double alpha = 0.01;
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] *
                                                             (dependency_buffer[b][0] > 0 ? 1.0 : alpha);
                }
                break;

            case GELU:
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    double x = dependency_buffer[b][0];
                    double tanh_arg = sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
                    double tanh_val = tanh(tanh_arg);
                    double derivative = 0.5 * (1 + tanh_val + x * (1 - tanh_val * tanh_val) * sqrt(2.0 / M_PI) * (1 + 3 * 0.044715 * x * x));
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * derivative;
                }
                break;

            case EXP:
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * s->value[b];
                }
                break;

            case NEG:
                for (int b = 0; b < BATCH_SIZE; b++)
                {

                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * -1.0;
                }
                break;

            case DIV:
                for (int b = 0; b < BATCH_SIZE; b++)
                {

                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] / dependency_buffer[b][1];
                    slots[s->dependencies[1]].gradient[b] -= s->gradient[b] * dependency_buffer[b][0] / (dependency_buffer[b][1] * dependency_buffer[b][1]);
                }
                break;

            case LOG:
                for (int b = 0; b < BATCH_SIZE; b++)
                {

                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * (1.0 / dependency_buffer[b][0]);
                }
                break;

            default:
                break;
            }
        }
    }
}

int *wrap_value_in_array(int a)
{
    int *arr = malloc(1 * sizeof(int));
    arr[0] = a;
    return arr;
}

int *wrap_in_array(int a, int b)
{
    int *arr = malloc(2 * sizeof(int));
    arr[0] = a;
    arr[1] = b;
    return arr;
}

double xavier_init(int fan_in, int fan_out)
{
    double limit = sqrt(6.0 / (fan_in + fan_out));
    return ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit;
}

double he_init(int fan_in)
{
    double std = sqrt(2.0 / fan_in);
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int *create_feedforward_network(int *layer_sizes, int num_layers)
{
    int *prev_layer_slots = NULL;
    int *curr_layer_slots = NULL;

    for (int layer = 0; layer < num_layers; layer++)
    {
        curr_layer_slots = malloc(layer_sizes[layer] * sizeof(int));

        if (layer == 0)
        {
            for (int i = 0; i < layer_sizes[layer]; i++)
            {
                curr_layer_slots[i] = create_value_slot(0);
            }
        }
        else
        {
            for (int neuron = 0; neuron < layer_sizes[layer]; neuron++)
            {
                int sum = -1;
                for (int prev = 0; prev < layer_sizes[layer - 1]; prev++)
                {
                    int weight = create_value_slot(1);
                    for (int b = 0; b < BATCH_SIZE; b++)
                    {
                        double weight_init = he_init(layer_sizes[layer - 1]);
                        set_slot_value(weight, b, weight_init);
                    }
                    int mul = create_operation_slot(MULTIPLY, wrap_in_array(prev_layer_slots[prev], weight), 2);

                    if (sum == -1)
                    {
                        sum = mul;
                    }
                    else
                    {
                        sum = create_operation_slot(ADD, wrap_in_array(sum, mul), 2);
                    }
                }

                int bias = create_value_slot(1);
                int biased = create_operation_slot(ADD, wrap_in_array(sum, bias), 2);
                curr_layer_slots[neuron] = create_operation_slot(RELU, wrap_value_in_array(biased), 1);
            }
        }

        free(prev_layer_slots);
        prev_layer_slots = curr_layer_slots;
    }

    return curr_layer_slots;
}

int *create_softmax_layer(int *input_slots, int num_outputs)
{

    int *exp_slots = malloc(num_outputs * sizeof(int));

    for (int i = 0; i < num_outputs; i++)
    {
        exp_slots[i] = create_operation_slot(EXP, wrap_value_in_array(input_slots[i]), 1);
    }

    int sum_slot = exp_slots[0];
    for (int i = 1; i < num_outputs; i++)
    {
        sum_slot = create_operation_slot(ADD, wrap_in_array(sum_slot, exp_slots[i]), 2);
    }

    int *softmax_slots = malloc(num_outputs * sizeof(int));
    for (int i = 0; i < num_outputs; i++)
    {
        softmax_slots[i] = create_operation_slot(DIV, wrap_in_array(exp_slots[i], sum_slot), 2);
    }

    return softmax_slots;
}

int create_cross_entropy_loss(int *target_slots, int *softmax_slots, int num_outputs)
{
    int *log_slots = malloc(num_outputs * sizeof(int));
    int *product_slots = malloc(num_outputs * sizeof(int));

    for (int i = 0; i < num_outputs; i++)
    {
        int log_softmax = create_operation_slot(LOG, wrap_value_in_array(softmax_slots[i]), 1);
        log_slots[i] = log_softmax;
        product_slots[i] = create_operation_slot(MULTIPLY, wrap_in_array(target_slots[i], log_softmax), 2);
    }

    int sum_cross_entropy = create_operation_slot(ADD, product_slots, num_outputs);
    int neg_cross_entropy = create_operation_slot(NEG, wrap_value_in_array(sum_cross_entropy), 1);
    return neg_cross_entropy;
}

int zerograd()
{
    for (int j = 0; j < slot_counter; j++)
    {
        for (int b = 0; b < BATCH_SIZE; b++)
        {
            slots[j].gradient[b] = 0.0;
            slots[j].visited = 0;
        }
    }
    return 0;
}