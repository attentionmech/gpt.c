#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define BATCH_SIZE 10
#define MAX_ELEMENTS 1000000 // maximum elements in a single tensor
#define MAX_SLOTS 10000000
#define MAX_FILE_SIZE 10000

// bpe related
#define MAX_VOCAB_SIZE 10000
#define MAX_MERGES 100

typedef struct
{
    int pair[2];
    int merged;
} BPEMerge;

typedef enum
{
    LAYER_FEEDFORWARD,
    LAYER_ATTENTION
} LayerType;

typedef enum
{
    ADD,
    MULTIPLY,
    SUB,
    POW2, // square
    NEG,
    EXP,
    DIV,
    LOG,

    SIGMOID,
    RELU,
    GELU,
    LEAKY_RELU,

    PARAMETER, // nodes which are just values (leaf nodes)
} OperationType;

typedef struct
{
    double *value;
    double *gradient;
    int *shape;
    int *strides;
    int num_dimensions;
    int size;
    OperationType operation;
    int *dependencies;
    int num_dependencies;
    int learnable_param;
    int visited;
} Slot;

Slot slots[MAX_SLOTS];
int slot_counter = 0;
double **dependency_buffer;
bool is_referenced[MAX_SLOTS] = {false}; // orphan detection, optional

void reverse_merge(int *tokens, int *num_tokens, BPEMerge *merge, int data_length)
{
    int new_tokens[data_length * 2];
    int new_tokens_index = 0;

    for (int i = 0; i < *num_tokens; i++)
    {
        if (tokens[i] == merge->merged)
        {
            new_tokens[new_tokens_index++] = merge->pair[0];
            new_tokens[new_tokens_index++] = merge->pair[1];
        }
        else
        {
            new_tokens[new_tokens_index++] = tokens[i];
        }
    }

    for (int i = 0; i < new_tokens_index; i++)
    {
        tokens[i] = new_tokens[i];
    }
    *num_tokens = new_tokens_index;
}

void recover_original_tokens(int *tokens, int *num_tokens, BPEMerge *merges, int num_merges, int data_length)
{
    for (int i = num_merges - 1; i >= 0; i--)
    {
        reverse_merge(tokens, num_tokens, &merges[i], data_length);
    }
}

void find_most_frequent_pair(int *numbers, int num_numbers, int *pair)
{
    int max_count = 0;
    int most_frequent_pair[2] = {-1, -1};

    for (int i = 0; i < num_numbers - 1; i++)
    {
        int current_pair[2] = {numbers[i], numbers[i + 1]};
        int count = 0;

        for (int j = 0; j < num_numbers - 1; j++)
        {
            if (numbers[j] == current_pair[0] && numbers[j + 1] == current_pair[1])
            {
                count++;
            }
        }

        if (count > max_count)
        {
            max_count = count;
            most_frequent_pair[0] = current_pair[0];
            most_frequent_pair[1] = current_pair[1];
        }
    }

    pair[0] = most_frequent_pair[0];
    pair[1] = most_frequent_pair[1];
}

void merge_pair(int *numbers, int *num_numbers, int *pair, int merged_token, int data_length)
{
    int new_numbers[data_length * 2];
    int new_numbers_index = 0;

    for (int i = 0; i < *num_numbers;)
    {
        if (i < *num_numbers - 1 &&
            numbers[i] == pair[0] &&
            numbers[i + 1] == pair[1])
        {
            new_numbers[new_numbers_index++] = merged_token;
            i += 2;
        }
        else
        {
            new_numbers[new_numbers_index++] = numbers[i++];
        }
    }

    for (int i = 0; i < new_numbers_index; i++)
    {
        numbers[i] = new_numbers[i];
    }
    *num_numbers = new_numbers_index;
}

void bpe_tokenize(int *numbers, int *num_numbers, BPEMerge *merges, int num_merges, int data_length)
{
    for (int i = 0; i < num_merges; i++)
    {
        int pair[2] = {merges[i].pair[0], merges[i].pair[1]};
        merge_pair(numbers, num_numbers, pair, merges[i].merged, data_length);
    }
}

void learn_bpe_merges(int *numbers, int *num_numbers, BPEMerge *merges, int *num_merges, int data_length)
{
    int next_token = 256;

    for (int i = 0; i < MAX_MERGES; i++)
    {
        printf("BPE Merge %d\n", i);
        int pair[2] = {-1, -1};
        find_most_frequent_pair(numbers, *num_numbers, pair);

        if (pair[0] == -1 || pair[1] == -1)
        {
            break;
        }

        merges[*num_merges].pair[0] = pair[0];
        merges[*num_merges].pair[1] = pair[1];
        merges[*num_merges].merged = next_token;
        (*num_merges)++;
        next_token++;

        merge_pair(numbers, num_numbers, pair, merges[*num_merges - 1].merged, data_length);
    }
}

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

void detect_orphans()
{

    for (int i = 0; i < slot_counter; i++)
    {
        Slot *s = &slots[i];
        for (int j = 0; j < s->num_dependencies; j++)
        {
            int dep_slot = s->dependencies[j];
            is_referenced[dep_slot] = true;
        }
    }

    printf("Orphan slots:\n");
    for (int i = 0; i < slot_counter; i++)
    {
        if (!is_referenced[i] && slots[i].num_dependencies > 0)
        {
            printf("Slot %d (Operation: %s) is an orphan.\n", i, get_operation_name(slots[i].operation));
        }
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
    fprintf(file, "    rankdir=LR; // Left-to-right graph layout\n");
    fprintf(file, "    node [shape=record, style=filled];\n");

    for (int i = 0; i < slot_counter; i++)
    {
        Slot *s = &slots[i];

        fprintf(file, "    slot_%d [label=\"{%d | {", i, i);

        fprintf(file, "Op: %s", get_operation_name(s->operation));

        if (s->num_dimensions > 0)
        {
            fprintf(file, " | Shape: [");
            for (int d = 0; d < s->num_dimensions; d++)
            {
                fprintf(file, "%d", s->shape[d]);
                if (d < s->num_dimensions - 1)
                    fprintf(file, ", ");
            }
            fprintf(file, "]");
        }

        if (s->size > 0)
        {
            fprintf(file, " | Val: %.2f", s->value[0]);
            fprintf(file, " | Grad: %.2f", s->gradient[0]);
        }

        if (s->learnable_param)
        {
            fprintf(file, " | Learnable");
        }

        fprintf(file, "}}\", fillcolor=");

        if (s->operation == PARAMETER)
        {
            fprintf(file, "lightgreen");
        }
        else if (s->num_dependencies == 0)
        {
            fprintf(file, "lightblue");
        }
        else
        {
            fprintf(file, "lightpink");
        }

        fprintf(file, "];\n");

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

int increment_slot()
{
    if (slot_counter >= MAX_SLOTS)
    {
        fprintf(stderr, "Error: Exceeded maximum number of slots (%d)\n", MAX_SLOTS);
        exit(EXIT_FAILURE);
    }
    return slot_counter++;
}

double generate_normal(double mean, double stddev)
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    return z0 * stddev + mean;
}

int create_value_slot(int learnable_param, int *shape, int num_dimensions)
{
    slots[slot_counter].num_dimensions = num_dimensions;
    slots[slot_counter].shape = (int *)malloc(num_dimensions * sizeof(int));
    for (int i = 0; i < num_dimensions; i++)
    {
        slots[slot_counter].shape[i] = shape[i];
    }

    slots[slot_counter].strides = (int *)malloc(num_dimensions * sizeof(int));
    slots[slot_counter].strides[num_dimensions - 1] = 1;
    for (int i = num_dimensions - 2; i >= 0; i--)
    {
        slots[slot_counter].strides[i] =
            slots[slot_counter].strides[i + 1] * shape[i + 1];
    }

    int total_size = 1;
    for (int i = 0; i < num_dimensions; i++)
    {
        total_size *= shape[i];
    }

    if (total_size > MAX_ELEMENTS)
    {
        fprintf(stderr, "Error: Exceeded maximum number of elements in a single tensor (%d)\n", MAX_ELEMENTS);
        exit(EXIT_FAILURE);
    }

    slots[slot_counter].size = total_size;

    slots[slot_counter].value = (double *)malloc(total_size * sizeof(double));
    slots[slot_counter].gradient = (double *)malloc(total_size * sizeof(double));

    if (learnable_param)
    {
        for (int b = 0; b < slots[slot_counter].size; b++)
        {
            slots[slot_counter].value[b] = generate_normal(0.0, 1.0);
        }
    }

    slots[slot_counter].operation = PARAMETER;
    slots[slot_counter].num_dependencies = 0;
    slots[slot_counter].learnable_param = learnable_param;

    return increment_slot();
}

int create_operation_slot(OperationType op, int *dep, int num_dependencies, int *shape, int num_dimensions)
{
    slots[slot_counter].operation = op;
    slots[slot_counter].dependencies = dep;
    slots[slot_counter].num_dependencies = num_dependencies;

    slots[slot_counter].num_dimensions = num_dimensions;
    slots[slot_counter].shape = (int *)malloc(num_dimensions * sizeof(int));
    for (int i = 0; i < num_dimensions; i++)
    {
        slots[slot_counter].shape[i] = shape[i];
    }

    slots[slot_counter].strides = (int *)malloc(num_dimensions * sizeof(int));
    slots[slot_counter].strides[num_dimensions - 1] = 1;
    for (int i = num_dimensions - 2; i >= 0; i--)
    {
        slots[slot_counter].strides[i] =
            slots[slot_counter].strides[i + 1] * shape[i + 1];
    }

    int total_size = 1;
    for (int i = 0; i < num_dimensions; i++)
    {
        total_size *= shape[i];
    }

    if (total_size > MAX_ELEMENTS)
    {
        fprintf(stderr, "Error: Exceeded maximum number of elements in a single tensor (%d)\n", MAX_ELEMENTS);
        exit(EXIT_FAILURE);
    }

    slots[slot_counter].size = total_size;
    slots[slot_counter].value = (double *)malloc(total_size * sizeof(double));
    slots[slot_counter].gradient = (double *)malloc(total_size * sizeof(double));

    slots[slot_counter].learnable_param = 1;

    return increment_slot();
}

void set_slot_value_by_position(int slot, int *position, int num_dimensions, double value)
{
    int pos = 0;
    for (int i = 0; i < num_dimensions - 1; i++)
    {
        pos += slots[slot].strides[i] * position[i];
    }
    pos += position[num_dimensions - 1];

    if (pos >= slots[slot].size)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    slots[slot].value[pos] = value;
}

void set_slot_value(int slot, int b_index, double v)
{
    slots[slot].value[b_index] = v;
}

double get_slot_value(int slot, int index)
{
    if (index >= slots[slot].size)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }
    return slots[slot].value[index];
}

double get_slot_value_by_position(int slot, int *position, int num_dimensions)
{
    int pos = 0;
    for (int i = 0; i < num_dimensions - 1; i++)
    {
        pos += slots[slot].strides[i] * position[i];
    }
    pos += position[num_dimensions - 1];

    if (pos >= slots[slot].size)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    return slots[slot].value[pos];
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
            for (int b = 0; b < s->size; b++)
            {
                s->value[b] = _sum(dep_value, b, s->num_dependencies);
            }
            break;
        case MULTIPLY:
            for (int b = 0; b < s->size; b++)
            {
                s->value[b] = _mul(dep_value, b, s->num_dependencies);
            }
            break;

        case EXP:

            for (int b = 0; b < s->size; b++)
            {
                s->value[b] = exp(dep_value[0][b]);
            }
            break;
        case DIV:
            for (int b = 0; b < s->size; b++)
            {
                s->value[b] = dep_value[0][b] / dep_value[1][b];
            }

            break;

        case NEG:
            for (int b = 0; b < s->size; b++)
            {
                s->value[b] = -dep_value[0][b];
            }
            break;

        case LOG:
            for (int b = 0; b < s->size; b++)
            {
                s->value[b] = log(dep_value[0][b]);
            }
            break;

        case SUB:
            for (int b = 0; b < s->size; b++)
            {
                s->value[b] = dep_value[0][b] - dep_value[0][b];
            }
            break;
        case POW2:
            for (int b = 0; b < s->size; b++)
            {
                s->value[b] = dep_value[0][b] * dep_value[0][b];
            }
            break;
        case SIGMOID:
            for (int b = 0; b < s->size; b++)
            {

                s->value[b] = 1.0 / (1.0 + exp(-dep_value[0][b]));
            }
            break;
        case RELU:
            for (int b = 0; b < s->size; b++)
            {

                s->value[b] = fmax(0.0, dep_value[0][b]);
            }
            break;
        case LEAKY_RELU:
            for (int b = 0; b < s->size; b++)
            {
                double alpha = 0.01;
                s->value[b] = (dep_value[0][b] > 0) ? dep_value[0][b] : alpha * dep_value[0][b];
            }
            break;

        case GELU:
            for (int b = 0; b < s->size; b++)
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

    if (dependency_buffer == NULL)
    {
        dependency_buffer = (double **)malloc(MAX_ELEMENTS * sizeof(double *));
        for (int b = 0; b < MAX_ELEMENTS; b++)
        {
            dependency_buffer[b] = (double *)malloc(slots[slot].num_dependencies * sizeof(double));
        }
    }

    for (int curr = slot; curr >= 0; curr--)
    {
        Slot *s = &slots[curr];

        if (s->num_dependencies > 0)
        {

            for (int j = 0; j < s->num_dependencies; j++)
            {
                for (int b = 0; b < s->size; b++)
                {
                    dependency_buffer[b][j] = get_slot_value(s->dependencies[j], b);
                }
            }

            switch (s->operation)
            {
            case ADD:
                for (int i = 0; i < s->num_dependencies; i++)
                {
                    for (int b = 0; b < s->size; b++)
                    {
                        slots[s->dependencies[i]].gradient[b] += s->gradient[b];
                    }
                }
                break;

            case MULTIPLY:
                for (int b = 0; b < s->size; b++)
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
                for (int b = 0; b < s->size; b++)
                {
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b];
                    slots[s->dependencies[1]].gradient[b] -= s->gradient[b];
                }
                break;

            case POW2:
                for (int b = 0; b < s->size; b++)
                {
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * 2.0 * dependency_buffer[b][0];
                }
                break;

            case SIGMOID:
                for (int b = 0; b < s->size; b++)
                {
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * s->value[b] * (1.0 - s->value[b]);
                }
                break;

            case RELU:
                for (int b = 0; b < s->size; b++)
                {
                    if (dependency_buffer[b][0] > 0)
                    {
                        slots[s->dependencies[0]].gradient[b] += s->gradient[b];
                    }
                }
                break;

            case LEAKY_RELU:
                for (int b = 0; b < s->size; b++)
                {
                    double alpha = 0.01;
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] *
                                                             (dependency_buffer[b][0] > 0 ? 1.0 : alpha);
                }
                break;

            case GELU:
                for (int b = 0; b < s->size; b++)
                {
                    double x = dependency_buffer[b][0];
                    double tanh_arg = sqrt(2.0 / M_PI) * (x + 0.044715 * x * x * x);
                    double tanh_val = tanh(tanh_arg);
                    double derivative = 0.5 * (1 + tanh_val + x * (1 - tanh_val * tanh_val) * sqrt(2.0 / M_PI) * (1 + 3 * 0.044715 * x * x));
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * derivative;
                }
                break;

            case EXP:
                for (int b = 0; b < s->size; b++)
                {
                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * s->value[b];
                }
                break;

            case NEG:
                for (int b = 0; b < s->size; b++)
                {

                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] * -1.0;
                }
                break;

            case DIV:
                for (int b = 0; b < s->size; b++)
                {

                    slots[s->dependencies[0]].gradient[b] += s->gradient[b] / dependency_buffer[b][1];
                    slots[s->dependencies[1]].gradient[b] -= s->gradient[b] * dependency_buffer[b][0] / (dependency_buffer[b][1] * dependency_buffer[b][1]);
                }
                break;

            case LOG:
                for (int b = 0; b < s->size; b++)
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

int zerograd()
{
    for (int j = 0; j < slot_counter; j++)
    {
        for (int b = 0; b < slots[j].size; b++)
        {
            slots[j].gradient[b] = 0.0;
            slots[j].visited = 0;
        }
    }
    return 0;
}

//--------------------------------------------
// basicgrad ends here
// --------------------------------------------

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

double he_init(int fan_in)
{
    double std = sqrt(2.0 / fan_in);
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;
    return std * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int *create_softmax_layer(int *input_slots, int num_outputs)
{

    int *exp_slots = malloc(num_outputs * sizeof(int));

    for (int i = 0; i < num_outputs; i++)
    {
        exp_slots[i] = create_operation_slot(EXP, wrap_value_in_array(input_slots[i]), 1, (int[]){BATCH_SIZE, 1}, 2);
    }

    int sum_slot = exp_slots[0];
    for (int i = 1; i < num_outputs; i++)
    {
        sum_slot = create_operation_slot(ADD, wrap_in_array(sum_slot, exp_slots[i]), 2, (int[]){BATCH_SIZE, 1}, 2);
    }

    int *softmax_slots = malloc(num_outputs * sizeof(int));
    for (int i = 0; i < num_outputs; i++)
    {
        softmax_slots[i] = create_operation_slot(DIV, wrap_in_array(exp_slots[i], sum_slot), 2, (int[]){BATCH_SIZE, 1}, 2);
    }

    return softmax_slots;
}

int create_cross_entropy_loss(int *target_slots, int *softmax_slots, int num_outputs)
{
    int *log_slots = malloc(num_outputs * sizeof(int));
    int *product_slots = malloc(num_outputs * sizeof(int));

    for (int i = 0; i < num_outputs; i++)
    {
        int log_softmax = create_operation_slot(LOG, wrap_value_in_array(softmax_slots[i]), 1, (int[]){BATCH_SIZE, 1}, 2);
        log_slots[i] = log_softmax;
        product_slots[i] = create_operation_slot(MULTIPLY, wrap_in_array(target_slots[i], log_softmax), 2, (int[]){BATCH_SIZE, 1}, 2);
    }

    int sum_cross_entropy = create_operation_slot(ADD, product_slots, num_outputs, (int[]){BATCH_SIZE, 1}, 2);
    int neg_cross_entropy = create_operation_slot(NEG, wrap_value_in_array(sum_cross_entropy), 1, (int[]){BATCH_SIZE, 1}, 2);
    return neg_cross_entropy;
}

int *create_multihead_attention_layer(int *input_slots, int num_inputs, int d_model, int num_heads)
{
    int head_dim = d_model / num_heads;
    if (d_model % num_heads != 0)
    {
        printf("Error: d_model must be divisible by num_heads.\n");
        return NULL;
    }

    int **Q_weights = malloc(num_heads * sizeof(int *));
    int **K_weights = malloc(num_heads * sizeof(int *));
    int **V_weights = malloc(num_heads * sizeof(int *));
    for (int h = 0; h < num_heads; h++)
    {
        Q_weights[h] = malloc(head_dim * num_inputs * sizeof(int));
        K_weights[h] = malloc(head_dim * num_inputs * sizeof(int));
        V_weights[h] = malloc(head_dim * num_inputs * sizeof(int));

        for (int i = 0; i < head_dim * num_inputs; i++)
        {
            Q_weights[h][i] = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);
            K_weights[h][i] = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);
            V_weights[h][i] = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);

            double weight_init = he_init(num_inputs);
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                set_slot_value_by_position(Q_weights[h][i], (int[]){b, 0}, 2, weight_init);
                set_slot_value_by_position(K_weights[h][i], (int[]){b, 0}, 2, weight_init);
                set_slot_value_by_position(V_weights[h][i], (int[]){b, 0}, 2, weight_init);
            }
        }
    }

    int **Q = malloc(num_heads * sizeof(int *));
    int **K = malloc(num_heads * sizeof(int *));
    int **V = malloc(num_heads * sizeof(int *));
    for (int h = 0; h < num_heads; h++)
    {
        Q[h] = malloc(num_inputs * head_dim * sizeof(int));
        K[h] = malloc(num_inputs * head_dim * sizeof(int));
        V[h] = malloc(num_inputs * head_dim * sizeof(int));

        for (int i = 0; i < num_inputs; i++)
        {
            for (int j = 0; j < head_dim; j++)
            {
                int q_sum = -1, k_sum = -1, v_sum = -1;
                for (int k = 0; k < num_inputs; k++)
                {
                    int q_mul = create_operation_slot(MULTIPLY, wrap_in_array(input_slots[k], Q_weights[h][j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    int k_mul = create_operation_slot(MULTIPLY, wrap_in_array(input_slots[k], K_weights[h][j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    int v_mul = create_operation_slot(MULTIPLY, wrap_in_array(input_slots[k], V_weights[h][j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);

                    if (q_sum == -1)
                        q_sum = q_mul;
                    else
                        q_sum = create_operation_slot(ADD, wrap_in_array(q_sum, q_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                    if (k_sum == -1)
                        k_sum = k_mul;
                    else
                        k_sum = create_operation_slot(ADD, wrap_in_array(k_sum, k_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                    if (v_sum == -1)
                        v_sum = v_mul;
                    else
                        v_sum = create_operation_slot(ADD, wrap_in_array(v_sum, v_mul), 2, (int[]){BATCH_SIZE, 1}, 2);
                }
                Q[h][i * head_dim + j] = q_sum;
                K[h][i * head_dim + j] = k_sum;
                V[h][i * head_dim + j] = v_sum;
            }
        }
    }

    int seq_length = num_inputs;
    int *context = malloc(seq_length * d_model * sizeof(int));

    for (int h = 0; h < num_heads; h++)
    {
        int *attention_scores = malloc(seq_length * seq_length * sizeof(int));
        for (int i = 0; i < seq_length; i++)
        {
            for (int j = 0; j < seq_length; j++)
            {
                int sum = -1;
                for (int k = 0; k < head_dim; k++)
                {
                    int q_idx = i * head_dim + k;
                    int k_idx = j * head_dim + k;
                    int mul = create_operation_slot(MULTIPLY, wrap_in_array(Q[h][q_idx], K[h][k_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    if (sum == -1)
                        sum = mul;
                    else
                        sum = create_operation_slot(ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
                }
                double scale = sqrt(head_dim);
                int scale_slot = create_value_slot(0, (int[]){BATCH_SIZE, 1}, 2);
                for (int b = 0; b < BATCH_SIZE; b++)
                    set_slot_value_by_position(scale_slot, (int[]){b, 0}, 2, scale);

                attention_scores[i * seq_length + j] =
                    create_operation_slot(DIV, wrap_in_array(sum, scale_slot), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
        }

        int *attention_weights = malloc(seq_length * seq_length * sizeof(int));
        for (int i = 0; i < seq_length; i++)
        {
            int *row_scores = &attention_scores[i * seq_length];
            int *softmax_row = create_softmax_layer(row_scores, seq_length);
            for (int j = 0; j < seq_length; j++)
            {
                attention_weights[i * seq_length + j] = softmax_row[j];
            }
        }

        for (int i = 0; i < seq_length; i++)
        {
            for (int k = 0; k < head_dim; k++)
            {
                int sum = -1;
                for (int j = 0; j < seq_length; j++)
                {
                    int v_idx = j * head_dim + k;
                    int mul = create_operation_slot(MULTIPLY, wrap_in_array(attention_weights[i * seq_length + j], V[h][v_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    if (sum == -1)
                        sum = mul;
                    else
                        sum = create_operation_slot(ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
                }
                context[i * d_model + h * head_dim + k] = sum;
            }
        }

        free(attention_scores);
        free(attention_weights);
    }

    free(Q_weights);
    free(K_weights);
    free(V_weights);
    free(Q);
    free(K);
    free(V);

    return context;
}

int *create_attention_layer(int *input_slots, int num_inputs, int d_model)
{
    int Q_weights[d_model * num_inputs];
    int K_weights[d_model * num_inputs];
    int V_weights[d_model * num_inputs];

    for (int i = 0; i < d_model * num_inputs; i++)
    {
        Q_weights[i] = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);
        K_weights[i] = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);
        V_weights[i] = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);

        double weight_init = he_init(num_inputs);
        for (int b = 0; b < BATCH_SIZE; b++)
        {
            set_slot_value_by_position(Q_weights[i], (int[]){b, 0}, 2, weight_init);
            set_slot_value_by_position(K_weights[i], (int[]){b, 0}, 2, weight_init);
            set_slot_value_by_position(V_weights[i], (int[]){b, 0}, 2, weight_init);
        }
    }

    int Q[num_inputs * d_model];
    int K[num_inputs * d_model];
    int V[num_inputs * d_model];

    for (int i = 0; i < num_inputs; i++)
    {
        for (int j = 0; j < d_model; j++)
        {
            int q_sum = -1, k_sum = -1, v_sum = -1;
            for (int k = 0; k < num_inputs; k++)
            {
                int q_mul = create_operation_slot(MULTIPLY, wrap_in_array(input_slots[k], Q_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                int k_mul = create_operation_slot(MULTIPLY, wrap_in_array(input_slots[k], K_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                int v_mul = create_operation_slot(MULTIPLY, wrap_in_array(input_slots[k], V_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (q_sum == -1)
                    q_sum = q_mul;
                else
                    q_sum = create_operation_slot(ADD, wrap_in_array(q_sum, q_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (k_sum == -1)
                    k_sum = k_mul;
                else
                    k_sum = create_operation_slot(ADD, wrap_in_array(k_sum, k_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (v_sum == -1)
                    v_sum = v_mul;
                else
                    v_sum = create_operation_slot(ADD, wrap_in_array(v_sum, v_mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            Q[i * d_model + j] = q_sum;
            K[i * d_model + j] = k_sum;
            V[i * d_model + j] = v_sum;
        }
    }

    int seq_length = num_inputs;
    int *attention_scores = malloc(seq_length * seq_length * sizeof(int));

    for (int i = 0; i < seq_length; i++)
    {
        for (int j = 0; j < seq_length; j++)
        {
            int sum = -1;
            for (int k = 0; k < d_model; k++)
            {
                int q_idx = i * d_model + k;
                int k_idx = j * d_model + k;
                int mul = create_operation_slot(MULTIPLY, wrap_in_array(Q[q_idx], K[k_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                if (sum == -1)
                    sum = mul;
                else
                    sum = create_operation_slot(ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            double scale = sqrt(d_model);
            int scale_slot = create_value_slot(0, (int[]){BATCH_SIZE, 1}, 2);
            for (int b = 0; b < BATCH_SIZE; b++)
                set_slot_value_by_position(scale_slot, (int[]){b, 0}, 2, scale);

            attention_scores[i * seq_length + j] =
                create_operation_slot(DIV, wrap_in_array(sum, scale_slot), 2, (int[]){BATCH_SIZE, 1}, 2);
        }
    }

    int *attention_weights = malloc(seq_length * seq_length * sizeof(int));
    for (int i = 0; i < seq_length; i++)
    {
        int *row_scores = &attention_scores[i * seq_length];
        int *softmax_row = create_softmax_layer(row_scores, seq_length);
        for (int j = 0; j < seq_length; j++)
        {
            attention_weights[i * seq_length + j] = softmax_row[j];
        }
    }

    int *context = malloc(seq_length * d_model * sizeof(int));
    for (int i = 0; i < seq_length; i++)
    {
        for (int k = 0; k < d_model; k++)
        {
            int sum = -1;
            for (int j = 0; j < seq_length; j++)
            {
                int v_idx = j * d_model + k;
                int mul = create_operation_slot(MULTIPLY, wrap_in_array(attention_weights[i * seq_length + j], V[v_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                if (sum == -1)
                    sum = mul;
                else
                    sum = create_operation_slot(ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            context[i * d_model + k] = sum;
            // printf("Context: %d\n", context[i * d_model + k]);
        }
    }

    free(attention_scores);
    free(attention_weights);
    return context;
}

int *create_feedforward_network(int *prev_layer_slots, int prev_layer_size, int curr_layer_size)
{
    int *curr_layer_slots = malloc(curr_layer_size * sizeof(int));

    for (int neuron = 0; neuron < curr_layer_size; neuron++)
    {
        int sum = -1;
        for (int prev = 0; prev < prev_layer_size; prev++)
        {
            int weight = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);
            for (int b = 0; b < slots[weight].size; b++)
            {
                double weight_init = he_init(prev_layer_size);
                set_slot_value_by_position(weight, (int[]){b, 0}, 2, weight_init);
            }
            int mul = create_operation_slot(MULTIPLY, wrap_in_array(prev_layer_slots[prev], weight), 2, (int[]){BATCH_SIZE, 1}, 2);

            if (sum == -1)
            {
                sum = mul;
            }
            else
            {
                sum = create_operation_slot(ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
        }

        int bias = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);
        int biased = create_operation_slot(ADD, wrap_in_array(sum, bias), 2, (int[]){BATCH_SIZE, 1}, 2);
        curr_layer_slots[neuron] = create_operation_slot(RELU, wrap_value_in_array(biased), 1, (int[]){BATCH_SIZE, 1}, 2);
    }

    return curr_layer_slots;
}

void train(double **inputs, int labels[], int num_samples, double learning_rate, int *layer_sizes, int num_heads, LayerType *layer_types, int num_layers, int *index_to_char, int vocab_size, int data_length, BPEMerge *merges, int num_merges)
{
    int num_inputs = layer_sizes[0];
    int num_outputs = layer_sizes[num_layers - 1];

    int *prev_layer = NULL;
    int *curr_layer = NULL;

    prev_layer = malloc(num_inputs * vocab_size * sizeof(int));
    for (int i = 0; i < num_inputs; i++)
    {
        for (int j = 0; j < vocab_size; j++)
        {
            int slot_index = i * vocab_size + j;
            prev_layer[slot_index] = create_value_slot(0, (int[]){BATCH_SIZE, 1}, 2);
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                set_slot_value_by_position(prev_layer[slot_index], (int[]){b, 0}, 2, inputs[b][i] == j ? 1.0 : 0.0);
            }
        }
    }

    for (int i = 1; i < num_layers; i++)
    {
        if (layer_types[i] == LAYER_ATTENTION)
        {
            // curr_layer = create_attention_layer(prev_layer, layer_sizes[i - 1], layer_sizes[i]); // calling d_model layer_size is anti-naming but meh for now
            curr_layer = create_multihead_attention_layer(prev_layer, layer_sizes[i - 1], layer_sizes[i], num_heads);
        }
        else if (layer_types[i] == LAYER_FEEDFORWARD)
        {
            if (layer_types[i - 1] == LAYER_ATTENTION)
            {
                curr_layer = create_feedforward_network(curr_layer, layer_sizes[i - 1] * layer_sizes[i - 2], layer_sizes[i]);
            }
            else
            {
                curr_layer = create_feedforward_network(prev_layer, layer_sizes[i - 1], layer_sizes[i]);
            }
        }
        else
        {
            fprintf(stderr, "Unknown layer type at index %d\n", i);
            exit(1);
        }

        if (prev_layer)
            free(prev_layer);
        prev_layer = curr_layer;
    }
    int *output_slots = curr_layer;
    int *softmax_slots = create_softmax_layer(output_slots, num_outputs);

    int target_slots[num_outputs * vocab_size];
    for (int i = 0; i < num_outputs; i++)
    {
        for (int j = 0; j < vocab_size; j++)
        {
            int slot_index = i * vocab_size + j;
            target_slots[slot_index] = create_value_slot(0, (int[]){BATCH_SIZE, 1}, 2);
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                // One-hot encoding: set 1.0 for the correct token index, 0.0 otherwise
                set_slot_value_by_position(target_slots[slot_index], (int[]){b, 0}, 2, labels[b] == j ? 1.0 : 0.0);
            }
        }
    }
    int loss_slot = create_cross_entropy_loss(target_slots, softmax_slots, num_outputs);

    detect_orphans();
    // exit(1);

    srand(time(NULL));

    int EPOCHS = 1000;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_loss = 0.0;

        for (int i = 0; i + BATCH_SIZE < num_samples; i += BATCH_SIZE)
        {

            zerograd();

            for (int k = 0; k < num_inputs; k++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    set_slot_value_by_position(k, (int[]){b, 0}, 2, inputs[i + b][k]);
                }
            }

            for (int l = 1; l <= num_outputs; l++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    if (l == labels[i + b])
                    {
                        set_slot_value_by_position(target_slots[l - 1], (int[]){b, 0}, 2, 1);
                    }
                    else
                    {
                        set_slot_value_by_position(target_slots[l - 1], (int[]){b, 0}, 2, 0);
                    }
                }
            }

            compute_graph(loss_slot);

            for (int b = 0; b < BATCH_SIZE; b++)
            {
                total_loss += get_slot_value_by_position(loss_slot, (int[]){b, 0}, 2);
                slots[loss_slot].gradient[b] = 1.0;
            }

            compute_grad(loss_slot);

            for (int j = 0; j < slot_counter; j++)
            {
                if (slots[j].learnable_param == 1)
                {
                    double grad_sum = 0.0;
                    for (int b = 0; b < slots[j].size; b++)
                    {
                        grad_sum += slots[j].gradient[b];
                    }

                    for (int b = 0; b < slots[j].size; b++)
                    {
                        set_slot_value(j, b, get_slot_value(j, b) - learning_rate * grad_sum / slots[j].size);
                    }
                }
            }
        }

        // int seq_len = num_inputs / vocab_size;
        // int max_index = -111;

        // for (int p = 0; p < 50; p++)
        // {
        //     if (max_index != -111)
        //     {
        //         for (int k = 0; k < (seq_len - 1); k++)
        //         {
        //             set_slot_value_by_position(k, (int[]){0, 0}, 2, inputs[0][k + 1]);
        //         }
        //         inputs[0][(seq_len - 1)] = max_index;
        //     }

        //     compute_graph(loss_slot);

        //     double temperature = 0.6;
        //     double softmax_values[num_outputs];
        //     double exp_sum = 0.0;

        //     for (int j = 0; j < num_outputs; j++)
        //     {
        //         double raw_value = get_slot_value_by_position(softmax_slots[j], (int[]){0, 0}, 2);
        //         softmax_values[j] = exp(raw_value / temperature);
        //         exp_sum += softmax_values[j];
        //     }

        //     for (int j = 0; j < num_outputs; j++)
        //     {
        //         softmax_values[j] /= exp_sum;
        //     }

        //     double cumulative_prob = 0.0;
        //     double random_value = (double)rand() / RAND_MAX;
        //     int max_index = -111;

        //     for (int j = 0; j < num_outputs; j++)
        //     {
        //         cumulative_prob += softmax_values[j];
        //         if (random_value <= cumulative_prob)
        //         {
        //             max_index = j;
        //             break;
        //         }
        //     }
        //     int num_tokens = 1;
        //     int temp[MAX_MERGES] = {0};
        //     temp[0] = index_to_char[max_index];
        //     recover_original_tokens(temp, &num_tokens, merges, num_merges, data_length);
        //     assert(num_tokens < MAX_MERGES);

        //     // alternatively I can pass all the characters together also,
        //     //  but then that would need a larger allocation of array
        //     //  and this is also equivalent only
        //     for (int x = 0; x < num_tokens; x++)
        //     {
        //         printf("%c", (char)temp[x]);
        //     }
        // }
        // printf("\n");

        printf("Epoch %d, Avg. Loss: %f\n\n", epoch + 1, total_loss / num_samples);
    }
}

int main()
{
    FILE *file = fopen("dataset/tinystories.txt", "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error opening file\n");
        return 1;
    }

    fseek(file, 0, SEEK_END);
    long file_size = ftell(file);

    file_size = file_size > MAX_FILE_SIZE ? MAX_FILE_SIZE : file_size;

    fseek(file, 0, SEEK_SET);

    char *data = (char *)malloc(file_size + 1);
    int data_length = fread(data, sizeof(char), file_size, file);
    data[data_length] = '\0';
    fclose(file);

    int *numbers = (int *)malloc(data_length * 2 * sizeof(int));
    int num_numbers = 0;
    for (int i = 0; i < data_length; i++)
    {
        numbers[num_numbers++] = (unsigned char)data[i];
    }

    BPEMerge merges[MAX_MERGES];
    int num_merges = 0;
    learn_bpe_merges(numbers, &num_numbers, merges, &num_merges, data_length);

    bpe_tokenize(numbers, &num_numbers, merges, num_merges, data_length);

    int token_to_index[MAX_VOCAB_SIZE];
    int index_to_token[MAX_VOCAB_SIZE];
    for (int i = 0; i < MAX_VOCAB_SIZE; i++)
    {
        token_to_index[i] = -1;
    }

    int vocab_size = 0;
    for (int i = 0; i < num_numbers; i++)
    {
        if (token_to_index[numbers[i]] == -1)
        {
            token_to_index[numbers[i]] = vocab_size;
            index_to_token[vocab_size] = numbers[i];
            vocab_size++;
        }
    }

    int SEQUENCE_LENGTH = 4;

    int input_size = SEQUENCE_LENGTH;
    int num_samples = num_numbers - SEQUENCE_LENGTH;

    if (num_samples > 10000)
    {
        num_samples = 10000;
    }

    int labels[num_samples];
    double **inputs = (double **)malloc(num_samples * sizeof(double *));
    for (int i = 0; i < num_samples; i++)
    {
        inputs[i] = (double *)malloc(input_size * sizeof(double));
    }

    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < input_size; j++)
        {
            inputs[i][j] = (double)token_to_index[numbers[i + j]];
        }
        labels[i] = (double)token_to_index[numbers[i + input_size]];
    }

    double learning_rate = 0.01;
    int layer_sizes[] = {input_size * vocab_size, 4, 4, 4, vocab_size};
    int num_heads = 2;
    LayerType layer_types[] = {LAYER_FEEDFORWARD, LAYER_FEEDFORWARD, LAYER_ATTENTION, LAYER_FEEDFORWARD, LAYER_FEEDFORWARD};
    int num_layers = 5;

    train(inputs, labels, num_samples, learning_rate, layer_sizes, num_heads, layer_types, num_layers, index_to_token, vocab_size, data_length, merges, num_merges);

    for (int i = 0; i < num_samples; i++)
    {
        free(inputs[i]);
    }
    free(inputs);
    free(data);

    return 0;
}
