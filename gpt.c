#include <stdbool.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define BATCH_SIZE 10
#define MAX_NODES 10000000
#define MAX_ELEMENTS 1000000 // maximum elements in a single tensor
#define MAX_FILE_SIZE 10000
#define MAX_SAMPLES 100

// bpe related
// byte pair encoder just takes your characters and
// merges them together as many times you wish
// until you get a desired vocabulary size
// merges are based on frequency of token pairs
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
    LAYER_ATTENTION,
} ComponentType;

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

    DROPOUT,

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
    double *adam_m;
    double *adam_v;
    double dropout_rate;
    bool dropped;
} Node;

typedef struct
{
    Node *nodes;
    int *input_nodes;
    int *output_nodes;
    int *target_nodes;
    int *softmax_nodes;
    int loss_node;
    int num_inputs;
    int num_outputs;
    int embed_size;
} Model;

int node_counter = 0;
double **dependency_buffer;

// PE related functions

double *generate_positional_encoding(int seq_length, int embed_size)
{
    double *positional_encoding = (double *)malloc(seq_length * embed_size * sizeof(double));
    for (int pos = 0; pos < seq_length; pos++)
    {
        for (int i = 0; i < embed_size; i++)
        {
            double angle = pos / pow(10000, (2 * (i / 2)) / (double)embed_size);
            if (i % 2 == 0)
            {
                positional_encoding[pos * embed_size + i] = sin(angle);
            }
            else
            {
                positional_encoding[pos * embed_size + i] = cos(angle);
            }
        }
    }
    return positional_encoding;
}

// BPE related functions

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

double *preprocess_string(const char *str, int *token_to_index, BPEMerge *merges, int num_merges, int vocab_size, int seq_length)
{
    int len = strlen(str);
    int numbers[len];
    int num_numbers = 0;
    for (int i = 0; i < len; i++)
    {
        numbers[num_numbers++] = (unsigned char)str[i];
    }

    bpe_tokenize(numbers, &num_numbers, merges, num_merges, len);

    double *sequence = (double *)malloc(seq_length * sizeof(double));
    for (int i = 0; i < seq_length; ++i)
    {
        if (i < num_numbers)
        {
            if (token_to_index[numbers[i]] != -1)
            {
                sequence[i] = (double)token_to_index[numbers[i]];
            }
            else
            {
                fprintf(stderr, "Error: Token not found in vocabulary.\n");
                exit(1);
            }
        }
        else
        {
            sequence[i] = 0; // Pad with a default value
        }
    }
    return sequence;
}

// BPE functions endding here

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

void detect_orphans(Model *model)
{
    bool *is_referenced = (bool *)malloc(MAX_NODES * sizeof(bool));
    for (int i = 0; i < MAX_NODES; i++)
    {
        is_referenced[i] = false;
    }

    for (int i = 0; i < node_counter; i++)
    {
        Node *s = &model->nodes[i];
        for (int j = 0; j < s->num_dependencies; j++)
        {
            int dep_node = s->dependencies[j];
            is_referenced[dep_node] = true;
        }
    }

    printf("Orphan nodes (only loss node should be here):\n");
    for (int i = 0; i < node_counter; i++)
    {
        if (!is_referenced[i] && model->nodes[i].num_dependencies > 0)
        {
            printf("Node %d (Operation: %s) is an orphan.\n", i, get_operation_name(model->nodes[i].operation));
        }
    }

    free(is_referenced);
}

void export_graph_to_dot(Model *model, const char *filename)
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

    for (int i = 0; i < node_counter; i++)
    {
        Node *s = &model->nodes[i];

        fprintf(file, "    node_%d [label=\"{%d | {", i, i);

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
            fprintf(file, "    node_%d -> node_%d;\n", s->dependencies[j], i);
        }
    }

    fprintf(file, "}\n");
    fclose(file);
    printf("Graph exported to %s. Use 'dot -Tpng %s -o graph.png' to generate an image.\n", filename, filename);
}

// --------------------------------------------

int increment_node()
{
    if (node_counter >= MAX_NODES)
    {
        fprintf(stderr, "Error: Exceeded maximum number of nodes (%d)\n", MAX_NODES);
        exit(EXIT_FAILURE);
    }
    return node_counter++;
}

double generate_normal(double mean, double stddev)
{
    double u1 = (double)rand() / RAND_MAX;
    double u2 = (double)rand() / RAND_MAX;

    double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

    return z0 * stddev + mean;
}

int create_value_node(Model *model, int learnable_param, int *shape, int num_dimensions)
{
    model->nodes[node_counter].num_dimensions = num_dimensions;
    model->nodes[node_counter].shape = (int *)malloc(num_dimensions * sizeof(int));
    for (int i = 0; i < num_dimensions; i++)
    {
        model->nodes[node_counter].shape[i] = shape[i];
    }

    model->nodes[node_counter].strides = (int *)malloc(num_dimensions * sizeof(int));
    model->nodes[node_counter].strides[num_dimensions - 1] = 1;
    for (int i = num_dimensions - 2; i >= 0; i--)
    {
        model->nodes[node_counter].strides[i] =
            model->nodes[node_counter].strides[i + 1] * shape[i + 1];
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

    model->nodes[node_counter].size = total_size;

    model->nodes[node_counter].value = (double *)malloc(total_size * sizeof(double));
    model->nodes[node_counter].gradient = (double *)malloc(total_size * sizeof(double));

    if (learnable_param)
    {
        for (int b = 0; b < model->nodes[node_counter].size; b++)
        {
            model->nodes[node_counter].value[b] = generate_normal(0.0, 1.0);
        }
    }

    model->nodes[node_counter].operation = PARAMETER;
    model->nodes[node_counter].num_dependencies = 0;
    model->nodes[node_counter].learnable_param = learnable_param;

    model->nodes[node_counter].dropped = false;
    model->nodes[node_counter].dropout_rate = 0;

    return increment_node();
}

int create_operation_node(Model *model, OperationType op, int *dep, int num_dependencies, int *shape, int num_dimensions)
{
    model->nodes[node_counter].operation = op;
    model->nodes[node_counter].dependencies = dep;
    model->nodes[node_counter].num_dependencies = num_dependencies;

    model->nodes[node_counter].num_dimensions = num_dimensions;
    model->nodes[node_counter].shape = (int *)malloc(num_dimensions * sizeof(int));
    for (int i = 0; i < num_dimensions; i++)
    {
        model->nodes[node_counter].shape[i] = shape[i];
    }

    model->nodes[node_counter].strides = (int *)malloc(num_dimensions * sizeof(int));
    model->nodes[node_counter].strides[num_dimensions - 1] = 1;
    for (int i = num_dimensions - 2; i >= 0; i--)
    {
        model->nodes[node_counter].strides[i] =
            model->nodes[node_counter].strides[i + 1] * shape[i + 1];
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

    model->nodes[node_counter].size = total_size;
    model->nodes[node_counter].value = (double *)malloc(total_size * sizeof(double));
    model->nodes[node_counter].gradient = (double *)malloc(total_size * sizeof(double));

    model->nodes[node_counter].learnable_param = 1;

    model->nodes[node_counter].dropped = false;
    model->nodes[node_counter].dropout_rate = 0;

    return increment_node();
}

void set_node_value_by_position(Model *model, int node, int *position, int num_dimensions, double value)
{
    int pos = 0;
    for (int i = 0; i < num_dimensions - 1; i++)
    {
        pos += model->nodes[node].strides[i] * position[i];
    }
    pos += position[num_dimensions - 1];

    if (pos >= model->nodes[node].size)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    model->nodes[node].value[pos] = value;
}

void set_node_value(Model *model, int node, int b_index, double v)
{
    model->nodes[node].value[b_index] = v;
}

double get_node_value(Model *model, int node, int index)
{
    if (index >= model->nodes[node].size)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }
    return model->nodes[node].value[index];
}

double get_node_value_by_position(Model *model, int node, int *position, int num_dimensions)
{
    int pos = 0;
    for (int i = 0; i < num_dimensions - 1; i++)
    {
        pos += model->nodes[node].strides[i] * position[i];
    }
    pos += position[num_dimensions - 1];

    if (pos >= model->nodes[node].size)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    return model->nodes[node].value[pos];
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

double *compute_graph(Model *model, int node)
{
    Node *s = &model->nodes[node];

    if (s->visited)
    {
        return s->value;
    }

    if (s->num_dependencies > 0)
    {
        double *dep_value[s->num_dependencies];
        for (int j = 0; j < s->num_dependencies; j++)
        {
            dep_value[j] = compute_graph(model, s->dependencies[j]);
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
        case DROPOUT:
            for (int b = 0; b < s->size; ++b)
            {
                if ((double)rand() / RAND_MAX > s->dropout_rate)
                {
                    s->value[b] = dep_value[0][b];
                    s->dropped = 0;
                }
                else
                {
                    s->value[b] = 0.0;
                    s->dropped = 1;
                }
            }
            break;

        default:
            break;
        }
    }

    s->visited = 1;
    return s->value;
}

void compute_grad(Model *model, int node)
{

    if (dependency_buffer == NULL)
    {
        dependency_buffer = (double **)malloc(MAX_ELEMENTS * sizeof(double *));
        for (int b = 0; b < MAX_ELEMENTS; b++)
        {
            dependency_buffer[b] = (double *)malloc(model->nodes[node].num_dependencies * sizeof(double));
        }
    }

    for (int curr = node; curr >= 0; curr--)
    {
        Node *s = &model->nodes[curr];

        if (s->dropped == true)
        {
            continue;
        }

        if (s->num_dependencies > 0)
        {

            for (int j = 0; j < s->num_dependencies; j++)
            {
                for (int b = 0; b < s->size; b++)
                {
                    dependency_buffer[b][j] = get_node_value(model, s->dependencies[j], b);
                }
            }

            switch (s->operation)
            {
            case ADD:
                for (int i = 0; i < s->num_dependencies; i++)
                {
                    for (int b = 0; b < s->size; b++)
                    {
                        model->nodes[s->dependencies[i]].gradient[b] += s->gradient[b];
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
                        model->nodes[s->dependencies[i]].gradient[b] += s->gradient[b] * (product / dependency_buffer[b][i]);
                    }
                }
                break;

            case SUB:
                for (int b = 0; b < s->size; b++)
                {
                    model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b];
                    model->nodes[s->dependencies[1]].gradient[b] -= s->gradient[b];
                }
                break;

            case POW2:
                for (int b = 0; b < s->size; b++)
                {
                    model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b] * 2.0 * dependency_buffer[b][0];
                }
                break;

            case SIGMOID:
                for (int b = 0; b < s->size; b++)
                {
                    model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b] * s->value[b] * (1.0 - s->value[b]);
                }
                break;

            case RELU:
                for (int b = 0; b < s->size; b++)
                {
                    if (dependency_buffer[b][0] > 0)
                    {
                        model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b];
                    }
                }
                break;

            case LEAKY_RELU:
                for (int b = 0; b < s->size; b++)
                {
                    double alpha = 0.01;
                    model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b] *
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
                    model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b] * derivative;
                }
                break;

            case EXP:
                for (int b = 0; b < s->size; b++)
                {
                    model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b] * s->value[b];
                }
                break;

            case NEG:
                for (int b = 0; b < s->size; b++)
                {

                    model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b] * -1.0;
                }
                break;

            case DIV:
                for (int b = 0; b < s->size; b++)
                {

                    model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b] / dependency_buffer[b][1];
                    model->nodes[s->dependencies[1]].gradient[b] -= s->gradient[b] * dependency_buffer[b][0] / (dependency_buffer[b][1] * dependency_buffer[b][1]);
                }
                break;

            case LOG:
                for (int b = 0; b < s->size; b++)
                {

                    model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b] * (1.0 / dependency_buffer[b][0]);
                }
                break;

            case DROPOUT:
                for (int b = 0; b < s->size; b++)
                {
                    if (!s->dropped)
                    {
                        model->nodes[s->dependencies[0]].gradient[b] += s->gradient[b];
                    }
                }
                break;

            default:
                break;
            }
        }
    }
}

int zerograd(Model *model)
{
    for (int j = 0; j < node_counter; j++)
    {
        for (int b = 0; b < model->nodes[j].size; b++)
        {
            model->nodes[j].gradient[b] = 0.0;
            model->nodes[j].visited = 0;
            model->nodes[j].dropped = 0;
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

int *create_softmax_layer(Model *model, int *input_nodes, int num_outputs)
{

    int *exp_nodes = malloc(num_outputs * sizeof(int));

    for (int i = 0; i < num_outputs; i++)
    {
        exp_nodes[i] = create_operation_node(model, EXP, wrap_value_in_array(input_nodes[i]), 1, (int[]){BATCH_SIZE, 1}, 2);
    }

    int sum_node = exp_nodes[0];
    for (int i = 1; i < num_outputs; i++)
    {
        sum_node = create_operation_node(model, ADD, wrap_in_array(sum_node, exp_nodes[i]), 2, (int[]){BATCH_SIZE, 1}, 2);
    }

    int *softmax_nodes = malloc(num_outputs * sizeof(int));
    for (int i = 0; i < num_outputs; i++)
    {
        softmax_nodes[i] = create_operation_node(model, DIV, wrap_in_array(exp_nodes[i], sum_node), 2, (int[]){BATCH_SIZE, 1}, 2);
    }

    return softmax_nodes;
}

int create_cross_entropy_loss(Model *model, int *target_nodes, int *softmax_nodes, int num_outputs)
{
    int *log_nodes = malloc(num_outputs * sizeof(int));
    int *product_nodes = malloc(num_outputs * sizeof(int));

    for (int i = 0; i < num_outputs; i++)
    {
        int log_softmax = create_operation_node(model, LOG, wrap_value_in_array(softmax_nodes[i]), 1, (int[]){BATCH_SIZE, 1}, 2);
        log_nodes[i] = log_softmax;
        product_nodes[i] = create_operation_node(model, MULTIPLY, wrap_in_array(target_nodes[i], log_softmax), 2, (int[]){BATCH_SIZE, 1}, 2);
    }

    int sum_cross_entropy = create_operation_node(model, ADD, product_nodes, num_outputs, (int[]){BATCH_SIZE, 1}, 2);
    int neg_cross_entropy = create_operation_node(model, NEG, wrap_value_in_array(sum_cross_entropy), 1, (int[]){BATCH_SIZE, 1}, 2);
    return neg_cross_entropy;
}

int *create_multihead_attention_layer(Model *model, int *input_nodes, int num_inputs, int d_model, int num_heads, double dropout_rate)
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
            Q_weights[h][i] = create_value_node(model, 1, (int[]){BATCH_SIZE, 1}, 2);
            K_weights[h][i] = create_value_node(model, 1, (int[]){BATCH_SIZE, 1}, 2);
            V_weights[h][i] = create_value_node(model, 1, (int[]){BATCH_SIZE, 1}, 2);

            double weight_init = he_init(num_inputs);
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                set_node_value_by_position(model, Q_weights[h][i], (int[]){b, 0}, 2, weight_init);
                set_node_value_by_position(model, K_weights[h][i], (int[]){b, 0}, 2, weight_init);
                set_node_value_by_position(model, V_weights[h][i], (int[]){b, 0}, 2, weight_init);
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
                    int q_mul = create_operation_node(model, MULTIPLY, wrap_in_array(input_nodes[k], Q_weights[h][j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    int k_mul = create_operation_node(model, MULTIPLY, wrap_in_array(input_nodes[k], K_weights[h][j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    int v_mul = create_operation_node(model, MULTIPLY, wrap_in_array(input_nodes[k], V_weights[h][j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);

                    if (q_sum == -1)
                        q_sum = q_mul;
                    else
                        q_sum = create_operation_node(model, ADD, wrap_in_array(q_sum, q_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                    if (k_sum == -1)
                        k_sum = k_mul;
                    else
                        k_sum = create_operation_node(model, ADD, wrap_in_array(k_sum, k_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                    if (v_sum == -1)
                        v_sum = v_mul;
                    else
                        v_sum = create_operation_node(model, ADD, wrap_in_array(v_sum, v_mul), 2, (int[]){BATCH_SIZE, 1}, 2);
                }
                Q[h][i * head_dim + j] = q_sum;
                K[h][i * head_dim + j] = k_sum;
                V[h][i * head_dim + j] = v_sum;
                int q_dropout = create_operation_node(model, DROPOUT, wrap_value_in_array(Q[h][i * head_dim + j]), 1, (int[]){BATCH_SIZE, 1}, 2);
                model->nodes[q_dropout].dropout_rate = dropout_rate;
                Q[h][i * head_dim + j] = q_dropout;

                int k_dropout = create_operation_node(model, DROPOUT, wrap_value_in_array(K[h][i * head_dim + j]), 1, (int[]){BATCH_SIZE, 1}, 2);
                model->nodes[k_dropout].dropout_rate = dropout_rate;
                K[h][i * head_dim + j] = k_dropout;

                int v_dropout = create_operation_node(model, DROPOUT, wrap_value_in_array(V[h][i * head_dim + j]), 1, (int[]){BATCH_SIZE, 1}, 2);
                model->nodes[v_dropout].dropout_rate = dropout_rate;
                V[h][i * head_dim + j] = v_dropout;
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
                    int mul = create_operation_node(model, MULTIPLY, wrap_in_array(Q[h][q_idx], K[h][k_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    if (sum == -1)
                        sum = mul;
                    else
                        sum = create_operation_node(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
                }
                double scale = sqrt(head_dim);
                int scale_node = create_value_node(model, 0, (int[]){BATCH_SIZE, 1}, 2);
                for (int b = 0; b < BATCH_SIZE; b++)
                    set_node_value_by_position(model, scale_node, (int[]){b, 0}, 2, scale);

                attention_scores[i * seq_length + j] =
                    create_operation_node(model, DIV, wrap_in_array(sum, scale_node), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
        }

        int *dropout_scores = malloc(seq_length * seq_length * sizeof(int));
        for (int i = 0; i < seq_length * seq_length; i++)
        {
            dropout_scores[i] = create_operation_node(model, DROPOUT, wrap_value_in_array(attention_scores[i]), 1, (int[]){BATCH_SIZE, 1}, 2);
            model->nodes[dropout_scores[i]].dropout_rate = dropout_rate;
        }

        int *attention_weights = malloc(seq_length * seq_length * sizeof(int));
        for (int i = 0; i < seq_length; i++)
        {
            int *row_scores = &dropout_scores[i * seq_length];
            int *softmax_row = create_softmax_layer(model, row_scores, seq_length);
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
                    int mul = create_operation_node(model, MULTIPLY, wrap_in_array(attention_weights[i * seq_length + j], V[h][v_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    if (sum == -1)
                        sum = mul;
                    else
                        sum = create_operation_node(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
                }
                context[i * d_model + h * head_dim + k] = sum;
            }
        }

        free(attention_scores);
        free(attention_weights);
        free(dropout_scores);
    }

    free(Q_weights);
    free(K_weights);
    free(V_weights);
    free(Q);
    free(K);
    free(V);

    return context;
}

// not used, keeping to to compare
int *create_attention_layer(Model *model, int *input_nodes, int num_inputs, int d_model)
{
    int Q_weights[d_model * num_inputs];
    int K_weights[d_model * num_inputs];
    int V_weights[d_model * num_inputs];

    for (int i = 0; i < d_model * num_inputs; i++)
    {
        Q_weights[i] = create_value_node(model, 1, (int[]){BATCH_SIZE, 1}, 2);
        K_weights[i] = create_value_node(model, 1, (int[]){BATCH_SIZE, 1}, 2);
        V_weights[i] = create_value_node(model, 1, (int[]){BATCH_SIZE, 1}, 2);

        double weight_init = he_init(num_inputs);
        for (int b = 0; b < BATCH_SIZE; b++)
        {
            set_node_value_by_position(model, Q_weights[i], (int[]){b, 0}, 2, weight_init);
            set_node_value_by_position(model, K_weights[i], (int[]){b, 0}, 2, weight_init);
            set_node_value_by_position(model, V_weights[i], (int[]){b, 0}, 2, weight_init);
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
                int q_mul = create_operation_node(model, MULTIPLY, wrap_in_array(input_nodes[k], Q_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                int k_mul = create_operation_node(model, MULTIPLY, wrap_in_array(input_nodes[k], K_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                int v_mul = create_operation_node(model, MULTIPLY, wrap_in_array(input_nodes[k], V_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (q_sum == -1)
                    q_sum = q_mul;
                else
                    q_sum = create_operation_node(model, ADD, wrap_in_array(q_sum, q_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (k_sum == -1)
                    k_sum = k_mul;
                else
                    k_sum = create_operation_node(model, ADD, wrap_in_array(k_sum, k_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (v_sum == -1)
                    v_sum = v_mul;
                else
                    v_sum = create_operation_node(model, ADD, wrap_in_array(v_sum, v_mul), 2, (int[]){BATCH_SIZE, 1}, 2);
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
                int mul = create_operation_node(model, MULTIPLY, wrap_in_array(Q[q_idx], K[k_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                if (sum == -1)
                    sum = mul;
                else
                    sum = create_operation_node(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            double scale = sqrt(d_model);
            int scale_node = create_value_node(model, 0, (int[]){BATCH_SIZE, 1}, 2);
            for (int b = 0; b < BATCH_SIZE; b++)
                set_node_value_by_position(model, scale_node, (int[]){b, 0}, 2, scale);

            attention_scores[i * seq_length + j] =
                create_operation_node(model, DIV, wrap_in_array(sum, scale_node), 2, (int[]){BATCH_SIZE, 1}, 2);
        }
    }

    int *attention_weights = malloc(seq_length * seq_length * sizeof(int));
    for (int i = 0; i < seq_length; i++)
    {
        int *row_scores = &attention_scores[i * seq_length];
        int *softmax_row = create_softmax_layer(model, row_scores, seq_length);
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
                int mul = create_operation_node(model, MULTIPLY, wrap_in_array(attention_weights[i * seq_length + j], V[v_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                if (sum == -1)
                    sum = mul;
                else
                    sum = create_operation_node(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            context[i * d_model + k] = sum;
            // printf("Context: %d\n", context[i * d_model + k]);
        }
    }

    free(attention_scores);
    free(attention_weights);
    return context;
}

int *create_feedforward_network(Model *model, int *prev_layer_nodes, int prev_layer_size, int curr_layer_size, double dropout_rate)
{
    int *curr_layer_nodes = malloc(curr_layer_size * sizeof(int));

    for (int neuron = 0; neuron < curr_layer_size; neuron++)
    {
        int sum = -1;
        for (int prev = 0; prev < prev_layer_size; prev++)
        {
            int weight = create_value_node(model, 1, (int[]){BATCH_SIZE, 1}, 2);
            for (int b = 0; b < model->nodes[weight].size; b++)
            {
                double weight_init = he_init(prev_layer_size);
                set_node_value_by_position(model, weight, (int[]){b, 0}, 2, weight_init);
            }
            int mul = create_operation_node(model, MULTIPLY, wrap_in_array(prev_layer_nodes[prev], weight), 2, (int[]){BATCH_SIZE, 1}, 2);

            if (sum == -1)
            {
                sum = mul;
            }
            else
            {
                sum = create_operation_node(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
        }

        int bias = create_value_node(model, 1, (int[]){BATCH_SIZE, 1}, 2);
        int biased = create_operation_node(model, ADD, wrap_in_array(sum, bias), 2, (int[]){BATCH_SIZE, 1}, 2);

        int dropout_node = create_operation_node(model, DROPOUT, wrap_value_in_array(biased), 1, (int[]){BATCH_SIZE, 1}, 2);
        model->nodes[dropout_node].dropout_rate = dropout_rate;

        curr_layer_nodes[neuron] = create_operation_node(model, RELU, wrap_value_in_array(dropout_node), 1, (int[]){BATCH_SIZE, 1}, 2);
    }

    return curr_layer_nodes;
}

Model *build_model(int num_inputs, int num_outputs, int vocab_size, int embed_size, int num_heads, int num_blocks, int mlp_size, int attention_size, double dropout_rate)
{
    Model *model = (Model *)malloc(sizeof(Model));

    model->nodes = (Node *)malloc(MAX_NODES * sizeof(Node));

    int *curr_layer = NULL;

    curr_layer = malloc(num_inputs * sizeof(int));

    int *input_nodes = curr_layer;
    for (int i = 0; i < num_inputs; i++)
    {
        curr_layer[i] = create_value_node(model, 0, (int[]){BATCH_SIZE, 1}, 2);
    }

    // embedding layer (ff based)
    curr_layer = create_feedforward_network(model, curr_layer, num_inputs, embed_size, 0.0); // TODO: not doing dropouts here, but should i?

    // doing another mapping so that the symmetry is better to apply residuals
    curr_layer = create_feedforward_network(model, curr_layer, embed_size, mlp_size, dropout_rate); // TODO: not doing dropouts here, but should i?

    for (int i = 0; i < num_blocks; i++)
    {

        curr_layer = create_multihead_attention_layer(model, curr_layer, mlp_size, attention_size, num_heads, dropout_rate);



        // looks messy, can simplify
        int *ff_input = curr_layer;
        curr_layer = create_feedforward_network(model, curr_layer, mlp_size * attention_size, mlp_size, dropout_rate);
        int *residual_ff_output = malloc(mlp_size * sizeof(int));
        for (int k = 0; k < mlp_size; k++)
        {
            residual_ff_output[k] = create_operation_node(model, ADD, wrap_in_array(ff_input[k], curr_layer[k]), 2, (int[]){BATCH_SIZE, 1}, 2);
        }
        free(curr_layer);
        curr_layer = residual_ff_output;
    }

    curr_layer = create_feedforward_network(model, curr_layer, mlp_size, vocab_size, dropout_rate);

    int *output_nodes = curr_layer;
    int *softmax_nodes = create_softmax_layer(model, output_nodes, num_outputs);

    int *target_nodes = (int *)malloc(num_outputs * sizeof(int));
    for (int i = 0; i < num_outputs; i++)
    {
        target_nodes[i] = create_value_node(model, 0, (int[]){BATCH_SIZE, 1}, 2);
    }
    int loss_node = create_cross_entropy_loss(model, target_nodes, softmax_nodes, num_outputs);

    model->input_nodes = input_nodes;
    model->output_nodes = output_nodes;
    model->loss_node = loss_node;
    model->softmax_nodes = softmax_nodes;
    model->target_nodes = target_nodes;
    model->num_inputs = num_inputs;
    model->num_outputs = num_outputs;
    model->embed_size = embed_size;

    return model;
}

void inference(Model *model, const char *input_string, int num_inputs, int *index_to_token, int *token_to_index, int vocab_size, int data_length, BPEMerge *merges, int num_merges, double *positional_encoding, int loss_node, int num_outputs, int *softmax_nodes)
{
    int seq_len = num_inputs / vocab_size;
    int max_index = -1;

    double *sequence = preprocess_string(input_string, token_to_index, merges, num_merges, vocab_size, seq_len); // preprocess the input string to BPE tokens

    printf("\n%s", input_string);
    for (int p = 0; p < 50; p++)
    {
        if (max_index != -1)
        {
            for (int k = 0; k < (seq_len - 1); k++)
            {
                sequence[k] = sequence[k + 1];
            }
            sequence[seq_len - 1] = max_index;
        }

        for (int j = 0; j < num_inputs; j++)
        {
            double val = sequence[j / vocab_size] == j % vocab_size ? 1.0 : 0.0;
            val += positional_encoding[j];
            set_node_value_by_position(model, j, (int[]){0, 0}, 2, val);
        }

        compute_graph(model, loss_node);

        double temperature = 0.6;
        double softmax_values[num_outputs];
        double exp_sum = 0.0;

        for (int j = 0; j < num_outputs; j++)
        {
            double raw_value = get_node_value_by_position(model, softmax_nodes[j], (int[]){0, 0}, 2);
            softmax_values[j] = exp(raw_value / temperature);
            exp_sum += softmax_values[j];
        }

        for (int j = 0; j < num_outputs; j++)
        {
            softmax_values[j] /= exp_sum;
        }

        double cumulative_prob = 0.0;
        double random_value = (double)rand() / RAND_MAX;
        max_index = -1;

        for (int j = 0; j < num_outputs; j++)
        {
            cumulative_prob += softmax_values[j];
            if (random_value <= cumulative_prob)
            {
                max_index = j;
                break;
            }
        }

        // Recover the original tokens from the BPE tokens
        int num_tokens = 1;
        int temp[MAX_MERGES] = {0};
        temp[0] = index_to_token[max_index];
        recover_original_tokens(temp, &num_tokens, merges, num_merges, data_length);
        assert(num_tokens < MAX_MERGES);

        for (int x = 0; x < num_tokens; x++)
        {
            printf("%c", (char)temp[x]);
        }
    }
    printf("\n");

    free(sequence);
}

void train(Model *model, double **inputs, int labels[], int num_samples, double learning_rate, int *index_to_token, int *token_to_index, int vocab_size, int data_length, BPEMerge *merges, int num_merges, int seq_length)
{
    int num_outputs = model->num_outputs;
    int num_inputs = model->num_inputs;
    int embed_size = model->embed_size;
    int *input_nodes = model->input_nodes;
    int *output_nodes = model->output_nodes;
    int loss_node = model->loss_node;
    int *softmax_nodes = model->softmax_nodes;
    int *target_nodes = model->target_nodes;

    detect_orphans(model);
    // adam related
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    double *positional_encoding = generate_positional_encoding(seq_length, embed_size);

    srand(time(NULL));

    int EPOCHS = 10000;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        printf("Epoch %d\n", epoch + 1);
        double total_loss = 0.0;

        for (int i = 0; i + BATCH_SIZE < num_samples; i += BATCH_SIZE)
        {
            zerograd(model);

            for (int j = 0; j < num_inputs; j++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    double val = inputs[i + b][j / vocab_size] == j % vocab_size ? 1.0 : 0.0;
                    val += positional_encoding[j];
                    set_node_value_by_position(model, j, (int[]){b, 0}, 2,
                                               val);
                }
            }

            for (int l = 0; l < num_outputs; l++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {

                    if (l == labels[i + b])
                    {
                        set_node_value_by_position(model, target_nodes[l], (int[]){b, 0}, 2, 1);
                    }
                    else
                    {
                        set_node_value_by_position(model, target_nodes[l], (int[]){b, 0}, 2, 0);
                    }
                }
            }

            compute_graph(model, loss_node);

            for (int b = 0; b < BATCH_SIZE; b++)
            {
                total_loss += get_node_value_by_position(model, loss_node, (int[]){b, 0}, 2);
                model->nodes[loss_node].gradient[b] = 1.0;
            }

            compute_grad(model, loss_node);

            for (int j = 0; j < node_counter; j++)
            {
                if (model->nodes[j].learnable_param == 1)
                {

                    if (model->nodes[j].adam_m == NULL)
                    {
                        model->nodes[j].adam_m = (double *)calloc(model->nodes[j].size, sizeof(double));
                        model->nodes[j].adam_v = (double *)calloc(model->nodes[j].size, sizeof(double));
                    }

                    for (int b = 0; b < model->nodes[j].size; b++)
                    {
                        model->nodes[j].adam_m[b] = beta1 * model->nodes[j].adam_m[b] + (1.0 - beta1) * model->nodes[j].gradient[b];
                        model->nodes[j].adam_v[b] = beta2 * model->nodes[j].adam_v[b] + (1.0 - beta2) * model->nodes[j].gradient[b] * model->nodes[j].gradient[b];

                        double m_hat = model->nodes[j].adam_m[b] / (1.0 - pow(beta1, epoch + 1));
                        double v_hat = model->nodes[j].adam_v[b] / (1.0 - pow(beta2, epoch + 1));

                        set_node_value(model, j, b, get_node_value(model, j, b) - learning_rate * m_hat / (sqrt(v_hat) + epsilon));
                    }
                }
            }
            printf("Epoch %d, Batch %d/ %d \n", epoch + 1, i / BATCH_SIZE, num_samples / BATCH_SIZE);
        }

        inference(model, "Hello ", num_inputs, index_to_token, token_to_index, vocab_size, data_length, merges, num_merges, positional_encoding, loss_node, num_outputs, softmax_nodes);

        printf("Epoch %d, Avg. Loss: %f\n\n", epoch + 1, total_loss / num_samples);

        fflush(stdin);
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

    setbuf(stdout, NULL);

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

    int seq_len = 64;
    int num_samples = num_numbers - seq_len;

    if (num_samples > MAX_SAMPLES)
    {
        num_samples = MAX_SAMPLES;
    }

    int labels[num_samples];
    double **inputs = (double **)malloc(num_samples * sizeof(double *));
    for (int i = 0; i < num_samples; i++)
    {
        inputs[i] = (double *)malloc(seq_len * sizeof(double));
    }

    for (int i = 0; i < num_samples; i++)
    {
        for (int j = 0; j < seq_len; j++)
        {
            inputs[i][j] = (double)token_to_index[numbers[i + j]];
        }
        labels[i] = (double)token_to_index[numbers[i + seq_len]];
    }

    double learning_rate = 0.01;
    int num_inputs = vocab_size * seq_len;
    int embed_size = 8;
    int num_heads = 2;
    int num_outputs = vocab_size;
    int num_blocks = 4;
    int mlp_size = 4;
    int attention_size = 16;
    double dropout_rate = 0.1;

    Model *model = build_model(num_inputs, num_outputs, vocab_size, embed_size, num_heads, num_blocks, mlp_size, attention_size, dropout_rate);

    train(model, inputs, labels, num_samples, learning_rate, index_to_token, token_to_index, vocab_size, data_length, merges, num_merges, seq_len);

    for (int i = 0; i < num_samples; i++)
    {
        free(inputs[i]);
    }
    free(inputs);
    free(data);

    return 0;
}
