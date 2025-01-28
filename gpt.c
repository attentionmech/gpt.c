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
    LAYER_ATTENTION,
    TRANSFORMER_BLOCK,
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
} Slot;

typedef struct
{
    Slot *slots;
    int *input_slots;
    int *output_slots;
    int *target_slots;
    int *softmax_slots;
    int loss_slot;
    int num_inputs;
    int num_outputs;
    int embed_size;
} Model;


int slot_counter = 0;
double **dependency_buffer;

// chunked array related stuff

typedef struct Chunk
{
    Slot **data;
} Chunk;

typedef struct ChunkedArray
{
    Chunk **index_array;
    int index_array_size;
    int total_chunks;
    int chunk_size;
} ChunkedArray;

ChunkedArray *chunked_array_init(int estimated_size, int chunk_size)
{
    if (chunk_size <= 0)
    {
        return NULL;
    }

    ChunkedArray *chunked_array = (ChunkedArray *)malloc(sizeof(ChunkedArray));
    chunked_array->chunk_size = chunk_size;
    chunked_array->total_chunks = 0;

    int estimated_chunks = (estimated_size + chunk_size - 1) / chunk_size;
    chunked_array->index_array_size = estimated_chunks;
    chunked_array->index_array = (Chunk **)malloc(chunked_array->index_array_size * sizeof(Chunk *));

    for (int i = 0; i < estimated_chunks; i++)
    {
        chunked_array->index_array[i] = NULL;
    }

    return chunked_array;
}

void expand_index_array(ChunkedArray *chunked_array)
{
    int new_size = chunked_array->index_array_size * 2;
    chunked_array->index_array = (Chunk **)realloc(chunked_array->index_array, new_size * sizeof(Chunk *));
    chunked_array->index_array_size = new_size;
}

void chunked_array_add(ChunkedArray *chunked_array, int index, Slot *value)
{
    int chunk_index = index / chunked_array->chunk_size;
    int position_in_chunk = index % chunked_array->chunk_size;

    if (chunk_index >= chunked_array->index_array_size)
    {
        expand_index_array(chunked_array);
    }

    if (chunked_array->index_array[chunk_index] == NULL)
    {
        Chunk *new_chunk = (Chunk *)malloc(sizeof(Chunk));
        new_chunk->data = (Slot **)calloc(chunked_array->chunk_size, sizeof(Slot *));

        for (int i = 0; i < chunked_array->chunk_size; i++)
        {
            new_chunk->data[i] = NULL;
        }

        chunked_array->index_array[chunk_index] = new_chunk;
        chunked_array->total_chunks++;
    }

    chunked_array->index_array[chunk_index]->data[position_in_chunk] = value;
}

Slot *chunked_array_get(ChunkedArray *chunked_array, int index)
{
    int chunk_index = index / chunked_array->chunk_size;
    int position_in_chunk = index % chunked_array->chunk_size;

    // printf("%d %d", chunk_index, position_in_chunk);

    if (chunk_index >= chunked_array->total_chunks || chunked_array->index_array[chunk_index] == NULL)
    {
        // fprintf(stdout, "Index out of bounds\n");
        return NULL;
    }

    return chunked_array->index_array[chunk_index]->data[position_in_chunk];
}

void chunked_array_cleanup(ChunkedArray *chunked_array)
{
    for (int i = 0; i < chunked_array->total_chunks; i++)
    {
        if (chunked_array->index_array[i] != NULL)
        { // Check if chunk exists
            free(chunked_array->index_array[i]->data);
            free(chunked_array->index_array[i]);
        }
    }
    free(chunked_array->index_array);
    free(chunked_array);
}

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
    bool *is_referenced = (bool *)malloc(MAX_SLOTS * sizeof(bool));
    for (int i = 0; i < MAX_SLOTS; i++)
    {
        is_referenced[i] = false;
    }

    for (int i = 0; i < slot_counter; i++)
    {
        Slot *s = &model->slots[i];
        for (int j = 0; j < s->num_dependencies; j++)
        {
            int dep_slot = s->dependencies[j];
            is_referenced[dep_slot] = true;
        }
    }

    printf("Orphan slots:\n");
    for (int i = 0; i < slot_counter; i++)
    {
        if (!is_referenced[i] && model->slots[i].num_dependencies > 0)
        {
            printf("Slot %d (Operation: %s) is an orphan.\n", i, get_operation_name(model->slots[i].operation));
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

    for (int i = 0; i < slot_counter; i++)
    {
        Slot *s = &model->slots[i];

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

int create_value_slot(Model *model, int learnable_param, int *shape, int num_dimensions)
{
    model->slots[slot_counter].num_dimensions = num_dimensions;
    model->slots[slot_counter].shape = (int *)malloc(num_dimensions * sizeof(int));
    for (int i = 0; i < num_dimensions; i++)
    {
        model->slots[slot_counter].shape[i] = shape[i];
    }

    model->slots[slot_counter].strides = (int *)malloc(num_dimensions * sizeof(int));
    model->slots[slot_counter].strides[num_dimensions - 1] = 1;
    for (int i = num_dimensions - 2; i >= 0; i--)
    {
        model->slots[slot_counter].strides[i] =
            model->slots[slot_counter].strides[i + 1] * shape[i + 1];
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

    model->slots[slot_counter].size = total_size;

    model->slots[slot_counter].value = (double *)malloc(total_size * sizeof(double));
    model->slots[slot_counter].gradient = (double *)malloc(total_size * sizeof(double));

    if (learnable_param)
    {
        for (int b = 0; b < model->slots[slot_counter].size; b++)
        {
            model->slots[slot_counter].value[b] = generate_normal(0.0, 1.0);
        }
    }

    model->slots[slot_counter].operation = PARAMETER;
    model->slots[slot_counter].num_dependencies = 0;
    model->slots[slot_counter].learnable_param = learnable_param;

    return increment_slot();
}

int create_operation_slot(Model *model, OperationType op, int *dep, int num_dependencies, int *shape, int num_dimensions)
{
    model->slots[slot_counter].operation = op;
    model->slots[slot_counter].dependencies = dep;
    model->slots[slot_counter].num_dependencies = num_dependencies;

    model->slots[slot_counter].num_dimensions = num_dimensions;
    model->slots[slot_counter].shape = (int *)malloc(num_dimensions * sizeof(int));
    for (int i = 0; i < num_dimensions; i++)
    {
        model->slots[slot_counter].shape[i] = shape[i];
    }

    model->slots[slot_counter].strides = (int *)malloc(num_dimensions * sizeof(int));
    model->slots[slot_counter].strides[num_dimensions - 1] = 1;
    for (int i = num_dimensions - 2; i >= 0; i--)
    {
        model->slots[slot_counter].strides[i] =
            model->slots[slot_counter].strides[i + 1] * shape[i + 1];
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

    model->slots[slot_counter].size = total_size;
    model->slots[slot_counter].value = (double *)malloc(total_size * sizeof(double));
    model->slots[slot_counter].gradient = (double *)malloc(total_size * sizeof(double));

    model->slots[slot_counter].learnable_param = 1;

    return increment_slot();
}

void set_slot_value_by_position(Model *model, int slot, int *position, int num_dimensions, double value)
{
    int pos = 0;
    for (int i = 0; i < num_dimensions - 1; i++)
    {
        pos += model->slots[slot].strides[i] * position[i];
    }
    pos += position[num_dimensions - 1];

    if (pos >= model->slots[slot].size)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    model->slots[slot].value[pos] = value;
}

void set_slot_value(Model *model, int slot, int b_index, double v)
{
    model->slots[slot].value[b_index] = v;
}

double get_slot_value(Model *model, int slot, int index)
{
    if (index >= model->slots[slot].size)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }
    return model->slots[slot].value[index];
}

double get_slot_value_by_position(Model *model, int slot, int *position, int num_dimensions)
{
    int pos = 0;
    for (int i = 0; i < num_dimensions - 1; i++)
    {
        pos += model->slots[slot].strides[i] * position[i];
    }
    pos += position[num_dimensions - 1];

    if (pos >= model->slots[slot].size)
    {
        fprintf(stderr, "Error: Index out of bounds\n");
        exit(EXIT_FAILURE);
    }

    return model->slots[slot].value[pos];
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

double *compute_graph(Model *model, int slot)
{
    Slot *s = &model->slots[slot];

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

        default:
            break;
        }
    }

    s->visited = 1;
    return s->value;
}

void compute_grad(Model *model, int slot)
{

    if (dependency_buffer == NULL)
    {
        dependency_buffer = (double **)malloc(MAX_ELEMENTS * sizeof(double *));
        for (int b = 0; b < MAX_ELEMENTS; b++)
        {
            dependency_buffer[b] = (double *)malloc(model->slots[slot].num_dependencies * sizeof(double));
        }
    }

    for (int curr = slot; curr >= 0; curr--)
    {
        Slot *s = &model->slots[curr];

        if (s->num_dependencies > 0)
        {

            for (int j = 0; j < s->num_dependencies; j++)
            {
                for (int b = 0; b < s->size; b++)
                {
                    dependency_buffer[b][j] = get_slot_value(model, s->dependencies[j], b);
                }
            }

            switch (s->operation)
            {
            case ADD:
                for (int i = 0; i < s->num_dependencies; i++)
                {
                    for (int b = 0; b < s->size; b++)
                    {
                        model->slots[s->dependencies[i]].gradient[b] += s->gradient[b];
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
                        model->slots[s->dependencies[i]].gradient[b] += s->gradient[b] * (product / dependency_buffer[b][i]);
                    }
                }
                break;

            case SUB:
                for (int b = 0; b < s->size; b++)
                {
                    model->slots[s->dependencies[0]].gradient[b] += s->gradient[b];
                    model->slots[s->dependencies[1]].gradient[b] -= s->gradient[b];
                }
                break;

            case POW2:
                for (int b = 0; b < s->size; b++)
                {
                    model->slots[s->dependencies[0]].gradient[b] += s->gradient[b] * 2.0 * dependency_buffer[b][0];
                }
                break;

            case SIGMOID:
                for (int b = 0; b < s->size; b++)
                {
                    model->slots[s->dependencies[0]].gradient[b] += s->gradient[b] * s->value[b] * (1.0 - s->value[b]);
                }
                break;

            case RELU:
                for (int b = 0; b < s->size; b++)
                {
                    if (dependency_buffer[b][0] > 0)
                    {
                        model->slots[s->dependencies[0]].gradient[b] += s->gradient[b];
                    }
                }
                break;

            case LEAKY_RELU:
                for (int b = 0; b < s->size; b++)
                {
                    double alpha = 0.01;
                    model->slots[s->dependencies[0]].gradient[b] += s->gradient[b] *
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
                    model->slots[s->dependencies[0]].gradient[b] += s->gradient[b] * derivative;
                }
                break;

            case EXP:
                for (int b = 0; b < s->size; b++)
                {
                    model->slots[s->dependencies[0]].gradient[b] += s->gradient[b] * s->value[b];
                }
                break;

            case NEG:
                for (int b = 0; b < s->size; b++)
                {

                    model->slots[s->dependencies[0]].gradient[b] += s->gradient[b] * -1.0;
                }
                break;

            case DIV:
                for (int b = 0; b < s->size; b++)
                {

                    model->slots[s->dependencies[0]].gradient[b] += s->gradient[b] / dependency_buffer[b][1];
                    model->slots[s->dependencies[1]].gradient[b] -= s->gradient[b] * dependency_buffer[b][0] / (dependency_buffer[b][1] * dependency_buffer[b][1]);
                }
                break;

            case LOG:
                for (int b = 0; b < s->size; b++)
                {

                    model->slots[s->dependencies[0]].gradient[b] += s->gradient[b] * (1.0 / dependency_buffer[b][0]);
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
    for (int j = 0; j < slot_counter; j++)
    {
        for (int b = 0; b < model->slots[j].size; b++)
        {
            model->slots[j].gradient[b] = 0.0;
            model->slots[j].visited = 0;
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

int *create_softmax_layer(Model *model, int *input_slots, int num_outputs)
{

    int *exp_slots = malloc(num_outputs * sizeof(int));

    for (int i = 0; i < num_outputs; i++)
    {
        exp_slots[i] = create_operation_slot(model, EXP, wrap_value_in_array(input_slots[i]), 1, (int[]){BATCH_SIZE, 1}, 2);
    }

    int sum_slot = exp_slots[0];
    for (int i = 1; i < num_outputs; i++)
    {
        sum_slot = create_operation_slot(model, ADD, wrap_in_array(sum_slot, exp_slots[i]), 2, (int[]){BATCH_SIZE, 1}, 2);
    }

    int *softmax_slots = malloc(num_outputs * sizeof(int));
    for (int i = 0; i < num_outputs; i++)
    {
        softmax_slots[i] = create_operation_slot(model, DIV, wrap_in_array(exp_slots[i], sum_slot), 2, (int[]){BATCH_SIZE, 1}, 2);
    }

    return softmax_slots;
}

int create_cross_entropy_loss(Model *model, int *target_slots, int *softmax_slots, int num_outputs)
{
    int *log_slots = malloc(num_outputs * sizeof(int));
    int *product_slots = malloc(num_outputs * sizeof(int));

    for (int i = 0; i < num_outputs; i++)
    {
        int log_softmax = create_operation_slot(model, LOG, wrap_value_in_array(softmax_slots[i]), 1, (int[]){BATCH_SIZE, 1}, 2);
        log_slots[i] = log_softmax;
        product_slots[i] = create_operation_slot(model, MULTIPLY, wrap_in_array(target_slots[i], log_softmax), 2, (int[]){BATCH_SIZE, 1}, 2);
    }

    int sum_cross_entropy = create_operation_slot(model, ADD, product_slots, num_outputs, (int[]){BATCH_SIZE, 1}, 2);
    int neg_cross_entropy = create_operation_slot(model, NEG, wrap_value_in_array(sum_cross_entropy), 1, (int[]){BATCH_SIZE, 1}, 2);
    return neg_cross_entropy;
}

int *create_multihead_attention_layer(Model *model, int *input_slots, int num_inputs, int d_model, int num_heads)
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
            Q_weights[h][i] = create_value_slot(model, 1, (int[]){BATCH_SIZE, 1}, 2);
            K_weights[h][i] = create_value_slot(model, 1, (int[]){BATCH_SIZE, 1}, 2);
            V_weights[h][i] = create_value_slot(model, 1, (int[]){BATCH_SIZE, 1}, 2);

            double weight_init = he_init(num_inputs);
            for (int b = 0; b < BATCH_SIZE; b++)
            {
                set_slot_value_by_position(model, Q_weights[h][i], (int[]){b, 0}, 2, weight_init);
                set_slot_value_by_position(model, K_weights[h][i], (int[]){b, 0}, 2, weight_init);
                set_slot_value_by_position(model, V_weights[h][i], (int[]){b, 0}, 2, weight_init);
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
                    int q_mul = create_operation_slot(model, MULTIPLY, wrap_in_array(input_slots[k], Q_weights[h][j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    int k_mul = create_operation_slot(model, MULTIPLY, wrap_in_array(input_slots[k], K_weights[h][j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    int v_mul = create_operation_slot(model, MULTIPLY, wrap_in_array(input_slots[k], V_weights[h][j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);

                    if (q_sum == -1)
                        q_sum = q_mul;
                    else
                        q_sum = create_operation_slot(model, ADD, wrap_in_array(q_sum, q_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                    if (k_sum == -1)
                        k_sum = k_mul;
                    else
                        k_sum = create_operation_slot(model, ADD, wrap_in_array(k_sum, k_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                    if (v_sum == -1)
                        v_sum = v_mul;
                    else
                        v_sum = create_operation_slot(model, ADD, wrap_in_array(v_sum, v_mul), 2, (int[]){BATCH_SIZE, 1}, 2);
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
                    int mul = create_operation_slot(model, MULTIPLY, wrap_in_array(Q[h][q_idx], K[h][k_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    if (sum == -1)
                        sum = mul;
                    else
                        sum = create_operation_slot(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
                }
                double scale = sqrt(head_dim);
                int scale_slot = create_value_slot(model, 0, (int[]){BATCH_SIZE, 1}, 2);
                for (int b = 0; b < BATCH_SIZE; b++)
                    set_slot_value_by_position(model, scale_slot, (int[]){b, 0}, 2, scale);

                attention_scores[i * seq_length + j] =
                    create_operation_slot(model, DIV, wrap_in_array(sum, scale_slot), 2, (int[]){BATCH_SIZE, 1}, 2);
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

        for (int i = 0; i < seq_length; i++)
        {
            for (int k = 0; k < head_dim; k++)
            {
                int sum = -1;
                for (int j = 0; j < seq_length; j++)
                {
                    int v_idx = j * head_dim + k;
                    int mul = create_operation_slot(model, MULTIPLY, wrap_in_array(attention_weights[i * seq_length + j], V[h][v_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                    if (sum == -1)
                        sum = mul;
                    else
                        sum = create_operation_slot(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
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

int *create_attention_layer(Model *model, int *input_slots, int num_inputs, int d_model)
{
    int Q_weights[d_model * num_inputs];
    int K_weights[d_model * num_inputs];
    int V_weights[d_model * num_inputs];

    for (int i = 0; i < d_model * num_inputs; i++)
    {
        Q_weights[i] = create_value_slot(model, 1, (int[]){BATCH_SIZE, 1}, 2);
        K_weights[i] = create_value_slot(model, 1, (int[]){BATCH_SIZE, 1}, 2);
        V_weights[i] = create_value_slot(model, 1, (int[]){BATCH_SIZE, 1}, 2);

        double weight_init = he_init(num_inputs);
        for (int b = 0; b < BATCH_SIZE; b++)
        {
            set_slot_value_by_position(model, Q_weights[i], (int[]){b, 0}, 2, weight_init);
            set_slot_value_by_position(model, K_weights[i], (int[]){b, 0}, 2, weight_init);
            set_slot_value_by_position(model, V_weights[i], (int[]){b, 0}, 2, weight_init);
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
                int q_mul = create_operation_slot(model, MULTIPLY, wrap_in_array(input_slots[k], Q_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                int k_mul = create_operation_slot(model, MULTIPLY, wrap_in_array(input_slots[k], K_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                int v_mul = create_operation_slot(model, MULTIPLY, wrap_in_array(input_slots[k], V_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (q_sum == -1)
                    q_sum = q_mul;
                else
                    q_sum = create_operation_slot(model, ADD, wrap_in_array(q_sum, q_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (k_sum == -1)
                    k_sum = k_mul;
                else
                    k_sum = create_operation_slot(model, ADD, wrap_in_array(k_sum, k_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (v_sum == -1)
                    v_sum = v_mul;
                else
                    v_sum = create_operation_slot(model, ADD, wrap_in_array(v_sum, v_mul), 2, (int[]){BATCH_SIZE, 1}, 2);
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
                int mul = create_operation_slot(model, MULTIPLY, wrap_in_array(Q[q_idx], K[k_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                if (sum == -1)
                    sum = mul;
                else
                    sum = create_operation_slot(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            double scale = sqrt(d_model);
            int scale_slot = create_value_slot(model, 0, (int[]){BATCH_SIZE, 1}, 2);
            for (int b = 0; b < BATCH_SIZE; b++)
                set_slot_value_by_position(model, scale_slot, (int[]){b, 0}, 2, scale);

            attention_scores[i * seq_length + j] =
                create_operation_slot(model, DIV, wrap_in_array(sum, scale_slot), 2, (int[]){BATCH_SIZE, 1}, 2);
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
                int mul = create_operation_slot(model, MULTIPLY, wrap_in_array(attention_weights[i * seq_length + j], V[v_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                if (sum == -1)
                    sum = mul;
                else
                    sum = create_operation_slot(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            context[i * d_model + k] = sum;
            // printf("Context: %d\n", context[i * d_model + k]);
        }
    }

    free(attention_scores);
    free(attention_weights);
    return context;
}

int *create_feedforward_network(Model *model, int *prev_layer_slots, int prev_layer_size, int curr_layer_size)
{
    int *curr_layer_slots = malloc(curr_layer_size * sizeof(int));

    for (int neuron = 0; neuron < curr_layer_size; neuron++)
    {
        int sum = -1;
        for (int prev = 0; prev < prev_layer_size; prev++)
        {
            int weight = create_value_slot(model, 1, (int[]){BATCH_SIZE, 1}, 2);
            for (int b = 0; b < model->slots[weight].size; b++)
            {
                double weight_init = he_init(prev_layer_size);
                set_slot_value_by_position(model, weight, (int[]){b, 0}, 2, weight_init);
            }
            int mul = create_operation_slot(model, MULTIPLY, wrap_in_array(prev_layer_slots[prev], weight), 2, (int[]){BATCH_SIZE, 1}, 2);

            if (sum == -1)
            {
                sum = mul;
            }
            else
            {
                sum = create_operation_slot(model, ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
        }

        int bias = create_value_slot(model, 1, (int[]){BATCH_SIZE, 1}, 2);
        int biased = create_operation_slot(model, ADD, wrap_in_array(sum, bias), 2, (int[]){BATCH_SIZE, 1}, 2);
        curr_layer_slots[neuron] = create_operation_slot(model, RELU, wrap_value_in_array(biased), 1, (int[]){BATCH_SIZE, 1}, 2);
    }

    return curr_layer_slots;
}

Model *build_model(int num_inputs, int num_outputs, int embed_size, int layer_sizes[], ComponentType layer_types[], int num_layers, int num_heads)
{
    Model *model = (Model *)malloc(sizeof(Model));

    model->slots = (Slot *)malloc(MAX_SLOTS * sizeof(Slot));

    int *prev_layer = NULL;
    int *curr_layer = NULL;

    prev_layer = malloc(num_inputs * sizeof(int));

    int *input_slots = prev_layer;
    for (int i = 0; i < num_inputs; i++)
    {
        prev_layer[i] = create_value_slot(model, 0, (int[]){BATCH_SIZE, 1}, 2);
    }

    prev_layer = create_feedforward_network(model, prev_layer, num_inputs, embed_size);

    for (int i = 1; i < num_layers; i++)
    {

        if (layer_types[i] == LAYER_ATTENTION)
        {
            curr_layer = create_multihead_attention_layer(model, prev_layer, layer_sizes[i - 1], layer_sizes[i], num_heads);
        }
        else if (layer_types[i] == LAYER_FEEDFORWARD)
        {
            if (layer_types[i - 1] == LAYER_ATTENTION)
            {
                curr_layer = create_feedforward_network(model, curr_layer, layer_sizes[i - 1] * layer_sizes[i - 2], layer_sizes[i]);
            }
            else
            {
                curr_layer = create_feedforward_network(model, prev_layer, layer_sizes[i - 1], layer_sizes[i]);
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
    int *softmax_slots = create_softmax_layer(model, output_slots, num_outputs);

    int *target_slots = (int *)malloc(num_outputs * sizeof(int));
    for (int i = 0; i < num_outputs; i++)
    {
        target_slots[i] = create_value_slot(model, 0, (int[]){BATCH_SIZE, 1}, 2);
    }
    int loss_slot = create_cross_entropy_loss(model, target_slots, softmax_slots, num_outputs);

    model->input_slots = input_slots;
    model->output_slots = output_slots;
    model->loss_slot = loss_slot;
    model->softmax_slots = softmax_slots;
    model->target_slots = target_slots;
    model->num_inputs = num_inputs;
    model->num_outputs = num_outputs;
    model->embed_size = embed_size;

    return model;
}

void train(Model *model, double **inputs, int labels[], int num_samples, double learning_rate, int *index_to_char, int vocab_size, int data_length, BPEMerge *merges, int num_merges, int seq_length)
{
    int num_outputs = model->num_outputs;
    int num_inputs = model->num_inputs;
    int embed_size = model->embed_size;
    int *input_slots = model->input_slots;
    int *output_slots = model->output_slots;
    int loss_slot = model->loss_slot;
    int *softmax_slots = model->softmax_slots;
    int *target_slots = model->target_slots;

    detect_orphans(model);
    // adam related
    double beta1 = 0.9;
    double beta2 = 0.999;
    double epsilon = 1e-8;

    double *positional_encoding = generate_positional_encoding(seq_length, embed_size);

    srand(time(NULL));

    int EPOCHS = 1000;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
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
                    set_slot_value_by_position(model, j, (int[]){b, 0}, 2,
                                               val);
                }
            }

            for (int l = 0; l < num_outputs; l++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {

                    if (l == labels[i + b])
                    {
                        set_slot_value_by_position(model, target_slots[l], (int[]){b, 0}, 2, 1);
                    }
                    else
                    {
                        set_slot_value_by_position(model, target_slots[l], (int[]){b, 0}, 2, 0);
                    }
                }
            }

            compute_graph(model, loss_slot);

            for (int b = 0; b < BATCH_SIZE; b++)
            {
                total_loss += get_slot_value_by_position(model, loss_slot, (int[]){b, 0}, 2);
                model->slots[loss_slot].gradient[b] = 1.0;
            }

            compute_grad(model, loss_slot);

            for (int j = 0; j < slot_counter; j++)
            {
                if (model->slots[j].learnable_param == 1)
                {

                    if (model->slots[j].adam_m == NULL)
                    {
                        model->slots[j].adam_m = (double *)calloc(model->slots[j].size, sizeof(double));
                        model->slots[j].adam_v = (double *)calloc(model->slots[j].size, sizeof(double));
                    }

                    for (int b = 0; b < model->slots[j].size; b++)
                    {
                        model->slots[j].adam_m[b] = beta1 * model->slots[j].adam_m[b] + (1.0 - beta1) * model->slots[j].gradient[b];
                        model->slots[j].adam_v[b] = beta2 * model->slots[j].adam_v[b] + (1.0 - beta2) * model->slots[j].gradient[b] * model->slots[j].gradient[b];

                        double m_hat = model->slots[j].adam_m[b] / (1.0 - pow(beta1, epoch + 1));
                        double v_hat = model->slots[j].adam_v[b] / (1.0 - pow(beta2, epoch + 1));

                        set_slot_value(model, j, b, get_slot_value(model, j, b) - learning_rate * m_hat / (sqrt(v_hat) + epsilon));
                    }
                }
            }
        }

        int seq_len = num_inputs / vocab_size;
        int max_index = -1;

        for (int p = 0; p < 50; p++)
        {
            if (max_index != -1)
            {
                for (int k = 0; k < (seq_len - 1); k++)
                {
                    inputs[0][k] = inputs[0][k + 1];
                }
                inputs[0][seq_len - 1] = max_index;
            }

            for (int j = 0; j < num_inputs; j++)
            {
                double val = inputs[0][j / vocab_size] == j % vocab_size ? 1.0 : 0.0;
                val += positional_encoding[j];
                set_slot_value_by_position(model, j, (int[]){0, 0}, 2, val);
            }

            compute_graph(model, loss_slot);

            double temperature = 0.6;
            double softmax_values[num_outputs];
            double exp_sum = 0.0;

            for (int j = 0; j < num_outputs; j++)
            {
                double raw_value = get_slot_value_by_position(model, softmax_slots[j], (int[]){0, 0}, 2);
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
            temp[0] = index_to_char[max_index];
            recover_original_tokens(temp, &num_tokens, merges, num_merges, data_length);
            assert(num_tokens < MAX_MERGES);

            for (int x = 0; x < num_tokens; x++)
            {
                printf("%c", (char)temp[x]);
            }
        }
        printf("\n");

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

    int seq_len = 4;
    int num_samples = num_numbers - seq_len;

    if (num_samples > 100)
    {
        num_samples = 100;
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
    int embed_size = 32;
    int num_layers = 5;
    int num_heads = 2;
    int num_outputs = vocab_size;

    ComponentType layer_types[] = {LAYER_FEEDFORWARD, LAYER_FEEDFORWARD, LAYER_ATTENTION, LAYER_FEEDFORWARD, LAYER_FEEDFORWARD};
    int layer_sizes[] = {embed_size, 4, 4, 4, vocab_size};

    Model *model = build_model(num_inputs, num_outputs, embed_size, layer_sizes, layer_types, num_layers, num_heads);

    train(model, inputs, labels, num_samples, learning_rate, index_to_token, vocab_size, data_length, merges, num_merges, seq_len);

    for (int i = 0; i < num_samples; i++)
    {
        free(inputs[i]);
    }
    free(inputs);
    free(data);

    return 0;
}
