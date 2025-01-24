#include "basicgrad.h"

#define MAX_ELEMENTS 1000000 // maximum elements in a single tensor

int slot_counter = 0;

double **dependency_buffer;

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

void export_graph_to_dot(const char *filename) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Error opening file for writing Graphviz DOT file.\n");
        return;
    }

    fprintf(file, "digraph ComputationalGraph {\n");
    fprintf(file, "    rankdir=LR; // Left-to-right graph layout\n");
    fprintf(file, "    node [shape=record, style=filled];\n");

    for (int i = 0; i < slot_counter; i++) {
        Slot *s = &slots[i];

        fprintf(file, "    slot_%d [label=\"{%d | {", i, i);

        fprintf(file, "Op: %s", get_operation_name(s->operation));

        if (s->num_dimensions > 0) {
            fprintf(file, " | Shape: [");
            for (int d = 0; d < s->num_dimensions; d++) {
                fprintf(file, "%d", s->shape[d]);
                if (d < s->num_dimensions - 1) fprintf(file, ", ");
            }
            fprintf(file, "]");
        }

        if (s->size > 0) {
            fprintf(file, " | Val: %.2f", s->value[0]);
            fprintf(file, " | Grad: %.2f", s->gradient[0]);
        }

        if (s->learnable_param) {
            fprintf(file, " | Learnable");
        }

        fprintf(file, "}}\", fillcolor=");

        if (s->operation == PARAMETER) {
            fprintf(file, "lightgreen");
        } else if (s->num_dependencies == 0) {
            fprintf(file, "lightblue");
        } else {
            fprintf(file, "lightpink");
        }

        fprintf(file, "];\n");

        for (int j = 0; j < s->num_dependencies; j++) {
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

double generate_normal(double mean, double stddev) {
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

    if(learnable_param){
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

void set_slot_value_by_position(int slot, int* position, int num_dimensions, double value)
{
    int pos = 0;
    for (int i = 0; i < num_dimensions-1; i++)
    {
        pos += slots[slot].strides[i]*position[i];
    }
    pos += position[num_dimensions-1];


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

double get_slot_value_by_position(int slot, int* position, int num_dimensions)
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