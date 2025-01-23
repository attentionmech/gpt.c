#include "basicgrad.h"

extern int slot_counter;

#define BATCH_SIZE 10

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
                curr_layer_slots[i] = create_value_slot(0, (int[]){BATCH_SIZE, 1}, 2);
            }
        }
        else
        {
            for (int neuron = 0; neuron < layer_sizes[layer]; neuron++)
            {
                int sum = -1;
                for (int prev = 0; prev < layer_sizes[layer - 1]; prev++)
                {
                    int weight = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);
                    for (int b = 0; b < slots[weight].size; b++)
                    {
                        double weight_init = he_init(layer_sizes[layer - 1]);
                        set_slot_value_by_position(weight, (int[]){b,0},2, weight_init);
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

int* create_attention_layer(int* input_slots, int num_inputs, int d_model) {
    int Q_weights[d_model * num_inputs];
    int K_weights[d_model * num_inputs];
    int V_weights[d_model * num_inputs];

    for (int i = 0; i < d_model * num_inputs; i++) {
        Q_weights[i] = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);
        K_weights[i] = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);
        V_weights[i] = create_value_slot(1, (int[]){BATCH_SIZE, 1}, 2);

        double weight_init = he_init(num_inputs);
        for (int b = 0; b < BATCH_SIZE; b++) {
            set_slot_value_by_position(Q_weights[i], (int[]){b, 0}, 2, weight_init);
            set_slot_value_by_position(K_weights[i], (int[]){b, 0}, 2, weight_init);
            set_slot_value_by_position(V_weights[i], (int[]){b, 0}, 2, weight_init);
        }
    }

    int Q[num_inputs * d_model];
    int K[num_inputs * d_model];
    int V[num_inputs * d_model];

    for (int i = 0; i < num_inputs; i++) {
        for (int j = 0; j < d_model; j++) {
            int q_sum = -1, k_sum = -1, v_sum = -1;
            for (int k = 0; k < num_inputs; k++) {
                int q_mul = create_operation_slot(MULTIPLY, wrap_in_array(input_slots[k], Q_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                int k_mul = create_operation_slot(MULTIPLY, wrap_in_array(input_slots[k], K_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);
                int v_mul = create_operation_slot(MULTIPLY, wrap_in_array(input_slots[k], V_weights[j * num_inputs + k]), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (q_sum == -1) q_sum = q_mul;
                else q_sum = create_operation_slot(ADD, wrap_in_array(q_sum, q_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (k_sum == -1) k_sum = k_mul;
                else k_sum = create_operation_slot(ADD, wrap_in_array(k_sum, k_mul), 2, (int[]){BATCH_SIZE, 1}, 2);

                if (v_sum == -1) v_sum = v_mul;
                else v_sum = create_operation_slot(ADD, wrap_in_array(v_sum, v_mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            Q[i * d_model + j] = q_sum;
            K[i * d_model + j] = k_sum;
            V[i * d_model + j] = v_sum;
        }
    }

    int seq_length = num_inputs;
    int* attention_scores = malloc(seq_length * seq_length * sizeof(int));

    for (int i = 0; i < seq_length; i++) {
        for (int j = 0; j < seq_length; j++) {
            int sum = -1;
            for (int k = 0; k < d_model; k++) {
                int q_idx = i * d_model + k;
                int k_idx = j * d_model + k;
                int mul = create_operation_slot(MULTIPLY, wrap_in_array(Q[q_idx], K[k_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                if (sum == -1) sum = mul;
                else sum = create_operation_slot(ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            double scale = sqrt(d_model);
            int scale_slot = create_value_slot(0, (int[]){BATCH_SIZE, 1}, 2);
            for (int b = 0; b < BATCH_SIZE; b++)
                set_slot_value_by_position(scale_slot, (int[]){b, 0}, 2, scale);

            attention_scores[i * seq_length + j] =
                create_operation_slot(DIV, wrap_in_array(sum, scale_slot), 2, (int[]){BATCH_SIZE, 1}, 2);
        }
    }

    int* attention_weights = malloc(seq_length * seq_length * sizeof(int));
    for (int i = 0; i < seq_length; i++) {
        int* row_scores = &attention_scores[i * seq_length];
        int* softmax_row = create_softmax_layer(row_scores, seq_length);
        for (int j = 0; j < seq_length; j++) {
            attention_weights[i * seq_length + j] = softmax_row[j];
        }
    }

    int* context = malloc(seq_length * d_model * sizeof(int));
    for (int i = 0; i < seq_length; i++) {
        for (int k = 0; k < d_model; k++) {
            int sum = -1;
            for (int j = 0; j < seq_length; j++) {
                int v_idx = j * d_model + k;
                int mul = create_operation_slot(MULTIPLY, wrap_in_array(attention_weights[i * seq_length + j], V[v_idx]), 2, (int[]){BATCH_SIZE, 1}, 2);
                if (sum == -1) sum = mul;
                else sum = create_operation_slot(ADD, wrap_in_array(sum, mul), 2, (int[]){BATCH_SIZE, 1}, 2);
            }
            context[i * d_model + k] = sum;
        }
    }

    free(attention_scores);
    free(attention_weights);
    return context;
}

void train(double **inputs, int labels[], int num_samples, double learning_rate, int *layer_sizes, int num_layers, int *index_to_char, int vocab_size)
{

    int num_inputs = layer_sizes[0];
    int num_outputs = layer_sizes[num_layers - 1];

    int* prev_layer = NULL;
    int* curr_layer = NULL;

    for (int i = 0; i < num_layers; i++) {
        if (layer_sizes[i] == -1) {
            curr_layer = create_attention_layer(prev_layer, layer_sizes[i-1], 2);
        } else {
            int sizes[] = {layer_sizes[i]};
            curr_layer = create_feedforward_network(sizes, 1);
        }
        if (prev_layer) free(prev_layer);
        prev_layer = curr_layer;
    }


    int *output_slots = curr_layer;
    int *softmax_slots = create_softmax_layer(output_slots, num_outputs);

    int target_slots[num_outputs];
    for (int i = 0; i < num_outputs; i++)
    {
        target_slots[i] = create_value_slot(0, (int[]){BATCH_SIZE, 1}, 2);
    }

    int loss_slot = create_cross_entropy_loss(target_slots, softmax_slots, num_outputs);

    export_graph_to_dot("test.dot");

    srand(time(NULL));

    int EPOCHS = 1;

    for (int epoch = 0; epoch < EPOCHS; epoch++)
    {
        double total_loss = 0.0;

        for (int i = 0; i < num_samples; i += BATCH_SIZE)
        {

            zerograd();

            for (int k = 0; k < num_inputs; k++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    set_slot_value_by_position(k, (int[]){b,0},2, inputs[i + b][k]);
                }
            }

            for (int l = 1; l <= num_outputs; l++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    if (l == labels[i + b])
                    {
                        set_slot_value_by_position(target_slots[l - 1], (int[]){b,0},2, 1);
                    }
                    else
                    {
                        set_slot_value_by_position(target_slots[l - 1], (int[]){b,0},2, 0);
                    }
                }
            }

            compute_graph(loss_slot);

            for (int b = 0; b < BATCH_SIZE; b++)
            {
                total_loss += get_slot_value_by_position(loss_slot, (int[]){b,0},2);
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
                        set_slot_value_by_position(j, (int[]){b,0},2, get_slot_value_by_position(j, (int[]){b,0},2) - learning_rate * grad_sum / slots[j].size);
                    }
                }
            }
        }

        int seq_len = num_inputs / vocab_size;
        int max_index = -111;

        for (int p = 0; p < 50; p++)
        {
            if (max_index != -111)
            {
                for (int k = 0; k < (seq_len - 1); k++)
                {
                    set_slot_value_by_position(k, (int[]){0,0},2, inputs[0][k + 1]);
                }
                inputs[0][(seq_len - 1)] = max_index;
            }

            compute_graph(loss_slot);

            double temperature = 0.6;
            double softmax_values[num_outputs];
            double exp_sum = 0.0;

            for (int j = 0; j < num_outputs; j++)
            {
                double raw_value = get_slot_value_by_position(softmax_slots[j], (int[]){0,0},2);
                softmax_values[j] = exp(raw_value / temperature);
                exp_sum += softmax_values[j];
            }

            for (int j = 0; j < num_outputs; j++)
            {
                softmax_values[j] /= exp_sum;
            }

            double cumulative_prob = 0.0;
            double random_value = (double)rand() / RAND_MAX;
            int max_index = -111;

            for (int j = 0; j < num_outputs; j++)
            {
                cumulative_prob += softmax_values[j];
                if (random_value <= cumulative_prob)
                {
                    max_index = j;
                    break;
                }
            }

            printf("%c", (char)index_to_char[max_index]);
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
    fseek(file, 0, SEEK_SET);

    char *data = (char *)malloc(file_size + 1);
    int data_length = fread(data, sizeof(char), file_size, file);
    data[data_length] = '\0';
    fclose(file);

    int MAX_CHARACTERS = 256;
    int SEQUENCE_LENGTH = 4;

    int char_to_index[256];
    int index_to_char[MAX_CHARACTERS];
    for (int i = 0; i < 256; i++)
    {
        char_to_index[i] = -1;
    }

    int vocab_size = 0;
    for (int i = 0; i < data_length; i++)
    {
        unsigned char c = data[i];
        if (char_to_index[c] == -1)
        {
            char_to_index[c] = vocab_size;
            index_to_char[vocab_size] = c;
            vocab_size++;
        }
    }

    int input_size = SEQUENCE_LENGTH;
    int num_samples = data_length - SEQUENCE_LENGTH;

    if (num_samples > 10)
    {
        num_samples = 10;
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
            unsigned char c = data[i + j];

            inputs[i][j] = char_to_index[c];
        }

        unsigned char next_char = data[i + SEQUENCE_LENGTH];
        labels[i] = char_to_index[next_char];
    }

    double learning_rate = 0.01;
    int layer_sizes[] = {input_size, 2, -1 , 2, vocab_size};

    int num_layers = 5;

    train(inputs, labels, num_samples, learning_rate, layer_sizes, num_layers, index_to_char, vocab_size);

    return 0;
}
