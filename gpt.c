#include "basicgrad.h"

extern int slot_counter;

void train(double **inputs, int labels[], int num_samples, double learning_rate, int *layer_sizes, int num_layers, int *index_to_char, int vocab_size)
{

    int num_inputs = layer_sizes[0];
    int num_outputs = layer_sizes[num_layers - 1];

    int target_slots[num_outputs];

    int *output_slots = create_feedforward_network(layer_sizes, num_layers);
    int *softmax_slots = create_softmax_layer(output_slots, num_outputs);

    for (int i = 0; i < num_outputs; i++)
    {
        target_slots[i] = create_value_slot(0);
    }

    int loss_slot = create_cross_entropy_loss(target_slots, softmax_slots, num_outputs);

    srand(time(NULL));

    

    int EPOCHS = 10000;

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
                    set_slot_value(k, b, inputs[i + b][k]);
                }
            }

            for (int l = 1; l <= num_outputs; l++)
            {
                for (int b = 0; b < BATCH_SIZE; b++)
                {
                    if (l == labels[i + b])
                    {
                        set_slot_value(target_slots[l - 1], b, 1);
                    }
                    else
                    {
                        set_slot_value(target_slots[l - 1], b, 0);
                    }
                }
            }

            compute_graph(loss_slot);

            for (int b = 0; b < BATCH_SIZE; b++)
            {
                total_loss += get_slot_value(loss_slot, b);

                slots[loss_slot].gradient[b] = 1.0;
            }

            compute_grad(loss_slot);
            
            for (int j = 0; j < slot_counter; j++)
            {
                if (slots[j].learnable_param == 1)
                {
                    double grad_sum = 0.0;
                    for (int b = 0; b < BATCH_SIZE; b++)
                    {
                        grad_sum += slots[j].gradient[b];
                    }

                    for (int b = 0; b < BATCH_SIZE; b++)
                    {
                        set_slot_value(j, b, get_slot_value(j, b) - learning_rate * grad_sum / BATCH_SIZE);
                    }
                }
            }
        }

        int seq_len = num_inputs / vocab_size;
        int max_index = -111;
        // printf("\n__________________\n");
        // for (int k = 0; k < num_inputs; k++)
        // {
        //     set_slot_value(k, 0, inputs[0][k]);
        //     printf("%c", index_to_char[(int)inputs[0][k]]);
        // }
        // printf("\n__________________\n");

        for (int p = 0; p < 50; p++)
        {
            if (max_index != -111)
            {
                for (int k = 0; k < (seq_len - 1); k++)
                {
                    set_slot_value(k, 0, inputs[0][k + 1]);
                }
                inputs[0][(seq_len - 1)] = max_index;
            }

            compute_graph(loss_slot);

            double temperature = 0.6;
            double softmax_values[num_outputs];
            double exp_sum = 0.0;

            for (int j = 0; j < num_outputs; j++)
            {
                double raw_value = get_slot_value(softmax_slots[j], 0);
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

    if (num_samples > 1000)
    {
        num_samples = 1000;
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
    int layer_sizes[] = {input_size, 64, vocab_size};

    int num_layers = 3;

    train(inputs, labels, num_samples, learning_rate, layer_sizes, num_layers, index_to_char, vocab_size);

    return 0;
}
