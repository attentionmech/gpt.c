#include <stdio.h>
#include <math.h>

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

int create_value_slot(double value, int learnable_param)
{
    slots[slot_count].value = value;
    slots[slot_count].gradient = 0.0;
    slots[slot_count].operation = PARAMETER;
    slots[slot_count].num_dependencies = 0;
    slots[slot_count].learnable_param = learnable_param;
    return slot_count++;
}

int create_operation_slot(OperationType op, int dep1, int dep2)
{
    slots[slot_count].operation = op;
    slots[slot_count].dependencies[0] = dep1;
    slots[slot_count].dependencies[1] = dep2;
    slots[slot_count].num_dependencies = 2;
    slots[slot_count].gradient = 0.0;
    slots[slot_count].learnable_param = 0;
    return slot_count++;
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

void train(double inputs[][2], int labels[], int num_samples, double learning_rate)
{
    int input1_slot = create_value_slot(0.0, 0);
    int input2_slot = create_value_slot(0.0, 0);

    
    int weight1_slot = create_value_slot(0.1, 1);
    int weight2_slot = create_value_slot(0.1, 1);
    int weight3_slot = create_value_slot(0.1, 1);
    int bias1_slot = create_value_slot(0.0, 1);
    int bias2_slot = create_value_slot(0.0, 1);

    
    int hidden_output1_slot = create_operation_slot(MULTIPLY, input1_slot, weight1_slot);
    int hidden_output2_slot = create_operation_slot(MULTIPLY, input2_slot, weight2_slot);
    int hidden_output1_biased_slot = create_operation_slot(ADD, hidden_output1_slot, bias1_slot);
    int hidden_output2_biased_slot = create_operation_slot(ADD, hidden_output2_slot, bias2_slot);

    
    int hidden1_sigmoid_slot = create_operation_slot(SIGMOID, hidden_output1_biased_slot, 0);
    int hidden2_sigmoid_slot = create_operation_slot(SIGMOID, hidden_output2_biased_slot, 0);

    
    int output_slot = create_operation_slot(MULTIPLY, hidden1_sigmoid_slot, weight3_slot);
    output_slot = create_operation_slot(ADD, output_slot, hidden2_sigmoid_slot);

    
    int target_slot = create_value_slot(0.0, 0);
    int loss_slot = create_operation_slot(MSE, output_slot, target_slot);

    for (int epoch = 0; epoch < 10000; epoch++)
    { 
        double predictions[num_samples];

        

        for (int i = 0; i < num_samples; i++)
        {

            for (int i = 0; i < slot_count; i++)
            {
                slots[i].gradient = 0.0;
            }

            
            set_value_slot(input1_slot, inputs[i][0]);
            set_value_slot(input2_slot, inputs[i][1]);
            set_value_slot(target_slot, labels[i]);

            
            compute_graph(loss_slot);
            predictions[i] = get_value_slot(output_slot);

            
            slots[loss_slot].gradient = 1.0;
            compute_grad(loss_slot);

            
            for (int j = 0; j < slot_count; j++)
            {
                if (slots[j].learnable_param == 1)
                {
                    printf("Slot %d gradient: %f\n", j, slots[j].gradient);
                    set_value_slot(j, get_value_slot(j) - learning_rate * slots[j].gradient);
                }
            }
        }

        double loss = calculate_loss(predictions, labels, num_samples);
        printf("Epoch %d Loss: %f\n", epoch + 1, loss);
    }

    
    printf("\nInference Results:\n");
    for (int i = 0; i < num_samples; i++)
    {
        set_value_slot(input1_slot, inputs[i][0]);
        set_value_slot(input2_slot, inputs[i][1]);

        
        compute_graph(output_slot);
        printf("Input: (%.1f, %.1f) => Predicted Output: %.4f\n", inputs[i][0], inputs[i][1], get_value_slot(output_slot));
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
