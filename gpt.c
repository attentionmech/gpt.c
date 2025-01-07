#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void model_0(FILE *file)
{
    int ch;
    while ((ch = fgetc(file)) != EOF)
    {
        putchar(ch);
    }
}

void model_1(FILE *file)
{
    int CHUNK_SIZE = 10;

    char buffer[CHUNK_SIZE + 1];
    int bytesRead;

    while ((bytesRead = fread(buffer, 1, CHUNK_SIZE, file)) > 0)
    {
        buffer[bytesRead] = '\0';
        printf("%s\n", buffer);
    }
}


void model_3(FILE *file) {
    int chunk_size = 100;
    char buffer[chunk_size + 1];
    int bytesRead;

    while ((bytesRead = fread(buffer, 1, chunk_size, file)) > 0) {
        buffer[bytesRead] = '\0';

        char word[chunk_size + 1];
        int word_index = 0;

        for (int i = 0; i <= bytesRead; i++) {
            if (buffer[i] == ' ' || buffer[i] == '\0') {
                if (word_index > 0) {
                    word[word_index] = '\0';
                    for (int j = 0, k = word_index - 1; j < k; j++, k--) {
                        char temp = word[j];
                        word[j] = word[k];
                        word[k] = temp;
                    }
                    printf("%s ", word);
                    word_index = 0;
                }
                if (buffer[i] == '\0') break;
            } else {
                word[word_index++] = buffer[i];
            }
        }
        printf("\n");
    }
}



void model_2(FILE *file) {
    int chunk_size = 100;
    char buffer[chunk_size + 1];
    int bytesRead;

    while ((bytesRead = fread(buffer, 1, chunk_size, file)) > 0) {
        buffer[bytesRead] = '\0';

        for (int i = 0, j = bytesRead - 1; i < j; i++, j--) {
            char temp = buffer[i];
            buffer[i] = buffer[j];
            buffer[j] = temp;
        }

        printf("%s\n", buffer);
    }
}


void model_4(FILE *file) {
    int chunk_size = 100;
    char buffer[chunk_size + 1];
    int bytesRead;
    int follow_up[256][256] = {0};

    while ((bytesRead = fread(buffer, 1, chunk_size, file)) > 0) {
        buffer[bytesRead] = '\0';
        for (int i = 0; i < bytesRead - 1; i++) {
            follow_up[(unsigned char)buffer[i]][(unsigned char)buffer[i + 1]]++;
        }
    }

    char current_char = buffer[0];
    printf("%c", current_char);

    for (int i = 0; i < 1000; i++) {
        int max_count = 0;
        char next_char = '\0';

        for (int j = 0; j < 256; j++) {
            if (follow_up[(unsigned char)current_char][j] > max_count) {
                max_count = follow_up[(unsigned char)current_char][j];
                next_char = (char)j;
            }
        }

        if (next_char == '\0') break;

        printf("%c", next_char);
        current_char = next_char;
    }

    printf("\n");
}



int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "%s <filename> <model_number>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (file == NULL)
    {
        perror("Error opening file");
        return 1;
    }


    model_4(file); //run latest model by default

    fclose(file);
    return 0;
}
