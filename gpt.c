#include <stdio.h>
#include <stdlib.h>

void model_0(FILE *file) {
    int ch;
    while ((ch = fgetc(file)) != EOF) {
        putchar(ch);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "%s <filename> <model_number>\n", argv[0]);
        return 1;
    }

    FILE *file = fopen(argv[1], "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    int model_number = atoi(argv[2]);
    if (model_number == 0) {
        model_0(file);
    } else {
        fprintf(stderr, "Unknown model number\n");
    }

    fclose(file);
    return 0;
}

