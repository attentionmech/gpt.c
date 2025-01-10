#include <stdio.h>
#include <stdlib.h>
#include "chunked_array.h"

ChunkedArray *chunked_array_init(int estimated_size, int chunk_size) {
    if (chunk_size <= 0) {
        return NULL;
    }

    ChunkedArray *chunked_array = (ChunkedArray *)malloc(sizeof(ChunkedArray));
    chunked_array->chunk_size = chunk_size;
    chunked_array->total_chunks = 0;

    int estimated_chunks = (estimated_size + chunk_size - 1) / chunk_size;
    chunked_array->index_array_size = estimated_chunks;
    chunked_array->index_array = (Chunk **)malloc(chunked_array->index_array_size * sizeof(Chunk *));

    for (int i = 0; i < estimated_chunks; i++) {
        chunked_array->index_array[i] = NULL;
    }

    return chunked_array;
}

void expand_index_array(ChunkedArray *chunked_array) {
    int new_size = chunked_array->index_array_size * 2;
    chunked_array->index_array = (Chunk **)realloc(chunked_array->index_array, new_size * sizeof(Chunk *));
    chunked_array->index_array_size = new_size;
}

void chunked_array_add(ChunkedArray *chunked_array, int index, Value *value) {
    int chunk_index = index / chunked_array->chunk_size;
    int position_in_chunk = index % chunked_array->chunk_size;

    if (chunk_index >= chunked_array->index_array_size) {
        expand_index_array(chunked_array);
    }

    if (chunked_array->index_array[chunk_index] == NULL) {
        Chunk *new_chunk = (Chunk *)malloc(sizeof(Chunk));
        new_chunk->data = (Value *)malloc(chunked_array->chunk_size * sizeof(int));
        chunked_array->index_array[chunk_index] = new_chunk;
        chunked_array->total_chunks++;
    }

    chunked_array->index_array[chunk_index]->data[position_in_chunk] = *value;
}

Value* chunked_array_get(ChunkedArray *chunked_array, int index) {
    int chunk_index = index / chunked_array->chunk_size;
    int position_in_chunk = index % chunked_array->chunk_size;

    if (chunk_index >= chunked_array->total_chunks || chunked_array->index_array[chunk_index] == NULL) {
        fprintf(stderr, "Index out of bounds\n");
        return NULL;
    }

    return &(chunked_array->index_array[chunk_index]->data[position_in_chunk]);
}

void chunked_array_cleanup(ChunkedArray *chunked_array) {
    for (int i = 0; i < chunked_array->total_chunks; i++) {
        free(chunked_array->index_array[i]->data);
        free(chunked_array->index_array[i]);
    }
    free(chunked_array->index_array);
    free(chunked_array);
}
