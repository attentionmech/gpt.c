#ifndef CHUNKED_ARRAY_H
#define CHUNKED_ARRAY_H

#include "gradops.h"

typedef struct Chunk {
    Value *data;
} Chunk;

typedef struct ChunkedArray {
    Chunk **index_array;
    int index_array_size;
    int total_chunks;
    int chunk_size;
} ChunkedArray;

ChunkedArray *chunked_array_init(int estimated_size, int chunk_size);
void expand_index_array(ChunkedArray *chunked_array);
void chunked_array_add(ChunkedArray *chunked_array, int index, Value *value);
Value* chunked_array_get(ChunkedArray *chunked_array, int index);
void chunked_array_cleanup(ChunkedArray *chunked_array);

#endif // CHUNKED_ARRAY_H
