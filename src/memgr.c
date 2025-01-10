#include <stdio.h>
#include <stdlib.h>
#include "memgr.h"

static MLLNode *temp_list = NULL;

void mgr_init(void) {
    temp_list = NULL;
}

void mgr_track_value(Value *v) {
    MLLNode *new_node = (MLLNode *)malloc(sizeof(MLLNode));
    if (new_node == NULL) {
        fprintf(stderr, "Memory allocation failed for MLLNode\n");
        return;
    }

    new_node->v = v;
    new_node->next = temp_list;
    temp_list = new_node;
}

void mgr_cleanup(void) {
    MLLNode *current = temp_list;
    MLLNode *next_node;

    // int length=0 ;
    while (current != NULL) {
        free(current->v); 
        next_node = current->next;
        free(current);
        current = next_node;
        // length++;
    }
    // printf("cleaned up: %d\n", length);

    temp_list = NULL; 
}

