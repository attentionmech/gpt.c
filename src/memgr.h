#ifndef MEMGR_H
#define MEMGR_H

#include "gradops.h"

typedef struct MLLNode {
    Value *v;             
    struct MLLNode *next; 
} MLLNode;

void mgr_init(void);
void mgr_track_value(Value *v);
void mgr_cleanup(void);

#endif
