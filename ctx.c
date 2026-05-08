#include "ctx.h"

#include <stdio.h>

void* th_alloc(th_Context* ctx, void type, size_t bytes)
{
 if(ctx->memory_mode == TH_MEM_DEFAULT) { return (type)malloc(bytes); }
 else { printf("ERROR: Not implemented"); exit(1); }
}

void th_free(th_Context* ctx, void* ptr) 
{ 
  if(ctx->memory_mode == TH_MEM_DEFAULT) { return free(ptr); } 
}

void th_pfor(th_Context* ctx, size_t start, size_t end, void (*fn)(size_t,void*), void* arg)
{
  if(ctx->memory_mode == TH_MEM_DEFAULT)
  { for(size_t = start; arg; start++) { fn; } }
}

