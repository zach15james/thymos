//////////////////////////////////////////////////////////////////////////////
// Thymos Context (c) 2023-2026 by Zachary R. James (@thymos)
// Public API for Thymos types, context (memory allocator + execution policy), err, vTable for core model structs
//////////////////////////////////////////////////////////////////////////////
#pragma once
#include <stdlib.h>

// INCLUDES //
#include <stddef.h>
#include <stdint.h> // for fixed-size types 
#include <stdbool.h>
#include <stdio.h>

#include "prelude.h"

typedef struct {} th_Resource;  // system state (not sure if i need this)

typedef enum
{
  TH_MEM_DEFAULT,     // std malloc/free
  TH_MEM_ARENA        // bump allocator for fast, contiguous allocs, free is nop
} th_MemoryMode;

typedef struct
{
  void* (*th_alloc)(size_t bytes); // same format as std malloc
  void (*th_free)(void* ptr); // 
  // NOTE: should I define init/destroy/realloc/variations/&c ?????

  struct 
  {
    uint8_t* base; // start of the arena (base of the data)
    // NOTE: should these still be size_t ????
    size_t capacity;
    size_t used;
  } arena;

} th_Allocator; // memory space + lifetime

typedef enum
{ 
  TH_EXEC_DEFAULT, // normal, no parallelization 
  TH_EXEC_CPU_PARALLEL, // cpu parallelization (normal single machine) 
  TH_EXEC_MPI, // distrubuted
  TH_EXEC_CUDA // cuda GPU parallelization
} th_ExecutionMode;


typedef struct
{ // all mpi,cuda, threadpools, &c need
  void (*th_pfor)(size_t start, size_t end, void (*fn)(size_t, void*), void* arg);
} th_ExecutionPolicy; // who runs & how


typedef struct
{
  th_Allocator allocator;
  th_ExecutionPolicy policy;

  th_ExecutionMode execution_mode;
  th_MemoryMode memory_mode;

  th_Error last_error; // for debugging and states
} th_Context;

extern th_Context th_default_ctx; // global context to be referenced...

// setup functions
th_Error th_ctx_init(th_Context* ctx); // defaults to std malloc & single-threaded / OS-default???
th_Error th_ctx_destroy(th_Context* ctx);

// th_MemoryMode enum defs // 
th_Error th_ctx_set_memory_default(th_Context* ctx); // 
th_Error th_ctx_set_memory_arena(th_Context* ctx, size_t capacity); // i guess max capacity is obvious
th_Error th_ctx_arena_reset(th_Context* ctx); // resets arena (ie all sub-frees are freed) 

// the alloc & free _fn are defined by the user (in a custom way, while the th_allocate and th_free are whatever is set and used generically thoughout the code)
// th_MemoryMode enum defs // 

th_Error th_ctx_set_allocator(th_Context* ctx, void* (*alloc_fn)(size_t), void (*free_fn)(void*));

// execution setup
// th_ExecutionMode enum defs //
th_Error th_ctx_set_exec_default(th_Context* ctx);
th_Error th_ctx_set_exec_parallel(th_Context* ctx);
//th_Error th_ctx_set_exec_cuda(th_Context* ctx);
//th_Error th_ctx_set_exec_mpi(th_Context* ctx);
// th_ExecutionMode enum defs //

// core helper methods
// in allocator: 
void* th_alloc(th_Context* ctx, size_t bytes);
void th_free(th_Context* ctx, void* ptr);
// in exectuion policy
void th_pfor(th_Context* ctx, size_t start, size_t end, void (*fn)(size_t, void*), void* arg);
