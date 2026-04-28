//////////////////////////////////////////////////////////////////////////////
// Thymos Context (c) 2023-2026 by Zachary R. James (@thymos)
// Public API for Thymos types, context (memory allocator + execution policy), err, vTable for core model structs
//////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

// INCLUDES //
#include <stddef.h>
#include <stdint.h> // for fixed-size types 
#include <stdbool.h>
#include <stdio.h>


// CONTEXTS & POLICIES // 

// idea:: 
// th_Allocator: memory space & lifetime
// th_ExecutionPolicy: who runs and how 
// th_Context := Allocator + ExecutionPolicy


// 3 layers of ctx: 
// 1. explicit server/model pass
// 2. thread-local

// to-do 
//th_Allocator
//th_ExecutionPolicy
//th_Ctx
//th_alloc
//th_scratch
//th_pfor


// monitors the system state (leveraged by thymos, thymos_server, & hpx:: later)
typedef struct 
{

} th_Resource; 


typedef enum
{
  TH_MEM_DEFAULT,     // std malloc/free
  TH_MEM_ARENA        // bump allocator for fast, contiguous allocs
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

} th_Allocator;


// EXECUTION CONTEXT //

typedef enum
{ 
  TH_EXEC_DEFAULT, // normal, no parallelization 
  // DO THESE LATER:
  TH_EXEC_CPU_PARALLEL, // cpu parallelization (normal single machine) 
  //TH_EXEC_MPI, // distrubuted
  //TH_EXEC_CUDA // cuda GPU parallelization
} th_ExecutionMode;

typedef struct
{
  void (*th_pfor)(size_t start, size_t end, void (*fn)(size_t, void*), void* arg); // for loop
} th_ExecutionPolicy;


// this is the context that is passed around and referenced for the program
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


// MODEL //

typedef struct th_Model th_Model; // forward-declare

// allows the quick structural definition of different ml models
typedef struct 
{
  th_Error (*forward)();
  th_Error (*backward)();
  th_Error (*train)();
  th_Error (*destroy)();
  //th_Error (*save)();
  //th_Error (*load)();
  //th_Error (*build_graph); // to be thought of later
} th_ModelVTable;

// MODEL MUST STORE ctx*
typedef struct th_Model
{
  const th_ModelVTable* vtable;
  th_Context* ctx;
  void* paramaters;
  //th_Model** children; // array for composites (ex layers)
  //size_t num_children;
  //void* graph; // for building hypergraph later maybe
} TH_Model;


// you know there is optimization structs and such


