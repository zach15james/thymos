//////////////////////////////////////////////////////////////////////////////
// Thymos Applied Category Theory (c) 2025-6 by Zachary R. James (@thymos)
// ACT-inspired library structs and methods allowing for a hypergraphic 
// representation and interpretability. 
// This will be eventually be used as a graphical representation --> ML compiler
// Citation: Spivak & Fong's text & YT Lecture Series  
//
// Overall goal: have an act representation of the model / library in the 
//              in the form of a hypergraph where i have views to:
//              (i) visualize (ports, hyperedges)
//              (ii) execute (linearized topo order + memory plan) 
//              (iii) backend-delegation (subgraphs to ORT/TRT)
// will need morphism, port, cat, hypergraph, & potnetial compiler stubs (in future)
//////////////////////////////////////////////////////////////////////////////
#if ! defined(__THYMOS_ACT_H__)
#define __THYMOS_ACT_H__

// INCLUDES //
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "ml.h"

// forward-declaration
typedef struct th_Category th_Category;

// stable ids
typedef uint32_t th_obj_id;
typedef uint32_t th_mor_id;
typedef uint32_t th_cat_id;
typedef uint32_t th_wire_id;

typedef enum
{
  TH_KIN_TENSOR,
  TH_KIN_SCALAR,
  TH_KIN_STRING,
  TH_KIN_BYTES,
  TH_KIN_JSON,
  TH_KIN_HANDLE
} th_Kind; 

typedef enum
{
  TH_TD_INVALID = 0,
  TH_TD_F16,
  TH_TD_F32,
  TH_TD_F64,
  TH_TD_I8,
  TH_TD_I32,
  TH_TD_I64,
  TH_TD_U8
} th_TypeDescriptor;


// NOTE: this is a type descriptor, NOT arb. data
// - unique integer ID
// - type: data type + rank + dims + layout
// - (optional) storage binding
// - (optional) name / descriptor (or should that be inferred by what it is pointed to?)
// - (optional) constrints (ex: device preferred, contiguity, precision)
typedef struct th_Object
{
    char *name; // name (optional) / or automatically given??? // is there a type detected? 
    void *data; // pointer to the struct 
    th_Category *implementation; // ptr to internal wiring diagram, primitive/atomic if NULL
    th_Category *parent_category; // for hierarchical nesting
    //size_t global_id; // unique id for quick lookup in hypergraph
} th_Object;

// allows the ACT compiler to (i) freely rewrite/fuse pure subgraphs, (ii) never reorder a cross effectful nodes, & (iii) schedule effectful nodes explicitly

/*
// GREAT METADATA
typedef enum
{
  TH_EFF_PURE,
  TH_EFF_STATEFUL,
  TH_EFF_RANDOM,
  TH_EFF_IO_NETWORK,
  TH_EFF_IO_FILE,
  TH_EFF_NDET,
  TH_EFF_BARRIER // forces ordering ie prevents rewrites across
} th_Effect;

typedef enum
{
  TH_CAP_CPU,
  TH_CAP_CUDA,
  TH_CAP_ORT,
  TH_CAP_TRT,
  TH_CAP_NETWORK,
  TH_CAP_FILESYSTEM,
  TH_CAP_API_KEY_X,
  TH_CAP_GPU_MEM, 
  TH_CAP_RAM,
  TH_CAP_MODEL
} th_Capabilities;
*/
typedef uint32_t th_EffectFlags;
typedef uint32_t th_CapabilityFlags;

typedef enum
{
  
} th_Signature;


// used to define transformations / operations & also use for execution to apply in the
// compiler traversal (generalzed idea of back-propogation in some sense)
//typedef struct th_Morphism th_Morphism; // forward declare
typedef struct th_Morphism
{
    char *name; // optional name
    th_Signature signature;
    th_Object **sources; // input objects
    th_Object **targets; // output objects
    size_t n_sources;
    size_t n_targets;
    th_Error (*apply)(th_Context *ctx, struct th_Morphism *m, void *args); // execution morphism
    //bool is_inplace; 
} th_Morphism;


// NOTE: SHOULD I JUST ELIMINATE THE CATEGORY OR MAKE IT (OBJECTC) A REDEF OF OBJECT (ITSELF) ??
// (Monoidal) categeory ie composable wiring diagram
// ex: models are categories, layers arre morphisms
// the monoidal structure is implicit via object-morphism arrays bound via tensor_product
//typedef struct th_Category th_Category; // forward declaration
typedef struct th_Category
{
    // includes zero_input + final_output
    //th_Object **objects; // array/list of pointers to objects inside the category
    //th_Morphism **morphisms;  // array/list of pointers to morphsims inside this category
    //size_t n_objects;
    //size_t m_morphisms; // m should be >= (n-1)
    //struct th_Category *tensor_product; // monoidal binding (ex: parallel layers)
    
    // for object ids:
    //uint_32 inputs[];
    //uint_32 outputs[];
} th_Category; // hypegraph category



// morphism w/ bdry
typedef struct 
{
  // do these need a size or should I just make them dynamic??
  //uint32_t input_ports[];
  uint32_t output_ports[];
    //void **nodes; // array / list of th_Objects / Categories 
    //size_t num_nodes;
    //size_t **sources; // source node indices (per hyperedge)
    //size_t **targets; // target node indices (per hyperedge) 
    //size_t **arities_in; // input arity (")
    //size_t **arities_out; // output "
    //size_t num_hyperedges;
    //uint8_t *edge_types; // bitarray? for opt. (ex: op types for fusion as colimits)

} th_HyperGraph;

// an endofunctor on hypergraph category optimizing natural transformations
typedef struct 
{
    th_HyperGraph *graph;
    th_Context *ctx;
    //bool enable_fusion; // opt-in flags ???
    //bool enable_codegen;
    th_Error (**passes)(th_HyperGraph *); // extensible arr of passes (ex: topo-sort, fusion)
    size_t num_passes;
} th_Compiler;

// topo-sorted array for execution 
// category: linearized as total order via functor (topo-sort as limit)
typedef struct
{
    void *data;
    th_Error (*execution_fn)(th_Context *ctx, void *inputs[], size_t n_in, void *outputs[], size_t n_out);
    size_t *dep_indices;
    size_t num_deps;
    size_t num_inputs, num_outputs;
} th_ExecutionNode;


// ACT fns
// basic init-destroy
th_Error th_init_objects();
th_Error th_destroy_objects();

th_Error th_init_morphism();
th_Error th_destroy_morphism();

th_Error th_init_category();
th_Error th_destroy_category();
// add/compose
th_Error th_add_object_to_category();
th_Error th_add_morphism_to_category();
th_Error th_compose_morphisms();
th_Error th_tensor_product_categories();

// HYPERGRAPH fns


// generates the hypergraph diagram from the cateogry given
th_Error th_generate_hypergraph(th_Category *root);



// COMPILER fns








// check this paper out (names/applies CAT to ML):  
// https://arxiv.org/html/2409.12100v1




/*
// representation structure (from the Category)

typedef struct PortGraph; // feed-forward only
typedef struct Prop; // feed backward
typedef struct Hypergraph; // standard hypergraph
typedef struct HypergraphProp; // complex hypergraphs

// will analyze the Category and then chose the appropriate option
typedef union
{
    Port port_graph;
    Prop prop_graph;
    Hypergraph hyper_graph;
    HypergraphProp hyper_prop_graph;
} Graph;


void init_hypergraph(Hypergraph *h);
void destroy_hypergraph(Hypergraph *h);
void add_object(Hypergraph *h, Object *o);
void add_morphism_between(Hypergraph *h, Object *source, Object *target);
void execute_hypergraph(Hypergraph *h); // essentially run the pipeline

// Preorder := (set P, order relation <=)
typedef struct Preorder;
// Monoid := (set M, binary operation *, identity element e)
typedef struct Monoid;
// MonoidalPreorder := (set P, order <=, binding operation *, identity element e)
typedef struct MonoidalPreorder;
// Category := (ObjectsC, MorphismsC, composition ;, id)
//typedef struct Category;
// MonoidalCategory := (Category C, monoid structure (M, *, e), identity morphism I)
typedef struct MonoidalCategory;


typedef struct SymmetricMonoidalCategory;
typedef struct SymmetricMonoidalPreorder;
typedef struct PropSignature;
typedef struct PropFunctor;
typedef struct LaxSymmetricMonoidalFunctor;

// Topos:= category w/ 
// (i) limits + colimits
// (ii) epi-mono factorization
// (iii) exponential objects
// (iv) subobject classifier

typedef struct Topos;
*/

#endif // __THYMOS_ACT_H__
