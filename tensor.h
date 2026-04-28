//////////////////////////////////////////////////////////////////////////////
// Thymos Linear Algebra (LA) (c) 2023-2026 by Zachary R. James (@thymos)
// Tensor := core building block ==> Vector + Matrix
// have dot products, norms, LUP, GE, S-Morrison, eigen_sip
//////////////////////////////////////////////////////////////////////////////

#include "ml.h"


#if ! defined(__THYMOS_LA_H__)
#define __THYMOS_LA_H__


// TENSORS //

// The the core mathematical linear algebra structure from which all structs are defined
typedef struct
{
  uint8_t *data;
  size_t *dimension_sizes;
  size_t number_dimensions;
} th_Tensor;

// renaming these for context but keeping implementation the same
typedef th_Tensor th_Vector;
typedef th_Tensor th_Matrix;

// creating, deletion, & copying (& printing) 

static void th_print_tensor_recursive(th_Tensor *p, size_t current_dim, size_t offset);
void th_init_tensor(th_Tensor *p, const size_t *dimension_sizes);
void th_init_vector(th_Vector *p, size_t vector_length);
void th_init_matrix(th_Matrix *p, size_t matrix_rows, size_t matrix_cols);
void th_destroy_tensor(th_Tensor *p);
void th_destroy_vector(th_Vector *p);
void th_destroy_matrix(th_Matrix *p);
void th_print_tensor(th_Tensor *p);
void th_print_tensor_recursive(th_Tensor *p, size_t current_dim, size_t offset);
void th_print_vector(th_Vector *p);
void th_print_matrix(th_Matrix *p);

// NOTE: for optimizations later: SIMD, Vectorization, Sparse, or Sub-matrix blocks

// add up to N tensors / matrices 
// NOTE: need to think about how I will do fast algorithms and modifications where I do things that 
//      minimize memory
th_Error tensor_add(th_Context *ctx, th_Tensor result, th_Tensor A, th_Tensor B);
// n = 1 will result in nothing happening, returning A 
th_Error tensor_add_n(th_Context *ctx, int n, th_Tensor result, th_Tensor A, ...);

th_Error tensor_multiply(th_Context *ctx, th_Tensor result, th_Tensor A, th_Tensor B);

// n = 1 will default to nothing (ie just result = A)
th_Error tensor_multiply_n(th_Context *ctx, int n, th_Tensor result, th_Tensor A, ...);

//typedef enum { TH_PNORM, TH_1NORM, TH_2NORM, TH_INFNORM, TH_CUSTOMNORM } th_Norm;
//th_Error th_norm(th_Context *ctx, th_Norm type, th_Tensor *result, th_Tensor *T);

//void* trace(void* M);
//void* rank(void* M);
//void* det(void* M);
//th_Error th_inversion(th_Tensor *result, th_Tensor *T);
//th_Error th_norm(th_Context *ctx, th_Norm type, th_Tensor *result, th_Tensor *T);


// matrix inversion
// eigenvalues & eigenvectors

// matrix decompositions
// diagonalization? 
// QR factorization
// LUP factorization, LU decomposition
// Sherman-Morrison algorithm 
// cholesky factorization
// svd


// potentially later - sub-matrix algorithms, potenially optimized for the gpu







// LA //




#endif // __THYMOS_LA_H__