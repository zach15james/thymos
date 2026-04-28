//////////////////////////////////////////////////////////////////////////////
// Thymos Optimization (c) 2026 by Zachary R. James (@thymos)
// Core optimization library public API
// much of my opt learning was under Dr. Hongchao Zhang
// who taught my applied optimizaton theory course. 
// should define problem, optimizer, solver, objective
// should: conform to mathematical literature conventions & be algorithmic-centric
// includes armijo + wolf line search + adam optimizer for NNs
//////////////////////////////////////////////////////////////////////////////

#if !defined(__THYMOS_OPT_H__)
#define __THYMOS_OPT_H__

#include <stddef.h>
#include <stdbool.h>
#include "ml.h"  // TH_Context, TH_Error, th_value_t
#include "la.h" // all the supporting work

// Problem descriptor: f and optionally g, optional Hessian-vector product
typedef th_Error (*th_eval_fn)(
  th_Context *ctx,
  void *user,
  const th_value_t *x, size_t n,
  th_value_t *out_f,          // may be NULL if only gradient is needed
  th_value_t *out_g           // length n, may be NULL if only f is needed
);

typedef th_Error (*th_hv_fn)(
  th_Context *ctx,
  void *user,
  const th_value_t *x, size_t n,
  const th_value_t *v,        // length n
  th_value_t *out_Hv          // length n
);

// Optimization problem (structure + callbacks)
typedef struct
{
  th_eval_fn   eval;
  th_hv_fn     hv;            // optional (NULL if not provided)
  void        *user;
  size_t       n;

  // Optional simple box constraints
  const th_value_t *lb;       // NULL if no lower bounds
  const th_value_t *ub;       // NULL if no upper bounds
} th_OptProblem;

// Methods and line search types
typedef enum
{
  TH_OPT_SD = 0,              // Steepest Descent
  TH_OPT_CG,                  // Nonlinear Conjugate Gradient
  TH_OPT_BFGS,                // Quasi-Newton BFGS
  TH_OPT_LBFGS,               // Limited-memory BFGS
  TH_OPT_TR,                  // Trust-region (dogleg/CG-Steihaug)
  TH_OPT_ADAM                 // Adam/SGD (stochastic)
} th_OptMethod;

typedef enum
{
  TH_LS_ARMIJO = 0,
  TH_LS_WOLFE_STRONG
} th_LineSearch;

typedef struct
{
  th_value_t delta;           // Armijo parameter (e.g., 1e-4)
  th_value_t sigma;           // Wolfe parameter in (0,1) (e.g., 0.9)
  th_value_t alpha0;          // initial step size
  th_value_t alpha_tol;       // min alpha tolerance
  int        max_iters;       // max iterations for line search
} th_LineSearchOpts;

// Method-specific knobs (extend as needed)
typedef struct { int variant; } th_CGOpts;      // 0=DY,1=FR,2=PRP (etc.)
typedef struct { bool reset_if_non_psd; } th_BFGSOpts;
typedef struct { int m_hist; } th_LBFGSOpts;    // history size
typedef struct { th_value_t trust0, eta; } th_TROpts;
typedef struct { th_value_t lr, beta1, beta2, eps; size_t batch_size; } th_AdamOpts;

// Optimizer configuration
typedef struct
{
  th_OptMethod      method;
  th_LineSearch     ls_kind;     // for line-search-based methods
  th_LineSearchOpts ls;          // parameters for line search

  // Termination
  th_value_t        tol;         // ||g|| tolerance
  int               max_iters;   // main loop cap

  // Regularization / structure
  th_value_t        l2;          // L2 (ridge)
  th_value_t        l1;          // L1 (lasso)
  bool              project_bounds; // project onto [lb, ub] if provided

  // Method-specific options
  union {
    th_CGOpts    cg;
    th_BFGSOpts  bfgs;
    th_LBFGSOpts lbfgs;
    th_TROpts    tr;
    th_AdamOpts  adam;
  } u;
} th_Optimizer;

// Status/results
typedef enum
{
  TH_OPT_OK = 0,
  TH_OPT_MAX_ITERS,
  TH_OPT_LS_FAILED,
  TH_OPT_INVALID_INPUT
} th_OptStatus;

typedef struct
{
  th_OptStatus  status;
  int           iterations;
  th_value_t    f_final;
  th_value_t    grad_norm_final;
  th_value_t    alpha_last;
} th_OptimizeResult;

// Single entry point
th_Error th_opt_solve(
  th_Context *ctx,
  const th_OptProblem *prob,
  const th_Optimizer  *opt,
  th_value_t *x_io,                // in/out: starting guess, modified in-place
  th_OptimizeResult *out_result
);

// Optional convenience wrappers (map to th_opt_solve with presets)
th_Error th_optimize_sd_armijo(
  th_Context *ctx, const th_OptProblem *prob, th_value_t *x_io,
  th_value_t tol, int max_iters,
  th_value_t armijo_delta, th_value_t alpha0, int ls_max_iters, th_value_t alpha_tol,
  th_OptimizeResult *out_result
);

th_Error th_optimize_cg_dy_wolfe(
  th_Context *ctx, const th_OptProblem *prob, th_value_t *x_io,
  th_value_t tol, int max_iters,
  th_value_t armijo_delta, th_value_t wolfe_sigma,
  th_value_t alpha0, int ls_max_iters, th_value_t alpha_tol,
  th_OptimizeResult *out_result
);

th_Error th_optimize_bfgs_wolfe(
  th_Context *ctx, const th_OptProblem *prob, th_value_t *x_io,
  th_value_t tol, int max_iters,
  th_value_t armijo_delta, th_value_t wolfe_sigma,
  th_value_t alpha0, int ls_max_iters, th_value_t alpha_tol,
  bool reset_if_non_psd,
  th_OptimizeResult *out_result
);

#endif // __THYMOS_OPT_H__



/// NOTES

// examples: 
// Linear Programming: min { c^Tx } st Ax=b, x>=0
// Quadratic Programming: min 1/2 * x^TQx + c^Td st Ax=b, x>=0
// Smooth/non-smooth
// convex/non-convex
// cts & discrete
// deterministic & stochastic

// tip: to solve hard problem, relax to easier problem and find conditions where your easier solution holds true for hard problem

// review: 
// vector norm: ||.||: R^n -> R^t st. 
// 1) ||x|| >= 0 && ||x||=0 <=> x=1 'non-negativity' 
// 2) ||ax|| == |a|||x|| 'positive homogeneity'
// 3) ||x+y|| <= ||x|| + ||y|| 'triangle inequality' 


// p-norm: ||x||_p := [ sum_i=1->n |x_i|^p ] ^ 1/p
// inf-norm: ||x||_inf := max{ |x_i| } for i<=i<=n
// Foubinious Matrix Norm: ||A||_F := (sum_i=1->m sum_j=1->n a_ij ^2) ^ 1/2 = (tr(AA^T)) ^ 1/2 = (tr(A^TA)) ^ 1/2
// --given U in R^mxm, U^TV=I, V in R^nxn, V^TV=I, ||UAV||_F = ||A||_F
// induced matrix norm ||A||_(m,n) = max{||Ax||} st ||x|| = 1, x in R^n
// trace: tr():=sum of diagonal elements
// rank: rank(A) := dimsentions of vector space generated by its columns, a measure of 'non-degeneratness' := dim(img(g(f)))
// determinant: det(A) := scalar-valued fns  := |A| st
// 1) det(I) = 1 'identity' 
// 2) exchange of 2 rows multiplies det by -1, 3) multiplying a row by a # multiplies the det by that # 'multipliers'
// 4) adding a multiple of one row to another row does not change the determinant 
// note: |A| != 0 <=> 1) A invertible & 2) corresponding linear map an isomorphism
// eigenvalue / 'characteristic vector': vector that has its direction unchanged or reversed by a given linear transformation
// eigenvector v, of linear transformation T, scaled by constant factor lambda, where linear transform applies to: Tv=lambdav

// recall: (AB)^T = B^TA^T, AB != BA generally



// basic optimization layout: 
// function f
// gradient of f (so you don't have to contunually re-calculate it)
// intial guess (x_0)
// max iterations
// max line search iterations
// armijo delta OR wolfe sigma 
// implementations: 
// (1) steepest descent (Armijo)
// (2) Dai-Yuan (DY) CG (Wolfe)
// (3) BFGS Quasi-Newton (Wolfe)
// ** find out what the 'BONUS condition' from your matlab assignment was 
// * need hessian calculation... ie be able to take the derivative
// for numerical differentiation: 
// - central (finite) difference: f'(x):= f(x+h) - f(x-h) / 2h
// - richardson extrapolation
// - taylor series 


// problem layout: 
// minimize ObjectiveFn
// subject to:
// g_i(x) <= 0, i = 1,...,m
// h_j(x) = 0, j = 1,...,p

// look further into: 
// primal <--> dual problems ?
// multi-objectie optimization?
// dynamic problems (Bellman equations, policy gradients)


// good refence (Convex opt, boyd):
// https://stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

typedef struct
{

//  (*th_ObjectiveFn) objective_fn;

} th_OptimizationProblem;

typedef struct { } th_Constraints;


// fn ptr syntax: return_type (*pointer_name)(parameter_types);

enum th_linesearch { TH_ARMIJO, TH_WOLFE };

void line_search(th_LineSearch search_type,
                 void *fn,
                 void *graf);

typedef struct { } th_Cone;
typedef struct { } th_Primal;
typedef struct { } th_Dual;





// optimization: minimize 'objective function' (real-valued, affine--automorphisms that maps onto itself--, & ratios of lengths)
//            f: R^n -> R, min f(x) st s in 'feasible set' (set of intersections of minitely-many half spaces--each defined by a linear inequality =: 'convex polytope') X subset/== R^n
// unconstrained optimization would be X = R^n
// linear programming (LP): min c^Tx st Ax=b, x>=0, ie find a point in polytope where the fn has the smallest(/largest) value
// so you have file vector X, that minimizes c^Tx, subject to Ax<=b
// quadratic programming (QP): minimize 1/2x^TQx + c^Td st Ax=b, or Ax <= b (component-wise ineq where every entry in the vector Ax = corresponding entry of vector b)
// - its a type of linear programming
// constrainted least squares: special case of QP when W is symmetric, positive-definite := minimize 1/2 ||Rx-d||^2 where Q=R^TR from Cholesky decomposition of Q & c=-R^T
// - note: any constrained least squares probelm can be framed as a QP problem
// QP generalization: when minimizing fn f in N(x_0), 'neighborhood', Q := Hessian matrix H(f(x_0)) & c := gradient f(x_0)
// - solved by: interior pt, active set, angled Lagrangian, conjugate gradient, gradient projection, & extensions of simplex
// quadratically constrained quadratic program (QCQP): minimize 1/2x^TP_0x + q_0^Tx subject to 1/2x^TP_ix + q_i^Tx + r_i <= 0 & Ax=b where {P_0, ..., P_m} nxn matrices, x optimizations variable
