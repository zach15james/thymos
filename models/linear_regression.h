//////////////////////////////////////////////////////////////////////////////
// Thymos Linear Regression (c) 2026 by Zachary R. James (@thymos)
// Linear regression model
// (implementaion of sublibrary for ml)
// the purose of this is to showcase thymos primitives in action
// later, I will have scikitlearn, c++ Eigen, &c tests to compare w/ these
//////////////////////////////////////////////////////////////////////////////


#ifdef TH_LR_H
#define TH_LR_H


#include <stdbool.h>
#include <stddef.h>
#include "../prelude.h" // th_Error
#include "../ctx.h"     // th_Context
#include "../tensor.h"  // th_Tensor
#include "../la.h"      // BLAS-shaped primitives

typedef enum
{
    TH_LR_OLS = 0,
    TH_LR_RIDGE = 1,
    TH_LR_LASSO = 2,
    TH_LR_ELASTIC_NET = 3
} th_LRPenalty;

typedef struct
{
    th_LRPenalty penalty;
    double lambda;
    double alpha;
    bool fit_intercept;
    double tol;
    int max_iter;
    const th_Tensor *sample_weights;
} th_LRConfig;

typedef struct
{
    th_Tensor *coefficients;
    double intercept;
    int n_features;
    int n_iterations;
    bool converged;
} th_LRModel;

typedef struct
{ double r_squared, adj_r_squared, mse, rmse, mae, huber; } th_LRMetrics;

th_LRConfig th_lr_default_config(void);
th_Error th_lr_fit(th_Context *ctx, const th_Tensor *X, const th_Tensor *y, const th_LRConfig *cfg, th_LRModel *out);
th_Error th_lr_predict(th_Context *ctx, const th_LRModel *m, const th_Tensor *X, th_Tensor *y_out);
th_Error th_lr_metrics(th_Context *ctx, const th_LRModel *m, const th_Tensor *x, const th_Tensor *y, th_LRMetrics *out);
void th_lr_free (th_Context *ctx, th_LRModel *m);

#endif // TH_LR_H


typedef enum
{ 
    NONE,
    RIDGE, // l2 norm
    LASSO, // l1 norm
    ELASTIC, // elastic net (covex combination of ridge + lasso)
} LinearRegularization;

typedef enum
{
    MSE,
    MAE,
    HUBER
} LossFunctions;

typedef enum
{
    Z_SCORE_NORMALIZATION,
    MIN_MAX
} FeatureScaling;





//SCIKIT LEARN
// Linear Models:

// OLS: min w wrt. ||Xw - y||_2 ^ 2

// note: setting alpoha (degree of sparsity of estimated coefficients)
//          cross validation
//          AIC (Akaike information criterion): -2log(L_maximum_liklihood) + 2d_#_params
//          BIC (Bayes information critereon):  -2log(L_maximum_liklihood) + log(N_samples)d
//      for Gaussian model, max liklihood estimate is:
//                          log(L) = -n/2 * log(2pi) - n/2 * ln(sigma^2) - (sum(y_i - y^_i)^2)/2sigma^2
//                      where sigma^2 := estimate of noise variance, y_i and y^_i are true and predicted, respectively



// RIDGE (regression): min w wrt ||Xw-y|| + alpha||w||2^2
//      solvers: lbfgs, cholesky, sparse_cg

// RIDGE (classifier): variant with binary targets {-1,1} 
//      ** same as least squares svm (LSSVM) with linear kernel
//      check: leave-one-out cross validation (& standard cross validation)

// LASSO: min w wrt  1/(2n samples) * ||Xw - y||22 + alpha||w||1 (least squares, l1-norm, penalty)
//      ** implemnation uses: coordinate descent to fit coef
//              try least angle regression for another implementation

// MULTI_TASK_LASSO: linear model estimates spare coef for multiple regression problems jointly
// y := 2d array of shape (n_samples, n_tasks)
// objective: minw wrt 1/2nsamples * ||XW - Y||^2 (Frobinus norm) + alpha||W||_21
//              where Frobinus norm: ||A||_Fro = root(sum_ij(a_ij ^ 2))
//              & l1l2 reads ||A||_21 = sum_i root(sum_j(a_ij^2))
//      implemnation: coordinatne descent 

// ELASTIC_NET
// objective:
// min_w wrt 1/2nsamples * ||Xw - y||^2 _2 + alpha * p||w||_1 + alpha(1-p)/2 * (||w||_2)^2 

// MULTI-task ELASTIC-NET


// LEAST ANGLE REGRESSION (LARS) := at each step finds the feature most correlated w/ target

// LARS LASSO Modification:  https://hastie.su.domains/Papers/LARS/LeastAngle_2002.pdf

// ORTHOGINAL MATCHING PURSUIT (OMP):
// optimim solution vector w/ fixed # of non-zero elements:
//      argmin_w || y - Xw ||_2 ^ 2 st ||w||_0 <= n nonzero_coefficients
// approx optimim solution vector w/ fixed number of non-zero elements:
//      argmin_w ||w||_0 st ||y - Xw||_2 ^ 2 <= tolerane

// BAYESIAN REGRESSION
// p(y | X, w, alpha ) = N(y | Xw, alpha^-1)

// BAYESIAN RIDGE REGRESSION
//  p(w|lambda) = N(w | 0, lambda^-1 * I_p)
// alpha and lambda are estimated by max log marginal liklihood


// AUTOMATIC RELEVANCE DETERMINATION (ARD) / "Sparse Bayesian Learning & Relevance Vector Machine"
//  p(w|lambda) = N(w|0, A^-1) 
// where A := positive definite diagonal matrix diag(A) = lambda = {labmda_1, ..., lambda_n}












