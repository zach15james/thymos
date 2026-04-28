

typedef enum 
{
    SIMPLE,
    MULTIPLE,
    POLYNOMIAL,
    ORDINARY_LEAST_SQUARES,
    WEIGHTED_LEAST_SQUARES,
    GENERALIZED_LEAST_SQUARES,

} LinearRegressionType;


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












