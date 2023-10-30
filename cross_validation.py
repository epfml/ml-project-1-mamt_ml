import numpy as np
from implementations import *

# IMPLEMENT Cross-validation for ridge regression :

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

# ************************************************************************************************************************

def cross_validation(y, tx, k_indices, k, lambda_):
    test_ind = k_indices[k]
    
    train_ind = k_indices[~(np.arange(k_indices.shape[0])==k)]
    train_ind = np.array(train_ind.reshape(-1))
    
    y_te = y[test_ind]
    y_tr = y[train_ind]
    x_te = tx[test_ind, :]
    x_tr = tx[train_ind, :]
    
    w, loss_tr = ridge_regression(y_tr, x_tr, lambda_)
    w_te,loss_te = ridge_regression(y_te, x_te, lambda_)
    return loss_tr, loss_te, w

# ************************************************************************************************************************
    
def best_lambda(y, tx, k_fold, lambdas, seed):
    """cross validation over regularisation parameter lambda.

    Args:
        y:       shape (N,)
        tx:      shape (N, D)
        degree:  integer, degree of the polynomial expansion
        k_fold:  integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_lambda : scalar, value of the best lambda 
    """
    k_indices = build_k_indices(y, k_fold, seed)
    
    rmse_tr = []
    rmse_te = []
    # ***************************************************
    # cross validation over lambdas
    # ***************************************************
    for lambda_ in lambdas:
        tr = []
        te = []
        for k in range(k_fold):
            loss_tr, loss_te, w = cross_validation(y, tx, k_indices, k, lambda_)
            tr.append(loss_tr)
            te.append(loss_te)
        rmse_tr.append(np.mean(tr))
        rmse_te.append(np.mean(te))
    
    best_lambda_index = np.argmin(rmse_te)
    best_lambda = lambdas[best_lambda_index]
    
    print('The best parameter lambda is:', best_lambda)
    
    return best_lambda
 
# ************************************************************************************************************************    

def ridge_regression_cross_validation(y, tx, k_fold, lambdas): 
    """
    We know create a function to find the best lambda for ridge regression and use it to compute our optimal weights 
    and loss

     Args:
        y:       shape=(N,)
        x:       shape=(N,D)
        k_fold:  integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test

    Returns:
        w: final weights of shape (D,) using the optimal lambda for thr ridge regression
        loss: final loss, a scalar, using the optimal lambda for thr ridge regression
    """
    seed = 12
    k_indices = build_k_indices (y, k_fold, seed)
    lambda_ = best_lambda(y, tx, k_fold, lambdas, seed)
    w, loss = ridge_regression(y, tx, lambda_)
    
    return w, loss
