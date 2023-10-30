import numpy as np

# We will use this function to compute the loss in many functions, so we define it here:
def compute_mse(y, tx, w):
    e = y - tx.dot(w)
    loss = np.dot(e.T,e)/(2*len(y))
    return loss

# ************************************************************************************************************************

"""implement Gradient Descent"""
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.
    Args:
        y: numpy array of shape=(N,)
        tx: numpy array of shape=(N,D)
        initial_w: numpy array of shape=(D,). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the step-size of GD

    Returns:
        w: the last weight obtained from GD
        loss: the last loss obtained from GD
    """
    w = initial_w
    if max_iters==0:
        e = y - tx.dot(w)
        loss0 = round(np.dot(e.T,e)/(2*tx.shape[0]), 6)
        return w, loss0
    else:
        for n_iter in range(max_iters):
            # Computation of the gradient
            e = y - tx.dot(w)
            gradient = -np.dot(tx.T, e)/(tx.shape[0])
            # Updated weights w with GD
            w = w - gamma*gradient
            #Computation of the loss, i.e. the MSE
            loss = compute_mse(y, tx, w)
        # We return the last weights and loss of GD
        return w, loss

# ************************************************************************************************************************
            
"""implement Stochastic Gradient Descent"""

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    err = y - tx.dot(w)
    grad = -tx.T.dot(err) / len(err)
    return grad, err


"""implement Stochastic Gradient Descent"""
def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, batch_size, max_iters, gamma):
    """Compute a stochastic gradient at w from a data sample batch of size B, where B < N, and their corresponding labels.
    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        initial_w: numpy array of shape=(2, ). The vector of initial model parameters
        batch_size: integer B, in general B=1 (here this is what we will consider)
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize of GD

    Returns:
        w: final weights of shape (D,)
        loss: scalar, mse
    """
    # For SGD, we want batch_size to always be equal to 1:
    assert batch_size == 1
        
    w = initial_w
    loss = 0
    
    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size = batch_size, num_batches = 1):
            grad, err = compute_stoch_gradient(y_batch, tx_batch, w)
            w = w - gamma * grad
            loss = compute_mse(y, tx, w)
    return w, loss

# ************************************************************************************************************************

"""implement Least Squares"""
def least_squares(y, tx):
    """Calculate the least squares solution. Return weights and loss.     
    Args:
        y: numpy array of shape (N,), N is the number of samples
        tx: numpy array of shape (N,D), D is the number of features

    Returns:
        w: optimal weights, numpy array of shape(D,).
        loss: scalar.
    """
    # Here, we use np.linalg.pinv instead of solve because we had a problem of singular matrix
    # when dividing our data set in 
    w = (np.linalg.pinv(np.dot(tx.T, tx))).dot(np.dot(tx.T,y))
    N = tx.shape[0]
    # Computation of the mean squared error:
    loss = compute_mse(y, tx, w)
    return w, loss

# ************************************************************************************************************************

"""implement Ridge Regression"""
def ridge_regression(y, tx, lambda_):
    """
    Args:
        y: numpy array of shape (N,), N is the number of samples
        tx: numpy array of shape (N,D), D is the number of features
        lambda_: scalar

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar
    """
    lambda_I = (2*lambda_*len(y)) * np.identity(tx.shape[1])
    A = tx.T.dot(tx)+ lambda_I
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)

    return w, loss

# ************************************************************************************************************************
def compute_logistic_loss(y, tx, w):
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    N = tx.shape[0]
    sigma = 1/(1+np.exp(-tx.dot(w)))
    loss = float(-(y.T.dot(np.log(sigma)) + (1-y).T.dot(np.log(1-sigma)))/N)
    return loss


"""implement Logistic Regression"""
def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """return the final weight and loss of GD
    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        w_initial: initial weight with shape=(D,), to be updated via Gradient Descent (GD)
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the step-size of GD

    Returns:
        w: final weight after GD of shape (D,)
        loss: scalar number
    """
    w = initial_w
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    N = tx.shape[0]

    if max_iters==0:
        sigma = 1/(1+np.exp(-tx.dot(w)))
        loss0 = compute_logistic_loss(y, tx, w)
        return w, loss0
    else:
        for n in range(max_iters):
            # Update of w with GD
            sigma = 1/(1+np.exp(-tx.dot(w)))
            gradient = tx.T.dot(sigma-y)/N
            w = w - gamma*gradient
            # Computation of the loss 
            loss = compute_logistic_loss(y, tx, w)
        # We return the last weights and loss obtained 
        return w, loss

# ************************************************************************************************************************

"""implement Regularized Logistic Regression"""
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """return the final weights and loss after the GD algorithm.
    Args:
        y:  shape=(N,)
        tx: shape=(N, D)
        lambda_: regularization parameter
        w_initial: initial weight with shape=(D,), to be updated via GD
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the step-size of GD

    Returns:
        w: final weight after GD of shape (D,)
        loss: scalar number
    """
    w = initial_w
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    
    N = tx.shape[0]
    
    if max_iters==0:
        sigma = 1/(1+np.exp(-tx.dot(w)))
        loss0 = compute_logistic_loss(y, tx, w)
        return w, loss0
    else:
        for n in range(max_iters):
            sigma = 1/(1+np.exp(-tx.dot(w)))
            gradient = tx.T.dot(sigma-y)/N + 2*lambda_*w
            # Update of w with GD
            w = w - gamma*gradient
            # Computation of the loss, without the regularization term
            loss = compute_logistic_loss(y, tx, w)
        # We return the last weights and loss obtained 
        return w, loss
