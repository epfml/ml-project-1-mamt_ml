import numpy as np
from standardization import*


def split_data(X, label):
    indices = []
    for i in range(X.shape[0]):
        # Supposing we choose to work with column AZ = SEX -> 51th column = 50th column here as we're counting 0 
        # 1 represents male and 2 represents female in our data set
        if X[i,50] in label:
            # If a line contains element 1 or 2 that is in label, there append it to the list of indices
            indices.append(i)
    return indices

# ************************************************************************************************************************

def intercept_(X):
    # We add a column full of ones to add the intercept term
    return np.c_[np.ones((X.shape[0], 1)), X]

# ************************************************************************************************************************

def replace_values(X):
    # Calculate the median of each column, ignoring NaN values
    column_medians = np.nanmedian(X, axis=0)

    # Find the indices of NaN values in the matrix
    nan_indices = np.isnan(X)
    
    # Replace NaN values with the respective column medians
    for col_idx in range(X.shape[1]):
        col_nan_indices = nan_indices[:, col_idx]
        X[col_nan_indices, col_idx] = column_medians[col_idx]

    return X

# ************************************************************************************************************************

def clean_data(X, label, correlation = False):
    X_new = X.copy()    
    X_new = replace_values(X_new)

    # Extract the indices for which the elements belong to the group "label"
    ind = split_data(X_new, label)
    X_new = X_new[ind]

    # Standardize the final matrix :
    X_new, mean, std_ = standardize(X_new)
    
    # Testing whether our matrix contains any nan values :
    has_nan = np.isnan(X_new).any()
    
    if has_nan:
        print("contains NaN values.")
    else:
        print("Youpi!")

    # Adding an intercept to each matrix
    X_new = intercept_(X_new)
    print(X_new.shape)
    
    return X_new, ind, mean, std_

# ************************************************************************************************************************
# Here, we clean the test set using the mean and std of the train set, this is why we need another function:
def clean_data_test_set(X, label, mean, std, correlation = False):
    X_new = X.copy()
    X_new = replace_values(X_new)

    # Extract the indices for which the elements belong to the group "label"
    ind = split_data(X_new, label)
    X_new = X_new[ind]
    
    # Testing whether our matrix contains any nan values or not:
    has_nan = np.isnan(X_new).any()
    if has_nan:
        print("contains NaN values.")
    else:
        print("Youpi!")
    
    # Standardize the final matrix :
    X_new = standardize_test(X_new, mean, std)

    # Adding an intercept to each matrix
    X_new = intercept_(X_new)
    print(X_new.shape)
    
    return X_new, ind

# ************************************************************************************************************************

def predict_labels(weights, data):
    y_pred = np.dot(data, weights)
    # We didn't pick 1/2 because our data set is unbalanced
    y_pred[np.where(y_pred <= 1/4)] = 0
    y_pred[np.where(y_pred > 1/4)] = 1
    return y_pred
