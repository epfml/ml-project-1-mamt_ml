import numpy as np


def standardize(x):
    centered_data = x - np.mean(x, axis=0)
    std_dev = np.std(centered_data, axis=0)
    # We here handle the case where the standard deviation is zero
    std_data = centered_data / np.where(std_dev == 0, 1e-6, std_dev)
    
    return std_data, np.mean(x,axis=0), np.std(centered_data,axis=0)


# We standardize for the test according the mean and std of the train set
def standardize_test(x, mean, std):
    centered_data = x - mean
    # We here handle the case where the standard deviation is zero
    std_data = centered_data / np.where(std == 0, 1e-6, std)

    return std_data