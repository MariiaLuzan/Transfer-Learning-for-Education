import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from scipy.spatial.distance import pdist

def calc_kernels_for_l(X1, X2, length_scale_list, quantiles):
    
    """
    For various settings of the kernel hyperparameter "length_scale",
    this function computes percentiles of the values within a kernel matrix between two arrays, X1 and X2
    
    Args:
    X1 (Numpy Array) - Data containing source features
    X2 (Numpy Array) - Data containing target features
    length_scale_list (list) - A list of potential values for the length_scale parameter
    quantiles (list) - A list of quantile levels
    
    Returns:
    length_scale_df (Pandas DataFrame) - A DataFrame containing the calculated percentiles of 
                                         elements within the kernel matrix.
    """
    
    length_scale_df = pd.DataFrame(index=length_scale_list, columns=quantiles)
    
    if np.array_equal(X1, X2):
        # Array of distances between all points in X1
        X_dist = pdist(X1)
    else:
        X_dist = pdist(np.concatenate((X1, X2)))
        #Take distances only between 2 group of points - X1 and X2
        full_matrix = np.zeros((len(X1)+len(X2),len(X1)+len(X2)))
        inds = np.triu_indices_from(full_matrix, k = 1)
        full_matrix[inds] = X_dist
        X_dist = full_matrix[:len(X1),len(X1):]
       
    for quantile in quantiles:
        length_scale_df.loc['dist', quantile] = np.percentile(X_dist, 100-quantile*100)
    
    
    for length_scale in length_scale_list:
        gaussian_kernel = RBF(length_scale=length_scale)
        K = gaussian_kernel.__call__(X1, X2)
        
        if np.array_equal(X1, X2):
            # Take upper triangle of a kernel matrix
            K = K[np.triu_indices(len(X1),1)]
            
        for quantile in quantiles:
            length_scale_df.loc[length_scale, quantile] = np.percentile(K, quantile*100) 
            
    return length_scale_df