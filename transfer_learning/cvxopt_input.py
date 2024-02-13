import pandas as pd
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from cvxopt import matrix

def get_cvxopt_input(source_df, target_df, length_scale):
    """
    Compute input data for CVXOPT optimization
    
    Args:
    source_df (pandas DataFrame) - DataFrame containing features for the source school
    target_df (pandas DataFrame) - DataFrame containing features for the target school
    length_scale (float) - Length scale of the kernel
    
    Returns:
    P,q,G,h - cvxopt matrices
    """
    m = len(source_df)
    n = len(target_df)
    epsilon = (m**(0.5) - 1) / m**0.5
    
    # Calculate kernels
    kernel = RBF(length_scale=length_scale)
    K = kernel.__call__(source_df, source_df)
    k = kernel.__call__(source_df, target_df)
    k = np.sum(k, axis=1) * (m/n)
    
    # Input for CVXOPT
    P = matrix(K, tc='d')
    q = matrix(-k, tc='d')
    
    
    #Constraints in the form Gx <= h:
    
    # First part of G and h to reflect the condition: Xi <= 1000
    G1 = np.diag(np.ones(m))
    h1 = np.ones(m)*1000
    
    # Second part of G to reflect the condition: Xi >= 0  =>  -Xi <= 0
    G2 = np.diag(-np.ones(m))
    h2 = np.zeros(m)
        
    # Third part of G to reflect the condition: sum(Xi) <= m+e*m
    G3 = np.ones(m).reshape(1,-1)
    
    # Forth part of G to reflect the condition: sum(Xi) >= m-e*m   =>
    # -sum(Xi) <= -m+e*m
    G4 = -np.ones(m).reshape(1,-1)
    
    # Concatenate all 4 types of conditions
    G = matrix(np.vstack((G1, G2, G3, G4)), tc='d')
    h = np.concatenate((h1,h2))
    h = matrix(np.append(h, [m + epsilon * m, -m + epsilon * m]), tc='d')
    
    return P, q, G, h
    
    