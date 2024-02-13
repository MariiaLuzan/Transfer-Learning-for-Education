import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def univariate_auc(df, factors_df, clf):
    """
    Estimates AUC for individual factors
    
    Args:
    df (pandas DataFrame) - Dataframe with features
    factors_df (pandas DataFrame) - Dataframe containing groups of factors and corresponding individual factors
    clf (sklearn estimator) - Model to estimate 
    
    Returns:
    factors_df (pandas DataFrame) - Input dataframe with an added column 'AUC'
    """
    
    factors_AUC = factors_df.copy()
        
    i=0
    factors_AUC['AUC'] = np.NaN
    
    for factor in factors_AUC['Factor']:
        X = df[factor].to_numpy()
        if type(factor)==str:
            X = X.reshape(-1,1)
        
        clf.fit(X, df['y_ENROLLED_1_YEAR_LATER'])
        y_pred = clf.predict_proba(X)
        auc = roc_auc_score(df['y_ENROLLED_1_YEAR_LATER'], y_pred[:,1])
        factors_AUC.iloc[i, 2] = auc
    
        i += 1
    
    return factors_AUC