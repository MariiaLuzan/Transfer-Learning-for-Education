import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from cvxopt import matrix
from cvxopt import solvers

from transfer_learning.cvxopt_input import get_cvxopt_input
from transformers.features_missing_vals import imputation_pipeline
from sklearn.preprocessing import StandardScaler


def transfer_learning(source_school, target_school,
                      df, imputation_pipeline,
                      length_scale_list,
                      clf, cols):
    """
    Conducts transfer learning using an instance weighting strategy
    
    Args:
    - source_school (string): Code of the source school (obtained from the "PRMRY_CRER_CD" column);
    - target_school (string): Code of the target school (obtained from the "PRMRY_CRER_CD" column);
    - df (pandas dataframe): A dataframe containing features and the dependent variable for all schools;
    - imputation_pipeline (pipeline): A pipeline used for imputing missing values;
    - length_scale_list (list of floats) - list of length scale values of the kernel
    - clf: Logistic regression classifier;
    - cols (list of strings): List of features for the transferred model.

    Returns:
    - auc (float): AUC of the transferred source model calculated on the target data;
    - auc_w (float): AUC of the transferred weighted source model calculated on the target data;
    - best_weights (Numpy Array): the weights of the best transfer learning model.
    """
    
    # Source and target data
    source_df = df[df['PRMRY_CRER_CD']==source_school].copy()
    target_df = df[df['PRMRY_CRER_CD']==target_school].copy()
    
    # Impute missing values and WoE
    source_imp = imputation_pipeline.fit_transform(source_df, source_df['y_ENROLLED_1_YEAR_LATER'])
    target_imp = imputation_pipeline.transform(target_df)
    
    # Standartization
    scaler = StandardScaler()
    scaler.fit(source_imp[cols])
    X_source = scaler.transform(source_imp[cols])
    X_target = scaler.transform(target_imp[cols])
    
        
    # Source model
    y = source_imp['y_ENROLLED_1_YEAR_LATER']
    clf.fit(X_source, y)
    
    # The model trained on the source data is applied to the target dataset
    y_pred = clf.predict_proba(X_target)
    auc = roc_auc_score(target_imp['y_ENROLLED_1_YEAR_LATER'], y_pred[:,1])
    
    aucs_w = []
    auc_max = 0
    best_weights = None
    
    for length_scale in length_scale_list:
        # Weights
        # Minimizing the difference between the target and source distributions using the CVXOPT optimizer
        P, q, G, h = get_cvxopt_input(X_source, X_target, length_scale)
        sol = solvers.qp(P,q,G,h)
        weights = np.array(sol['x']).flatten()
    
        # Weighted model
        clf_weighted = clf
        clf_weighted.fit(X_source, y, weights)
    
        # The model trained on the weighted source data is applied to the target dataset
        y_pred = clf_weighted.predict_proba(X_target)
        auc_w = roc_auc_score(target_imp['y_ENROLLED_1_YEAR_LATER'], y_pred[:,1])
        aucs_w.append(auc_w)
        
        if auc_max < auc_w:
            auc_max = auc_w
            best_weights = weights 
            
        print("AUCs, non-weighted and weighted, ", auc, auc_w)
    
    return auc, aucs_w, best_weights