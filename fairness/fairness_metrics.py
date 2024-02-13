import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric


def sliced_auc(y, y_pred, attribites, df):
    """
    Computes AUCs for subsets defined by values of features listed in 'attributes'.
    For instance, if the 'gender' attribute can take values 1 (female) and 0 (male),
    AUC will be estimated for the subset of females and the subset of males.
    
    Args:
    y (pandas Series) - Series containing true y values 
    y_pred (pandas Series) - Series containing predicted y values 
    attribites (List of strings) - List containing names of features (columns) which
                                   will be used to create subsets for AUC estimation
    df (pandas DataFrame) - DataFrame containing data set of features
    
    Returns:
    sliced_aucs (pandas DataFrame) - DataFrame containing AUCs computed for each subset
    """
    
    # define number of rows in the resulting data frame
    num_rows = 0
    for attribute in attribites:
        num_rows += len(df[attribute].unique())
        
    sliced_aucs = pd.DataFrame(index=range(num_rows), columns=['Attribute', 'Value', 'AUC'])
    
    ind = 0
    for attribute in attribites:
        for attr_val in df[attribute].unique():
            sliced_aucs.iloc[ind, 0] = attribute
            sliced_aucs.iloc[ind, 1] = attr_val
            sliced_aucs.iloc[ind, 2] = roc_auc_score(y[df[attribute]==attr_val], y_pred[df[attribute]==attr_val])
            ind += 1    
            
    return sliced_aucs



def fairn_metrics(y, y_pred, attribites_dic, df, threshold_list):
    """
    Computes the Equal Opportunity Difference for each value in the 'threshold_list'.
    Each threshold in the list defines the proportion of students who will receive
    intervention from the university. For example, a threshold of 10% means that the
    bottom 10% of students with the highest predicted dropout probability will receive
    help (labeled as 1), while the remaining students will be labeled as 0.
    
    Args:
    y (pandas Series) - Series containing true y values 
    y_pred (pandas Series) - Series containing predicted y values 
    attribites_dic (dictionary) - Dictionary containing privileged values of attributes
                                  in the format {column name: privileged value}
    df (pandas DataFrame) - DataFrame containing the data set of features 
    threshold_list (List of floats) - List containing thresholds to define which students 
                                      will be labeled as 1 (will receive help from the 
                                      university) or 0 (will not)
    
    Returns:
    fairn_metrics (pandas DataFrame) - Dataframe containing equal oportunity difference 
                                       for each threshold value from the 'threshold_list'
                                        
    """
    
    # define number of rows in the resulting data frame
    num_rows = 0
    for attribute in attribites_dic:
        num_rows += len(df[attribute].unique())
        
    cols = ['thresh_'+str(threshold)+'%' for threshold in threshold_list]    
    fairn_metrics = pd.DataFrame(index=range(num_rows), columns=['Attribute', 'Value']+cols)
       
    # Prepare input DataFrames
    df_fair1 = pd.DataFrame(data={'y': y})
    for column in attribites_dic:
        df_fair1[column] = df[column]
    df_fair2 = df_fair1.copy()    
    
    # Transform the input data frame into BinaryLabelDataset
    df_fair1_bld = BinaryLabelDataset(df=df_fair1, label_names=['y'],
                                      protected_attribute_names=list(attribites_dic.keys()))
        
    # Loop via attributes
    ind = 0
    for attribute in attribites_dic:
        for attr_val in df[attribute].unique():
            fairn_metrics.iloc[ind, 0] = attribute
            fairn_metrics.iloc[ind, 1] = attr_val
                       
            col_ind = 0
            for threshold in threshold_list:
                
                thresh = np.percentile(y_pred, 100-threshold)
                y_pred_labels = np.where(y_pred >= thresh, 1, 0)
                df_fair2['y'] = y_pred_labels 
                df_fair2_bld = BinaryLabelDataset(df=df_fair2, label_names=['y'],
                                                  protected_attribute_names=list(attribites_dic.keys()))
                            
                if attr_val == attribites_dic[attribute]:
                    fairn_metrics.iloc[ind, col_ind+2] = 0
                else:
                    cm = ClassificationMetric(df_fair1_bld, df_fair2_bld, 
                                              privileged_groups=[{attribute: [attribites_dic[attribute]]}],
                                              unprivileged_groups=[{attribute: [attr_val]}])
                    fairn_metrics.iloc[ind, col_ind+2] = cm.equal_opportunity_difference()
                    
                col_ind += 1  
            ind += 1        
            
    return fairn_metrics



def enthropy_metrics(y, y_pred, attribites_dic, df, threshold_list):
    
    """
    Computes the generalized entropy indices for each value in the 'threshold_list'.
    Each threshold in the list defines the proportion of students who will receive
    intervention from the university. For example, a threshold of 10% means that the
    bottom 10% of students with the highest predicted dropout probability will receive
    help (labeled as 1), while the remaining students will be labeled as 0.
    
    Args:
    y (pandas Series) - Series containing true y values 
    y_pred (pandas Series) - Series containing predicted y values 
    attribites_dic (dictionary) - Dictionary containing privileged values of attributes
                                  in the format {column name: privileged value}
    df (pandas DataFrame) - DataFrame containing the data set of features 
    threshold_list (List of floats) - List containing thresholds to define which students 
                                      will be labeled as 1 (will receive help from the 
                                      university) or 0 (will not)
    
    Returns:
    enthropy_metrics (pandas DataFrame) - Dataframe containing gneralized entopy index for 
                                          each threshold value from the 'threshold_list'
                                        
    """
    
    rows_ind = ['generalized entropy index', 
                'between groups generalized entropy index',
                'theil index', 
                'between groups theil index']
    
    cols = ['thresh_'+str(threshold)+'%' for threshold in threshold_list]    
    enthropy_metrics = pd.DataFrame(index=rows_ind, columns=cols)
       
    # Prepare input DataFrames
    df_fair1 = pd.DataFrame(data={'y': y})
    for column in attribites_dic:
        df_fair1[column] = df[column]
    df_fair2 = df_fair1.copy()    
    
    # Transform the input data frame into BinaryLabelDataset
    df_fair1_bld = BinaryLabelDataset(df=df_fair1, label_names=['y'],
                                      protected_attribute_names=list(attribites_dic.keys()))
    
    # Loop via thresholds
    col_ind = 0
    for threshold in threshold_list:
        
        thresh = np.percentile(y_pred, 100-threshold)
        y_pred_labels = np.where(y_pred >= thresh, 1, 0)
        df_fair2['y'] = y_pred_labels 
        df_fair2_bld = BinaryLabelDataset(df=df_fair2, label_names=['y'],
                                          protected_attribute_names=list(attribites_dic.keys()))
        cm = ClassificationMetric(df_fair1_bld, df_fair2_bld)
        
        enthropy_metrics.loc['generalized entropy index', 'thresh_'+str(threshold)+'%'] = cm.generalized_entropy_index()
        enthropy_metrics.loc['between groups generalized entropy index', 'thresh_'+str(threshold)+'%'] = cm.between_all_groups_generalized_entropy_index()
        enthropy_metrics.loc['theil index', 'thresh_'+str(threshold)+'%'] = cm.theil_index()
        enthropy_metrics.loc['between groups theil index', 'thresh_'+str(threshold)+'%'] = cm.between_all_groups_theil_index()
    
        col_ind += 1  
    
    
    
    return enthropy_metrics