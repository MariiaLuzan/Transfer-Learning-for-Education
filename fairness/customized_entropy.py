import numpy as np
import pandas as pd



def calc_benefits(df_true_predict, threshold):
    """
    Function for computing benefits used in estimating the generalized entropy index, 
    as implemented in the aif360 package (https://aif360.readthedocs.io/en/stable/)
    
    Args:
    df_true_predict (pandas DataFrame) - DataFrame containing true values (in column 'y') and 
                                         predicted values (in column 'y_pred') of y
    threshold (Float) - Threshold indicating which predicted dropout probability values are considered 
                        as label 1 (e.g., threshold=10 means the top 10% of students with the highest 
                        predicted probabilities are labeled as 1)
    
    Returns:
    df_true_predict (pandas DataFrame) - DataFrame with added column with benefits for each student (column 'b')
    """
    
    thresh = np.percentile(df_true_predict['y_pred'], 100-threshold)
    df_true_predict['y_pred_label'] = np.where(df_true_predict['y_pred'] >= thresh, 1, 0)
    
    df_true_predict['b'] = df_true_predict['y_pred_label'] - df_true_predict['y'] + 1
    
    return df_true_predict 



def calc_educ_benefits(df_true_predict, threshold):
    """
    Custom function to compute benefits for the generalized entropy index: 
    Assigns a benefit of 1 to true positive cases and a benefit of 0 to false negative cases.
    Any other cases are not taken into consideration.
    
    Args:
    df_true_predict (pandas DataFrame) - DataFrame containing true values (in column 'y') and 
                                         predicted values (in column 'y_pred') of y
    threshold (Float) - Threshold indicating which predicted dropout probability values are considered 
                        as label 1 (e.g., threshold=10 means the top 10% of students with the highest 
                        predicted probabilities are labeled as 1)
    
    Returns:
    df_true_predict (pandas DataFrame) - DataFrame with added column with benefits for each student (column 'b')
    """
    
    thresh = np.percentile(df_true_predict['y_pred'], 100-threshold)
    df_true_predict['y_pred_label'] = np.where(df_true_predict['y_pred'] >= thresh, 1, 0)
    
    df_true_predict['b'] = np.NaN
    
    # True positive
    df_true_predict['b'] = np.where((df_true_predict['y']==1)&(df_true_predict['y_pred_label']==1),
                                    1,
                                    df_true_predict['b'])
    
    # False negative
    df_true_predict['b'] = np.where((df_true_predict['y']==1)&(df_true_predict['y_pred_label']==0),
                                    0,
                                    df_true_predict['b'])
    
    return df_true_predict



def calc_educ_rank_benefits(df_true_predict, rank_thresholds):
    """
    Custom function for computing benefits used in the generalized entropy index: 
    Assigns benefits of 1 to true positive cases based on a student's risk rank 
    (where higher risk corresponds to a higher benefit) and assigns a benefit of 0 
    to false negative cases. Any other cases are not taken into consideration.
    
    Args:
    df_true_predict (pandas DataFrame) - DataFrame containing true values (in column 'y') and 
                                         predicted values (in column 'y_pred') of y
    rank_thresholds (List of floats) - Thresholds defining the segmentation of students' risk ranks. 
                                       Students with predicted dropout probabilities exceeding the maximum 
                                       threshold are assigned the highest risk rank, while those with predicted 
                                       dropout probabilities below the minimum threshold are assigned the lowest 
                                       risk rank. The number of ranks corresponds to the number of thresholds plus one.

    Returns:
    df_true_predict (pandas DataFrame) - DataFrame with added column with benefits for each student (column 'b')
    """
    
    df_true_predict['rank'] = 0

    rank_thresholds.append(100)

    for i in range(len(rank_thresholds)-1):
        thresh_1 = np.percentile(df_true_predict['y_pred'], rank_thresholds[i])
        thresh_2 = np.percentile(df_true_predict['y_pred'], rank_thresholds[i+1])
    
        df_true_predict['rank'] = np.where((df_true_predict['y_pred']>thresh_1)&(df_true_predict['y_pred']<=thresh_2),
                                           i+1,
                                           df_true_predict['rank'])
    
    df_true_predict['b'] = np.NaN
    
    # True positive
    df_true_predict['b'] = np.where((df_true_predict['y']==1)&(df_true_predict['rank']>0),
                                    df_true_predict['rank'],
                                    df_true_predict['b'])
    
    # False negative
    df_true_predict['b'] = np.where((df_true_predict['y']==1)&(df_true_predict['rank']==0),
                                    0,
                                    df_true_predict['b'])
    
    return df_true_predict 



def gen_entropy_index(alpha, df_true_predict, groups_col, benefits_function, *args):
    """
    Computes the generalized entropy index according to the methodology outlined in 
    Till Speicher, Hoda Heidari, Nina Grgic-Hlaca, Krishna P. Gummadi, Adish Singla, Adrian
    Weller, and Muhammad Bilal Zafar. A unified approach to quantifying algorithmic
    unfairness: Measuring individual and group unfairness via inequality indices. In Proceedings
    of the 24th ACM SIGKDD International Conference on Knowledge Discovery
    amp; Data Mining, KDD ’18. ACM, July 2018. doi: 10.1145/3219819.3220046. URL
    http://dx.doi.org/10.1145/3219819.3220046.
    
    Args:
    alpha (Float) - Parameter of the generalized entropy index
    df_true_predict (pandas DataFrame) - DataFrame containing true values (in column 'y') and 
                                         predicted values (in column 'y_pred') of y
    groups_col (pandas Series) - Column specifying a group for each student (e.g., ethnicity groups)
    benefits_function (Function) - Function defining how to compute benefits for the generalized entropy index
    *args - Possible parameters of the 'benefit_function'
    
    Returns:
    gen_entropy (Float) - Generalized entropy index
    between_group  - Between-groups generalized entropy index
    """
    
    df_true_predict = df_true_predict.copy()
    df_true_predict = benefits_function(df_true_predict, *args)
    df_true_predict['groups'] = groups_col
    
    df_true_predict.dropna(inplace=True)
    
    df_true_predict['enthropy_col'] = (df_true_predict['b'] / df_true_predict['b'].mean())**alpha - 1
    gen_entropy = df_true_predict['enthropy_col'].sum() / len(df_true_predict) / alpha / (alpha - 1)
    
    df_true_predict['b_group'] = df_true_predict.groupby('groups')['b'].transform("mean")
    df_true_predict['enthropy_group_col'] = (df_true_predict['b_group'] / df_true_predict['b_group'].mean())**alpha - 1
    between_group = df_true_predict['enthropy_group_col'].sum() / len(df_true_predict) / alpha / (alpha - 1)
    
    
    return gen_entropy, between_group  