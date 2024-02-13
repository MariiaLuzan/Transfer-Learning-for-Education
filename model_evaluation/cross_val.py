import numpy as np
import pandas as pd
# Import metrics
from sklearn.metrics import roc_auc_score
from scipy.stats import kstest
from fairness.fairness_metrics import sliced_auc
from fairness.fairness_metrics import fairn_metrics
from fairness.customized_entropy import calc_educ_benefits
from fairness.customized_entropy import calc_educ_rank_benefits
from fairness.customized_entropy import gen_entropy_index



def calc_metrics(df_true_predict, X_test_imp, attribites_dic, 
                 treshold, rank_thresholds):
    """
    Estimates several metrics of classification accuracy and fairness for the 
    provided data set
    
    Args:
    df_true_predict  (pandas DataFrame)- DataFrame containing true and predicted y values 
    X_test_imp (pandas DataFrame) - DataFrame containing data set of features 
    index_name (String) - Name of the model to reflect it in a dataframe with metric results
    attribites_dic (dictionary) - Dictionary containing privileged values of attributes
                                  in the format {column name: privileged value} 
    treshold (float) - Threshold that defines the proportion of students who will receive
                       intervention from the university. For example, a threshold of 10% 
                       means that the bottom 10% of students with the highest predicted 
                       dropout probability will receive help (labeled as 1), while the 
                       remaining students will be labeled as 0 (needed for fairness metrics)
    rank_thresholds (List of floats) - Thresholds defining the segmentation of students' 
                                       risk ranks. Students with predicted dropout probabilities 
                                       exceeding the maximum threshold are assigned the highest 
                                       risk rank, while those with predicted dropout probabilities 
                                       below the minimum threshold are assigned the lowest risk 
                                       rank. The number of ranks corresponds to the number of 
                                       thresholds plus one.
    Returns:
    Tuple of metrics
    """
    
    # AUC
    auc = roc_auc_score(df_true_predict['y'], df_true_predict['y_pred'])
        
    # Pietra Index
    pietra = kstest(df_true_predict.loc[df_true_predict['y']==0, 'y_pred'], 
                    df_true_predict.loc[df_true_predict['y']==1, 'y_pred']).statistic
        
    # Kolmogorov-Smirnov p-value
    KS_pval = kstest(df_true_predict.loc[df_true_predict['y']==0, 'y_pred'], 
                     df_true_predict.loc[df_true_predict['y']==1, 'y_pred']).pvalue
        
    # Sliced AUCs
    sliced_auc_df = sliced_auc(df_true_predict['y'], 
                               df_true_predict['y_pred'], 
                               ['STDNT_ETHNC_grouped', 'STDNT_FEMALE'], 
                               X_test_imp)
            
    # Equal opportunity difference
    equal_opport_df = \
        fairn_metrics(df_true_predict['y'], df_true_predict['y_pred'], attribites_dic, X_test_imp, treshold)
        
    # Generalized entropy index
    gen_entropy, between_groups = \
        gen_entropy_index(2, df_true_predict, X_test_imp['STDNT_ETHNC_grouped'], calc_educ_benefits, treshold[0])
        
    # Generalized entropy index with ranks
    gen_entropy_ranks, between_groups_ranks = \
        gen_entropy_index(2, df_true_predict, X_test_imp['STDNT_ETHNC_grouped'], calc_educ_rank_benefits, rank_thresholds)
    
    return (auc, pietra, KS_pval, sliced_auc_df, equal_opport_df, gen_entropy, between_groups, \
            gen_entropy_ranks, between_groups_ranks)  



def cv_splits_results(metrics, col, index_name, n_splits,
                      auc_list, pietra_list, KS_pval_list, sliced_auc_cv_res,
                      equal_opport_cv_res, gen_entropy_list, between_groups_list,
                      gen_entropy_ranks_list, between_groups_ranks_list):
    """
    Organizes cross-validation results into a more convenient format
    
    Args:
    metrics (list of strings) - List with names of cross-validation metrics 
    col (list of strings) - List of column names corresponding to cross-validation splits
                            (e.g., ['split_1', 'split_2', 'split_3'])
    index_name (String) - Name of the model to reflect it in dataframes with metric results
    n_splits (integer) - Number of cross-validation splits
    auc_list (list) - List of AUCs for all cross-validation splits 
    pietra_list (list) - List of Pietra indices for all cross-validation splits  
    KS_pval_list (list) - List of KS p-values indices for all cross-validation splits 
    sliced_auc_cv_res (pandas DataFrame) - DataFrame containing sliced AUCs for all cv splits
    equal_opport_cv_res  (pandas DataFrame) - DataFrame containing sliced EOD for all cv splits 
    gen_entropy_list (list) - List of generalized entropy indices for all cross-validation splits 
    between_groups_list (list) - List of between groups entropy for all cross-validation splits
    gen_entropy_ranks_list (list) - List of generalized entropy indices for students' ranks for 
                                    all cross-validation splits 
    between_groups_ranks_list  (list) - List of between groups entropy for students' ranks for 
                                        all cross-validation splits
    
    Returns:
    cv_res (dictionary) - Dictionary with cross-validation metrics in the format
                          {name of metric: value of metric}
    """
    
    cv_res = {}
    
    # AUC dataframe
    AUC_cv_res = pd.DataFrame(index=[index_name], columns=col)  
    AUC_cv_res.loc[index_name] = auc_list
    cv_res[metrics[0]] = AUC_cv_res

    # Pietra dataframe
    pietra_cv_res = pd.DataFrame(index=[index_name], columns=col)  
    pietra_cv_res.loc[index_name] = pietra_list
    cv_res[metrics[1]] = pietra_cv_res

    # KS p_val dataframe
    KS_pval_cv_res = pd.DataFrame(index=[index_name], columns=col)   
    KS_pval_cv_res.loc[index_name] = KS_pval_list
    cv_res[metrics[2]] = KS_pval_cv_res
    cv_res[metrics[2]][col] = cv_res[metrics[2]][col].apply(pd.to_numeric, downcast='float', errors='coerce')

    # Sliced AUC dataframe
    cv_res[metrics[3]] = sliced_auc_cv_res

    # Equal opportunity difference dataframe
    cv_res[metrics[4]] = equal_opport_cv_res

    # Generalized entropy index
    gen_entropy_cv_res = pd.DataFrame(index=['generalized_entropy', 'between_groups'], columns=col)  
    gen_entropy_cv_res.loc['generalized_entropy'] = gen_entropy_list
    gen_entropy_cv_res.loc['between_groups'] = between_groups_list
    cv_res[metrics[5]] = gen_entropy_cv_res

    # Generalized entropy index with ranks
    gen_entropy_ranks_cv_res = pd.DataFrame(index=['generalized_entropy_ranks', 'between_groups_ranks'], columns=col)  
    gen_entropy_ranks_cv_res.loc['generalized_entropy_ranks'] = gen_entropy_ranks_list
    gen_entropy_ranks_cv_res.loc['between_groups_ranks'] = between_groups_ranks_list
    cv_res[metrics[6]] = gen_entropy_ranks_cv_res

    for metric in metrics:
        cv_res[metric]['mean'] = cv_res[metric][col].mean(axis=1)
        cv_res[metric]['se'] = cv_res[metric][col].std(axis=1) / n_splits**0.5
        
    return cv_res



def direct_transfer_cv(target_imp, test_target_indx, clf_source, cols, 
                       index_name, attribites_dic, treshold, rank_thresholds):

    """
    Performs cross-validation for the direct transfer of the source model to
    the target dataset
    
    Args:
    target_imp (pandas DataFrame) - DataFrame containing target data set with imputed 
                                    missing values and WoE transformation
    test_target_indx (List of lists) - Indices of the testing sets for all cross-validation splits
    clf_source (sklearn estimator) - Fitted source model
    cols (List of strings) - List containing features names (column names) 
    index_name (String) - Name of the model to reflect it in dataframes with metric results 
    attribites_dic (dictionary) - Dictionary containing privileged values of attributes
                                  in the format {column name: privileged value} 
    treshold (float) - Threshold that defines the proportion of students who will receive
                       intervention from the university. For example, a threshold of 10% 
                       means that the bottom 10% of students with the highest predicted 
                       dropout probability will receive help (labeled as 1), while the 
                       remaining students will be labeled as 0 (needed for fairness metrics)
    rank_thresholds (List of floats) - Thresholds defining the segmentation of students' 
                                       risk ranks. Students with predicted dropout probabilities 
                                       exceeding the maximum threshold are assigned the highest 
                                       risk rank, while those with predicted dropout probabilities 
                                       below the minimum threshold are assigned the lowest risk 
                                       rank. The number of ranks corresponds to the number of 
                                       thresholds plus one.
    
    Returns:
    cv_res (dictionary) - Dictionary with cross-validation metrics in the format
                          {name of metric: value of metric}
    """
    
    # Name of columns
    col = []
    
    metrics = ['AUC', 'pietra_index', 'KS_pvalue', 'sliced_AUCs', 
               'equal_opportunity_diff', 'gen_entropy', 'gen_entropy_ranks']
    
    # Variables to keep metrics from all cross-validation splits
    auc_list = []
    pietra_list = []
    KS_pval_list = []
    sliced_auc_cv_res = None
    equal_opport_cv_res = None
    gen_entropy_list = []
    between_groups_list  = []
    gen_entropy_ranks_list = []
    between_groups_ranks_list  = []

    n_splits = len(test_target_indx)


    for i in range(n_splits):
        col.append('split_'+str(i))
    
        # Select the testing data set
        X_test_imp = target_imp.iloc[test_target_indx[i]].copy() 
        # Predict dropout probabilies using the source model
        y_pred = clf_source.predict_proba(X_test_imp[cols])
    
        # DataFrame containing true and predicted y
        df_true_predict = pd.DataFrame(data={'y': X_test_imp['y_ENROLLED_1_YEAR_LATER'], 'y_pred': y_pred[:,1]})
    
        # Calculate evaluation metrics for the split
        (auc, pietra, KS_pval, sliced_auc_df, equal_opport_df, \
         gen_entropy, between_groups, gen_entropy_ranks, between_groups_ranks) = \
            calc_metrics(df_true_predict, X_test_imp, attribites_dic, 
                         treshold, rank_thresholds)
        
        # Concatenate the results of splits
        auc_list.append(auc) # AUC
        pietra_list.append(pietra) # Pietra Index
        KS_pval_list.append(KS_pval) # Kolmogorov-Smirnov p-value
    
        # Sliced AUCs
        sliced_auc_df.rename(columns={'AUC': 'split_'+str(i)}, inplace=True)
        if type(sliced_auc_cv_res)==pd.core.frame.DataFrame:
            sliced_auc_cv_res = sliced_auc_cv_res.merge(sliced_auc_df, on=["Attribute", "Value"])
        else:
            sliced_auc_cv_res = sliced_auc_df.copy()
        
        # Equal opportunity difference
        equal_opport_df.rename(columns={'thresh_'+str(treshold[0])+'%': 'split_'+str(i)}, inplace=True)
        if type(equal_opport_cv_res)==pd.core.frame.DataFrame:
            equal_opport_cv_res = equal_opport_cv_res.merge(equal_opport_df, on=["Attribute", "Value"])
        else:
            equal_opport_cv_res = equal_opport_df.copy()
    
        gen_entropy_list.append(gen_entropy) # Generalized entropy index
        between_groups_list.append(between_groups) # Generalized entropy index
        gen_entropy_ranks_list.append(gen_entropy_ranks) # Generalized entropy index with ranks
        between_groups_ranks_list.append(between_groups_ranks) # Generalized entropy index with ranks

    cv_res = cv_splits_results(metrics, col, index_name, n_splits, 
                               auc_list, pietra_list, KS_pval_list, sliced_auc_cv_res,
                               equal_opport_cv_res, gen_entropy_list, between_groups_list,
                               gen_entropy_ranks_list, between_groups_ranks_list)
    
    return cv_res



def TrAdaBoost_transfer_cv(source_df, target_df, train_target_indx, test_target_indx, 
                           imputation_pipeline, TrAdaBoost_model, cols, 
                           index_name, attribites_dic, treshold, rank_thresholds):

    """
    Performs cross-validation for the TrAdaBoost transfer to the target dataset
    
    Args:
    source_df (pandas DataFrame) - DataFrame containing the source data set 
    target_df (pandas DataFrame) - DataFrame containing the target data set 
    train_target_indx (List of lists) - Indices of the training sets for all cross-validation splits
    test_target_indx (List of lists) - Indices of the testing sets for all cross-validation splits
    imputation_pipeline (sklearn pipeline) - Pipeline to impute missing values and apply WoE
                                             transformation
    TrAdaBoost_model (custom estimator) - Estimator of modified TrAdaBoost algorithm
    cols (List of strings) - List containing features names (column names) 
    index_name (String) - Name of the model to reflect it in dataframes with metric results 
    attribites_dic (dictionary) - Dictionary containing privileged values of attributes
                                  in the format {column name: privileged value} 
    treshold (float) - Threshold that defines the proportion of students who will receive
                       intervention from the university. For example, a threshold of 10% 
                       means that the bottom 10% of students with the highest predicted 
                       dropout probability will receive help (labeled as 1), while the 
                       remaining students will be labeled as 0 (needed for fairness metrics)
    rank_thresholds (List of floats) - Thresholds defining the segmentation of students' 
                                       risk ranks. Students with predicted dropout probabilities 
                                       exceeding the maximum threshold are assigned the highest 
                                       risk rank, while those with predicted dropout probabilities 
                                       below the minimum threshold are assigned the lowest risk 
                                       rank. The number of ranks corresponds to the number of 
                                       thresholds plus one.
    
    Returns:
    cv_res (dictionary) - Dictionary with cross-validation metrics in the format
                          {name of metric: value of metric}
    """
    
    # Name of columns
    col = []
    
    metrics = ['AUC', 'pietra_index', 'KS_pvalue', 'sliced_AUCs', 
               'equal_opportunity_diff', 'gen_entropy', 'gen_entropy_ranks']
    
    # Variables to keep metrics from all cross-validation splits
    auc_list = []
    pietra_list = []
    KS_pval_list = []
    sliced_auc_cv_res = None
    equal_opport_cv_res = None
    gen_entropy_list = []
    between_groups_list  = []
    gen_entropy_ranks_list = []
    between_groups_ranks_list  = []

    n_splits = len(test_target_indx)


    for i in range(n_splits):
        col.append('split_'+str(i))
    
        # Training data set
        X_train = pd.concat([source_df, target_df.iloc[train_target_indx[i]]])
        X_train.reset_index(drop=True, inplace=True)
        # Impute missing values and apply WoE transformation
        X_train_imp = imputation_pipeline.fit_transform(X_train, X_train['y_ENROLLED_1_YEAR_LATER'])
        
        # Testing data set
        X_test = target_df.iloc[test_target_indx[i]]
        X_test_imp = imputation_pipeline.transform(X_test)
        
        # Fit the model on the training data set
        TrAdaBoost_model.fit(X_train_imp[cols], X_train_imp['y_ENROLLED_1_YEAR_LATER'], 
                             source_indices=list(range(len(source_df))))
        
        # Predict dropout probabilities for the testing data set
        y_pred = TrAdaBoost_model.predict_proba(X_test_imp[cols])
        y_pred = y_pred.to_numpy()
    
        # DataFrame containing true and predicted y
        df_true_predict = pd.DataFrame(data={'y': X_test_imp['y_ENROLLED_1_YEAR_LATER'], 'y_pred': y_pred})
    
        # Calculate evaluation metrics for the split
        (auc, pietra, KS_pval, sliced_auc_df, equal_opport_df, \
         gen_entropy, between_groups, gen_entropy_ranks, between_groups_ranks) = \
            calc_metrics(df_true_predict, X_test_imp, attribites_dic, 
                         treshold, rank_thresholds)
        
        # Concatenate the results of splits
        auc_list.append(auc) # AUC
        pietra_list.append(pietra) # Pietra Index
        KS_pval_list.append(KS_pval) # Kolmogorov-Smirnov p-value
    
        # Sliced AUCs
        sliced_auc_df.rename(columns={'AUC': 'split_'+str(i)}, inplace=True)
        if type(sliced_auc_cv_res)==pd.core.frame.DataFrame:
            sliced_auc_cv_res = sliced_auc_cv_res.merge(sliced_auc_df, on=["Attribute", "Value"])
        else:
            sliced_auc_cv_res = sliced_auc_df.copy()
        
        # Equal opportunity difference
        equal_opport_df.rename(columns={'thresh_'+str(treshold[0])+'%': 'split_'+str(i)}, inplace=True)
        if type(equal_opport_cv_res)==pd.core.frame.DataFrame:
            equal_opport_cv_res = equal_opport_cv_res.merge(equal_opport_df, on=["Attribute", "Value"])
        else:
            equal_opport_cv_res = equal_opport_df.copy()
    
        gen_entropy_list.append(gen_entropy) # Generalized entropy index
        between_groups_list.append(between_groups) # Generalized entropy index
        gen_entropy_ranks_list.append(gen_entropy_ranks) # Generalized entropy index with ranks
        between_groups_ranks_list.append(between_groups_ranks) # Generalized entropy index with ranks

    cv_res = cv_splits_results(metrics, col, index_name, n_splits, 
                               auc_list, pietra_list, KS_pval_list, sliced_auc_cv_res,
                               equal_opport_cv_res, gen_entropy_list, between_groups_list,
                               gen_entropy_ranks_list, between_groups_ranks_list)
    
    return cv_res