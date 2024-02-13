import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from statsmodels.distributions.empirical_distribution import ECDF



def fit_test_model(pipeline, sample_train, sample_test=None):
    
    """
    Fits the estimator 'pipeline' using the dataframe 'sample_train'
    and assesses its performance on the 'sample_test' test set
    
    Args:
    pipeline (sklearn Pipeline) - Estimator to fit and evaluate
    sample_train (pandas DataFrame) - DataFrame containing the training data set 
    sample_test (pandas DataFrame) - DataFrame containing the testing data set
    
    Returns:
    aucs (dictionary) - Dictionary containing estimated AUCs for the training and testing data sets
    model_coef (pandas DataFrame) - DataFrame containing the coefficients of the model
    """
    
    pipeline.fit(sample_train, sample_train['y_ENROLLED_1_YEAR_LATER'])
    
    # Model coefficients
    cols = pipeline[1].get_params()['list_of_factors']
    model_coef = pd.DataFrame(data={'factors': cols, 'coef': pipeline[2].coef_[0]})
    
    # Training set AUC
    y_pred = pipeline.predict_proba(sample_train)
    auc_train = roc_auc_score(sample_train['y_ENROLLED_1_YEAR_LATER'], y_pred[:,1])
    
    # Testing set AUC
    if type(sample_test)==pd.core.frame.DataFrame:
        y_pred = pipeline.predict_proba(sample_test)
        auc_test = roc_auc_score(sample_test['y_ENROLLED_1_YEAR_LATER'], y_pred[:,1])
        aucs = {'auc_train': auc_train, 'auc_test': auc_test}
    else:        
        aucs = {'auc_train': auc_train}
    
    return aucs, model_coef



def plot_roc_curves(df_true_predict_1, df_true_predict_2,
                    label_1, label_2, title):
    """
    Plots ROC curves for two models
    
    Params:
    df_true_predict_1 (pandas DataFrame) - DataFrame with true and predicted y values for the first model
    df_true_predict_2 (pandas DataFrame) - DataFrame with true and predicted y values for the second model
    label_1 (String) - Label for the first ROC curve
    label_2 (String) - Label for the second ROC curve
    title (String) - Graph title   
    """
    
    # ROC curves
    fpr_1, tpr_1, thresholds_1 = roc_curve(df_true_predict_1['y'], 
                                           df_true_predict_1['y_pred'], 
                                           pos_label=1)

    fpr_2, tpr_2, thresholds_2 = roc_curve(df_true_predict_2['y'], 
                                           df_true_predict_2['y_pred'], 
                                           pos_label=1)
    
    plt.plot(fpr_1, tpr_1, label=label_1)
    plt.plot(fpr_2, tpr_2, label=label_2)
    plt.xlabel('False positive rate', fontsize=12)
    plt.ylabel('True positive rate', fontsize=12)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11);
    
    
    
def calc_y_mean(df_true_predict, percent_list):
    
    """
    Calculates the observed probability of dropout for student groups, categorized 
    based on the predicted probability of dropout. These student groups are delineated 
    by the 'percent_list' parameter.
       
    Args:
    df_true_predict (pandas DataFrame)- DataFrame containing true and predicted y values
    percent_list (List of floats) - List of percentiles defining student groups
    
    Returns:
    y_mean_list (List of floats) - List containing estimated observed dropout probabilities 
                                   for different student groups.
    """
    
    y_mean_list = []
    
    for i in range(len(percent_list)-1):
        
        y_mean = \
        df_true_predict.loc[(df_true_predict['y_pred'] >= np.percentile(df_true_predict['y_pred'], percent_list[i]))&\
                            (df_true_predict['y_pred'] < np.percentile(df_true_predict['y_pred'], percent_list[i+1])), 
                            'y'].mean()
        
        y_mean_list.append(y_mean)
        
    return y_mean_list     



def KS_test_comparison(ks_test_direct, ks_test_IWS):
    """
    Organizes the outcomes of the Kolmogorov-Smirnov (KS) test for two models 
    (direct transfer and IWS) into a pandas DataFrame
    
    Args:
    ks_test_direct - Results of the KS test for direct transfer
    ks_test_IWS - Results of the KS test for IWS transfer
    
    Returns:
    ks_test_res (pandas DataFrame) - DataFrame containing the results of the KS test 
                                     for both direct and IWS transfer
    """
    ks_test_res = pd.DataFrame(index = ['Direct tranfer', 'Instance weighting strategy'], 
                               columns = ['Pietra index', 'KS p_val'])
    ks_test_res.loc['Direct tranfer', 'Pietra index'] = ks_test_direct.statistic
    ks_test_res.loc['Direct tranfer', 'KS p_val'] = ks_test_direct.pvalue
    ks_test_res.loc['Instance weighting strategy', 'Pietra index'] = ks_test_IWS.statistic
    ks_test_res.loc['Instance weighting strategy', 'KS p_val'] = ks_test_IWS.pvalue
    ks_test_res['KS p_val'] = ks_test_res['KS p_val'].apply(pd.to_numeric, downcast='float', errors='coerce')
    return ks_test_res



def plot_KS_CDF(df_true_predict, df_true_predict_w):
    
    """
    Plots cumulative distribution functions of predicted probanilities of
    dropouts for students enrolled in the following year and for students 
    who dropped out.
    
    Args:
    df_true_predict (pandas DataFrame)- DataFrame containing true and predicted y values
                                        for direct transfer
    df_true_predict_w  (pandas DataFrame)- DataFrame containing true and predicted y values
                                           for IWS transfer
    
    """
    
    # Cumulative distribution functions for students enrolled in the 
    # following year and for students who dropped out (direct transfer)
    ecdf_0 = ECDF(df_true_predict.loc[df_true_predict['y']==0, 'y_pred'])
    ecdf_1 = ECDF(df_true_predict.loc[df_true_predict['y']==1, 'y_pred'])
    
    # Cumulative distribution functions for students enrolled in the 
    # following year and for students who dropped out (IWS)
    ecdf_w_0 = ECDF(df_true_predict_w.loc[df_true_predict_w['y']==0, 'y_pred'])
    ecdf_w_1 = ECDF(df_true_predict_w.loc[df_true_predict_w['y']==1, 'y_pred'])
    
    x = np.linspace(0, 1, num=5000)

    plt.plot(x, ecdf_1(x), label = 'Direct transfer: Dropped out students')
    plt.plot(x, ecdf_w_1(x), label = 'IWS: Dropped out students')


    plt.plot(x, ecdf_0(x), label = 'Direct transfer: Enrolled students', color='grey')
    plt.plot(x, ecdf_w_0(x), label = 'IWS: Enrolled students', color='black')


    plt.title('CDFs of predicted dropout probabilities for\nenrolled and discontinued students',
              fontsize=14)
    plt.xlabel('Predicted probability of dropout', fontsize=12)
    plt.ylabel('Frequency', fontsize=11)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(fontsize=12);
    
    
    
def plot_dropout_rate(df_true_predict, percent_list, 
                      names_list, title):
    """
    Plots observed dropout probabilities for students
    grouped by their predicted dropout probabilities
    
    Args:
    df_true_predict (pandas DataFrame)- DataFrame containing true and predicted y values 
    percent_list (List of floats) - List of percentiles defining student groups 
    names_list (List of strings) - X axis labels 
    title (String) - Graph title
    """
    
    y_mean_list = calc_y_mean(df_true_predict, percent_list)
    bar = plt.bar(names_list, y_mean_list)
    plt.bar_label(bar, fmt=lambda x: '{:.2f}%'.format(x * 100), fontsize=9)

    plt.title(title)
    plt.xlabel('Percentiles of students sorted by their predicted probability of dropout')
    plt.ylabel('Observed probability of dropout');  
    
    
    
def plot_dropout_rate_two_models(df_true_predict, df_true_predict_w, 
                                 percent_list, names_list):
    """
    Plots observed dropout probabilities for students
    grouped by their predicted dropout probabilities
    for two models
    
    Args:
    df_true_predict (pandas DataFrame)- DataFrame containing true and predicted y values 
                                        for the first model
    df_true_predict_w (pandas DataFrame)- DataFrame containing true and predicted y values 
                                        for the second model
    percent_list (List of floats) - List of percentiles defining student groups 
    names_list (List of strings) - X axis labels 
    """
    
    y_mean_list = calc_y_mean(df_true_predict, percent_list)
    y_mean_list_w = calc_y_mean(df_true_predict_w, percent_list)
    
    x_axis = np.arange(len(y_mean_list))
    plt.bar(x_axis-0.2, y_mean_list, width=0.4, label='Direct transfer')
    plt.bar(x_axis+0.2, y_mean_list_w, width=0.4, label='Instance weighting strategy')

    plt.legend()
    plt.xticks(x_axis, names_list)

    plt.title('Observed probabilty of dropout for students \ngrouped by their predicted probability of dropout')
    plt.xlabel('Percentiles of students sorted by their predicted probability of dropout')
    plt.ylabel('Observed probability of dropout');    
    
    
    
def gen_entropy_comparison(gen_entropy_direct, gen_entropy_IWS):
    """
    Organizes the outcomes of generalized entropy for two models (direct transfer and IWS)
    into a pandas DataFrame
    
    Args:
    gen_entropy_direct (Tuple of floats) - Results for direct transfer (the first element is a generalized 
                                           entropy index, the second element is the between-groups entropy)
    gen_entropy_IWS (Tuple of floats) - Results for IWS transfer (the first element is a generalized entropy 
                                        index, and the second element is the between-groups entropy)
    
    Returns:
    gen_entropy_res (pandas DataFrame) - DataFrame containing the generalized entropy results for both 
                                         direct and IWS transfer.
    """
    gen_entropy_res = pd.DataFrame(index = ['Direct tranfer', 'Instance weighting strategy'], 
                                   columns = ['Generalized entropy', 'Between groups entropy'])
    gen_entropy_res.loc['Direct tranfer', 'Generalized entropy'] = gen_entropy_direct[0]
    gen_entropy_res.loc['Direct tranfer', 'Between groups entropy'] = gen_entropy_direct[1]
    gen_entropy_res.loc['Instance weighting strategy', 'Generalized entropy'] = gen_entropy_IWS[0]
    gen_entropy_res.loc['Instance weighting strategy', 'Between groups entropy'] = gen_entropy_IWS[1]
    
    return gen_entropy_res    