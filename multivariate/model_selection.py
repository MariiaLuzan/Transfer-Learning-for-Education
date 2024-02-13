import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import ast

from fairness.customized_entropy import calc_educ_rank_benefits
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score



# sign
def check_model_signs(models_df, factors_sign_df, clf, sample_train_imp):
    """
    Verifies the coefficient signs for the models listed in the 'models_df' DataFrame 
    
    Args:
    models_df (pandas DataFrame) - DataFrame containing the models to be checked, where each model 
                                   is represented as a list of factor names
    factors_sign_df (pandas DataFrame) - DataFrame containing the correct sign for each factor
    clf (sklearn LogisicRegression) - Logistic regression model with specified parameters
    sample_train_imp (pandas DataFrame) - DataFrame containing the dataset containing both X and y variables
    
    Returns:
    models (pandas DataFrame) - DataFrame containing the models for verification, with an additional column 
                                'Correct_signs'. This column takes the value 1 if all the coefficients' signs 
                                within the models are correct, and 0 otherwise.
    """
    
    models = models_df.copy()
    models['Correct_signs'] = np.NaN
    
    for ind in range(len(models)):
        # Select a model
        model = models.loc[ind, 'MODEL']
        
        # Fit selected model
        clf.fit(sample_train_imp[model], sample_train_imp['y_ENROLLED_1_YEAR_LATER'])
        
        # Model coefficients
        model_factors = pd.DataFrame(data={'Factor': model, 'Coefficient': clf.coef_[0]})
        model_factors = model_factors.merge(factors_sign_df, how='left', on='Factor')
        model_factors.dropna(inplace=True)
        
        # Check coefficients' signs
        model_factors['Coef_check'] = model_factors['Coefficient'] * model_factors['Coef_sign']
        # Number of factors with incorrect signs
        number_coef = len(model_factors[model_factors['Coef_check']<0])
        
        if number_coef>0:
            correct_signs = 0
        else:
            correct_signs = 1
        
        models.loc[ind, 'Correct_signs'] = correct_signs
        
        if ind % 1000 == 0:
            print(ind)
        
        
    return models  




# cross-validation

def scorer_entropy_index(y, y_pred, alpha, benefits_function, benefits_function_param):
    """
    Custom scorer used for sklearn cross-validation, which calculates the generalized entropy index
    
    Args:
    y (pandas Series) - True values of y 
    y_pred (pandas Series) - Tredicted values of y 
    alpha (Integer) - alpha parameter for the generalized entropy index  
    benefits_function (Function) - Function to assess benefits for each student 
    benefits_function_param (Integer of List of floats) - Parameter of the 'benefits_function'
    
    Returns:
    gen_entropy (Float) - Generalized entropy index
    """
    
    
    # DataFrame containing true and predicted y
    df_true_predict = pd.DataFrame(data={'y': y, 'y_pred': y_pred})
    
    df_true_predict = benefits_function(df_true_predict, benefits_function_param)
        
    df_true_predict.dropna(inplace=True)
    
    df_true_predict['enthropy_col'] = (df_true_predict['b'] / df_true_predict['b'].mean())**alpha - 1
    gen_entropy = df_true_predict['enthropy_col'].sum() / len(df_true_predict) / alpha / (alpha - 1)
    
       
    return gen_entropy


def cross_validate_AUC_entropy(sample_train, models_filtered, num_folds, pipeline,
                               alpha, benefits_function, benefits_function_param):
    """
    Performs cross-validation on the specified models within the DataFrame "models_filtered" 
    providing the cross-validation AUC and generalized entropy index for each model 
    
    Args:
    sample_train (Pandas DataFrame) - DataFrame containing the dataset (both X and y)
    models_filtered (Pandas DataFrame) - DataFrame containing the selected models for cross-validation, 
                                         with each model represented as a list of factor names
    num_folds (Integer) - Number of folds for performing cross-validation
    pipeline (Pipeline) - Pipeline that includes all steps of the estimator (such as imputation of missing values, 
                          Weight of Evidence transformation, and the classification model)
    alpha (Integer) - alpha parameter for the generalized entropy index 
    benefits_function (Function) - Parameter for the generalized entropy index - a function to assess benefits for each student  
    benefits_function_param (Integer of List of floats) - Parameter of the 'benefits_function'
    
    Returns: 
    models (Pandas DataFrame) - DataFrame containing cross-validation scores for the specified models
    """
    
    models = models_filtered.copy()
    # Add columns for cross-validation scores
    models['test_AUC_mean'] = np.NaN 
    models['test_AUC_se'] = np.NaN # standard error
    models['train_AUC_mean'] = np.NaN
    models['test_Gen entropy_mean'] = np.NaN
    models['test_Gen entropy_se'] = np.NaN
    
    # Loop through the specified models
    #for ind in range(len(models_filtered)):
    for ind in models_filtered.index:
        model = models.loc[ind, 'MODEL']
        if type(model)!=list:
            model = ast.literal_eval(model)
            
        # Update the pipeline by incorporating the factors specific to the model
        pipeline.set_params(filter_columns__list_of_factors=model)
            
        # Define scorers for the cross-validation
        auc = make_scorer(roc_auc_score, needs_proba=True)
        scorer_entropy = make_scorer(scorer_entropy_index, needs_proba=True,
                                     alpha=alpha, benefits_function=benefits_function, 
                                     benefits_function_param=benefits_function_param)
        
        # Perform cross-validation
        cv_results = cross_validate(pipeline, sample_train, sample_train['y_ENROLLED_1_YEAR_LATER'], 
                                    scoring={'AUC': auc, 'Generalized entropy': scorer_entropy}, 
                                    cv=num_folds, return_train_score=True, n_jobs=-1)
        models.loc[ind, 'test_AUC_mean'] = cv_results['test_AUC'].mean()
        models.loc[ind, 'test_AUC_se'] = cv_results['test_AUC'].std() / num_folds**0.5
        models.loc[ind, 'train_AUC_mean'] = cv_results['train_AUC'].mean()
        models.loc[ind, 'test_Gen entropy_mean'] = cv_results['test_Generalized entropy'].mean()
        models.loc[ind, 'test_Gen entropy_se'] = cv_results['test_Generalized entropy'].std() / num_folds**0.5
        
        if ind % 100 == 0:
            print(ind)
    
    return models



def plot_cv_selected_models(model_indices, models_cv_res_, bins,
                            metrics, metric_names, n_row, n_col,
                            figsize):
    """
    Plots cross-validation metrics of selected models against the distribution of metrics
    for all potential models
    
    Args:
    model_indices (List of integers) - List of indices of selected models 
                                       (from column 'index' in the dataframe 'models_cv_res_')
    models_cv_res_ (pandas DataFrame) - Dataframe containing cross-validation results for models
    metrics (List of strings) - List of metrics to display (names of columns containing metrics in 
                                the dataframe 'models_cv_res_'). 
    metric_names (List of strings) - List of metrics names to display on the plot
    n_row (Integer) - Number of rows of subplots 
    n_col (Integer) - Number of columns of subplots 
    figsize (Tuple of floats) - Figure size
    """
    
    fig, axs = plt.subplots(n_row, n_col, figsize=figsize)
    model_colors = ['royalblue', 'red', 'midnightblue']

    # Loop through metrics
    for i in range(n_col):
        for j in range(n_row):
        
            metric_ind = i + 2*j
        
            axs[j, i].hist(models_cv_res_[metrics[metric_ind]], bins=bins[metric_ind], 
                           density=True, alpha=0.3, color='slategrey',
                           label='Metric distribution')
            axs[j, i].title.set_text(metric_names[metric_ind])
    
            # Loop through models
            for model_ind in range(len(model_indices)):
                axs[j,i].axvline(
                    x=models_cv_res_.loc[models_cv_res_['index']==model_indices[model_ind], 
                                         metrics[metric_ind]].values[0], 
                    label= 'Model_'+str(model_ind+1), color=model_colors[model_ind], 
                    linewidth=2)

    plt.suptitle('Cross-validation test set metrics: selected models vs. alternative candidates', 
                 fontsize=14)

    # Add more space between subplots
    plt.subplots_adjust(left=0.1,
    bottom=0.11,
    right=0.9,
    top=0.88,
    wspace=0.2,
    hspace=0.4)

    # Add a legend
    lines_labels = axs[0,0].get_legend_handles_labels()
    labels = lines_labels[1]
    plt.figlegend(labels, loc = 'lower center', ncol=len(model_indices)+1);