import numpy as np
import pandas as pd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin


class TrAdaBoost(BaseEstimator, ClassifierMixin):
    
    """
    Custom TrAdaBoost transfer learning estimator based on the work of 
    Wenyuan Dai, Qiang Yang, Gui-Rong Xue, and Yong Yu "Boosting for transfer learning" 
    (ICML '07). Link to the paper:  https://doi.org/10.1145/1273496.1273521

    This implementation introduces four modifications:

    1. Only source data points are weighted
    2. All weak learners are considered in prediction, as proposed by Xin J. Hunt, 
       Ilknur Kaynar Kabul, and Jorge Silva in "Transfer learning for education data"
       (ACM SIGKDD Conference, 2017). The paper is available at: 
       https://doi.org/10.1145/nnnnnnn.nnnnnnn.
    3. Iterations are stopped as soon as the error on the target data set starts to increase
    4. The 'predict' method has been adapted to return probabilities instead of labels.
   
    
    Params:
    __init__:
        estimator (sklearn estimator) - Estimator to use for fitting weak learners 
        max_num_iterations (integer) - Maximum number of iterations (or weak learners) 
        print_mes (boolean) - Indicator whether to produce messages regarding the fitting process
    fit:
        source_indices (list) - Indices of source data points in the X dataset
    """
    
    def __init__(self, estimator, max_num_iterations, print_mes=False):
        
        self.estimator = estimator
        self.max_num_iterations = max_num_iterations
        self.print_mes = print_mes
                
    
    def estimate_params(self, X, y, source_indices, target_indices, num_iterations, print_mes):
        
        errors = []
        iterations_weights = []
        estimators = []
        
        # Number of data points in the source and target data sets
        n = len(source_indices)
        m = len(X) - n
        
        # DataFrame to perform all calculation
        # Initially all data points have equal weights
        calc_df = pd.DataFrame(data={'y': y, 'weights': 1})
        beta_source = 1 / (1 + (2 * np.log(n)/num_iterations)**0.5)
                
        # Loop through iterations       
        for iteration in range(num_iterations):
            
            # Adjust the weights to ensure that their cumulative sum equals the total number of points
            calc_df['weights'] = calc_df['weights'] * (n+m) / calc_df['weights'].sum()
                                                
            # Fit the estimator
            estimator_iteration = clone(self.estimator)
            estimator_iteration.fit(X, y, calc_df['weights'])
            calc_df['y_pred'] = estimator_iteration.predict_proba(X)[:, 1]
            
            
            calc_df['error'] = np.where(calc_df.index.isin(target_indices), 
                                        np.abs(calc_df['y'] - calc_df['y_pred']),
                                        0)
            
            error_target = calc_df['error'].sum() / m
            
            if print_mes:
                print('Iteration # ' + str(iteration) + ': Target weights = ' + \
                      str(calc_df.loc[calc_df.index.isin(target_indices), 'weights'].sum() / (n+m)) + \
                      ". Error on target = " + str(error_target))
            
            errors.append(error_target)
            estimators.append(estimator_iteration)
            iterations_weights.append(error_target / (1-error_target))
            
            # Update weights for data points
            calc_df['weights'] = np.where(calc_df.index.isin(target_indices), 
                                          calc_df['weights'],
                                          calc_df['weights'] * beta_source**(np.abs(calc_df['y'] - calc_df['y_pred'])))
        
        return errors, iterations_weights, estimators
        
    
    def find_num_iter(self, X, y, source_indices, target_indices):
        
        for i in range(2, self.max_num_iterations+1):
            errors, _, _ = self.estimate_params(X, y, source_indices, target_indices, i, False)
            # Check that each iteration improves the error on the target
            if errors[-1] > errors[-2]:
                return(i-1)
        return i        
    
    
    def fit(self, X, y, source_indices):
        
        # Define target indices
        full_set_indices = set(X.index)
        source_set_indices = set(source_indices)
        target_set_indices = full_set_indices - source_set_indices
        target_indices = list(target_set_indices)
        target_indices.sort()
        
        # Find the number of iterations
        self.num_iterations = self.find_num_iter(X, y, source_indices, target_indices)
        
        _, self.iterations_weights_, self.estimators_ = \
            self.estimate_params(X, y, source_indices, target_indices, self.num_iterations, self.print_mes)
        
        return self
    
    
    def predict_proba(self, X):
        
        # Check if the trAdaBoost is fitted
        # https://scikit-learn.org/stable/modules/generated/sklearn.utils.validation.check_is_fitted.html
        check_is_fitted(self)
        
                       
        # Create dataframe for models results
        # Names of columns
        y_cols = []
        y_weights_cols = []
        num_iterations = min(self.num_iterations, len(self.estimators_))
        
        for i in range(0, num_iterations):
            y_cols.append('y_'+str(i))
            y_weights_cols.append('y_'+str(i)+'_weight')
        all_cols = y_cols + y_weights_cols    
        
        models_pred = pd.DataFrame(index=range(len(X)), columns=all_cols)
        
        models_pred['res_numerator'] = 0
        denominator = 0
        
        
        for iteration in range(0, num_iterations):
            if self.print_mes:
                print('Iteration # ' + str(iteration) +': iteration weight=' + str(self.iterations_weights_[iteration]))
            
            models_pred['y_'+str(iteration)] = self.estimators_[iteration].predict_proba(X)[:,1]
            models_pred['y_'+str(iteration)+'_weight'] = \
                models_pred['y_'+str(iteration)] * np.log(self.iterations_weights_[iteration]) 
            
            models_pred['res_numerator'] = models_pred['res_numerator'] + models_pred['y_'+str(iteration)+'_weight']
            denominator += np.log(self.iterations_weights_[iteration])
            
                    
        models_pred['proba'] = models_pred['res_numerator'] / denominator
        
        # models_pred.to_excel('models_pred.xlsx')
        
        return models_pred['proba']
    
    
    def get_num_iterations(self):
        return self.num_iterations
    
    
    def predict(self, X):
        err_mes = "The 'predict' method is not implemented for this estimator. Use the 'predict_proba' method instead."
        raise ValueError(err_mes)
                         
                         