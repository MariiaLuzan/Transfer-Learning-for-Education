import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureGPA(BaseEstimator, TransformerMixin):
    """
    Creates 
    - High School GPA Features: float ('HS_GPA') and categorical ('HS_GPA_BIN')
    - First term GPA Festures: categorical ('CURR_GPA_BIN')

    """
    
    def __init__(self, hs_gpa_bins, first_term_gpa_bins) -> None:    
        self.hs_gpa_bins = hs_gpa_bins
        self.first_term_gpa_bins = first_term_gpa_bins
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        # If HS_GPA=0, it is a mistake, change it to NaN
        transformed_X['HS_GPA'] = np.where(transformed_X['HS_GPA']==0, np.nan, transformed_X['HS_GPA'])
               
        # Student's GPA as a categorical variable
        transformed_X['HS_GPA_BIN'] = pd.cut(transformed_X['HS_GPA'], self.hs_gpa_bins)
        transformed_X['CURR_GPA_BIN'] = pd.cut(transformed_X['CURR_GPA'], self.first_term_gpa_bins)
                
        return transformed_X