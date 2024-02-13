import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FilterColumns(BaseEstimator, TransformerMixin):
    """
    Selects specific columns from the dataframe
    
    Args:
    list_of_factors (List of strings) - List containing the names of columns that need to be selected

    """
    
    def __init__(self, list_of_factors) -> None:    
        self.list_of_factors = list_of_factors
        
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        transformed_X = transformed_X[self.list_of_factors]
                
        return transformed_X