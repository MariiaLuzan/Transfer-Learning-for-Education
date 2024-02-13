import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureStateResident(BaseEstimator, TransformerMixin):
    """
    Calculates an independent variable that denotes the residency of a student:
        1: In State
        0: Out of State

    """
    
    def __init__(self) -> None:    
        pass
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        transformed_X['IN_STATE_RESIDENT'] = np.where(transformed_X['RES_SHORT_DES'] =='In State', 1, 0)
                
        return transformed_X