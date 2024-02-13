import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureIsFemale(BaseEstimator, TransformerMixin):
    """
    Calculates an independent variable that denotes the gender of a student:
        1: Female
        0: Not female

    """
    
    def __init__(self) -> None:        
        pass
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        transformed_X['STDNT_FEMALE'] = np.where(transformed_X['STDNT_SEX_CD']==1, 1, 0)
                
        return transformed_X