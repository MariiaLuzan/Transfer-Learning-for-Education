import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CombineEthnicGroups(BaseEstimator, TransformerMixin):
    """
    Combine ethnic groups, excluding white and asian students, as they lack a sufficient 
    number of dropouts for a meaningful fairness analysis

    """
    
    def __init__(self) -> None:    
        pass
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        # All students except white and asian
        transformed_X['STDNT_ETHNC_grouped'] = 2
        
        # White students
        transformed_X['STDNT_ETHNC_grouped'] = \
            np.where(transformed_X['STDNT_ETHNC_GRP_CD']==1, 0, transformed_X['STDNT_ETHNC_grouped'])
        
        # Asian students
        transformed_X['STDNT_ETHNC_grouped'] = \
            np.where(transformed_X['STDNT_ETHNC_GRP_CD']==4, 1, transformed_X['STDNT_ETHNC_grouped'])
                
        return transformed_X