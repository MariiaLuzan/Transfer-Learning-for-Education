import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureAcadLoad(BaseEstimator, TransformerMixin):
    """
    Calculates an independent variable that can take 3 values: 
    'Full-Time', 'Less Full-Time', 'No units'
    
    Creates dummies 'Less Full-Time' and 'No Units'

    """
    
    def __init__(self) -> None:        
        pass
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        transformed_X['ACAD_LOAD'] = np.where(transformed_X['ACAD_LOAD_SHORT_DES']=='Full-Time',
                                              'Full-Time',
                                              'Less Full-Time')
        transformed_X['ACAD_LOAD'] = np.where(transformed_X['ACAD_LOAD_SHORT_DES']=='No Units',
                                              'No Units',
                                              transformed_X['ACAD_LOAD'])
        
        # Dummies
        transformed_X['ACAD_LOAD_Less_Full-Time'] = np.where(transformed_X['ACAD_LOAD']=='Less Full-Time',
                                                             1, 0)
        transformed_X['ACAD_LOAD_No_Units'] = np.where(transformed_X['ACAD_LOAD']=='No Units',
                                                       1, 0)
                                      
                
        return transformed_X