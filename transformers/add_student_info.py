import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AddStudentInfo(BaseEstimator, TransformerMixin):
    """
    Adds student info from the table STDNT_INFO
    """
    
    def __init__(self, student_info:pd.DataFrame) -> None:        
        self.student_info = student_info
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        transformed_X = transformed_X.merge(self.student_info, right_on='STDNT_ID', left_index=True)
                
        return transformed_X