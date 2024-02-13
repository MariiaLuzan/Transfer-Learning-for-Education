import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureAge(BaseEstimator, TransformerMixin):
    """
    Calculates student's age at the beginning of a term
    Creates 2 features: float ('STDNT_AGE') and categorical ('STDNT_AGE')
    """
    
    term_dic = {'FA': 9, # The FALL term starts in September
                'WN': 1, # The WINTER term starts in January
                'SP': 5,
                'SU': 7,
                'SS': 5} 
    
    def __init__(self, age_bins) -> None:    
        #self.term_dic = {'FA': 9, # The FALL term starts in September
                         #'WN': 1, # The WINTER term starts in January
                         #'SP': 5,
                         #'SU': 7,
                         #'SS': 5} 
        self.age_bins = age_bins
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        # Add a proxi of a student birth date
        transformed_X['STDNT_BIRTH_MO'] = transformed_X['STDNT_BIRTH_MO'].astype(int)
        transformed_X['STDNT_BIRTH_YR'] = transformed_X['STDNT_BIRTH_YR'].astype(int)
        
        transformed_X["STDNT_BIRTH_DAT"] = transformed_X['STDNT_BIRTH_YR'].astype(str)+\
                                           transformed_X['STDNT_BIRTH_MO'].astype(str)
        transformed_X["STDNT_BIRTH_DAT"] = pd.to_datetime(transformed_X["STDNT_BIRTH_DAT"], format='%Y%m')
        
        
        # Add a column with the start month of a term
        transformed_X['SEMESTER_MO'] = transformed_X['SEMESTER']
        transformed_X['SEMESTER_MO'] = transformed_X['SEMESTER_MO'].map(self.term_dic)
    
        # Add start date of a term
        transformed_X['TERM_START_DAT'] = pd.to_datetime(
            transformed_X['YEAR'].astype(str)+transformed_X['SEMESTER_MO'].astype(str),
            format='%Y%m')
        
        # Student's age at the beginning of the first term (in years)
        transformed_X['STDNT_AGE'] = \
            (transformed_X['TERM_START_DAT'] - transformed_X['STDNT_BIRTH_DAT']) / np.timedelta64(1, 'Y')
                
        
        # Student's age as a categorical variable
        transformed_X['STDNT_AGE_BIN'] = pd.cut(transformed_X['STDNT_AGE'], self.age_bins)
        
        transformed_X = transformed_X.drop(columns=['TERM_START_DAT', 'STDNT_BIRTH_DAT', 
                                            'SEMESTER_MO'])
        return transformed_X