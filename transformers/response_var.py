import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class EnrolledOneYearLater(BaseEstimator, TransformerMixin):
    """
    Computes the response variable "y", indicating whether a student has enrolled one year later:
        1: Indicates non-enrollment
        0: Indicates enrollment
    
    This algorithm will be applied exclusively to undegrad first-year students, that's why it does not 
    take into account their graduation dates.
    
    Uses X - a dataframe with data from STDNT_TERM_INFO
    """
    
    def __init__(self) -> None:        
        pass
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        # Compute terms where the determination of the enrolment flag remains relevant
        # If the time span between a given term and the last term in the dataset is under a year, 
        # the calculation of the enrolment flag is not needed
        
        # The code of the last term in the dataset
        self.last_term = X['TERM_CD'].max()
        
        # The code for the last term where the computation of the enrolment flag remains relevant
        # (at least one year prior to the last term)
        self.enrol_flag_last_term = self.last_term - 50 
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        transformed_X = transformed_X.sort_values(['STDNT_ID', 'TERM_CD'])
        transformed_X['SEMESTER'] = transformed_X['TERM_SHORT_DES'].str.split(' ').str[0]
        transformed_X['YEAR'] = transformed_X['TERM_SHORT_DES'].str.split(' ').str[1].astype(int)
        
        # Calculate the difference between the first term and the subsequent terms for every student
        transformed_X['DIFF_WITH_THE_1ST_TERM'] = transformed_X['TERM_CD'] - \
            transformed_X.groupby('STDNT_ID')['TERM_CD'].transform('first')

        # If this difference is within the range of 50 to 100, it means that a student studied in the subsequent year 
        # following the first term
        transformed_X['1Y_LATER_FROM_1ST_TERM'] = np.where((transformed_X['DIFF_WITH_THE_1ST_TERM']>=50)&
                                                           (transformed_X['DIFF_WITH_THE_1ST_TERM']<100),
                                                           1,0)
        transformed_X['y_ENROLLED_1_YEAR_LATER'] = \
            transformed_X.groupby('STDNT_ID')['1Y_LATER_FROM_1ST_TERM'].transform('sum')
        transformed_X['y_ENROLLED_1_YEAR_LATER'] = np.where(transformed_X['y_ENROLLED_1_YEAR_LATER']>=1, 0, 1)
        
        transformed_X.drop(columns=['DIFF_WITH_THE_1ST_TERM', '1Y_LATER_FROM_1ST_TERM'], inplace=True)
        
        # If the difference with the last term in the data set is less than one year, the response variable should be NAN
        transformed_X['y_ENROLLED_1_YEAR_LATER'] = np.where(transformed_X['TERM_CD'] <= self.enrol_flag_last_term,
                                                            transformed_X['y_ENROLLED_1_YEAR_LATER'],
                                                            np.nan)
                
        return transformed_X