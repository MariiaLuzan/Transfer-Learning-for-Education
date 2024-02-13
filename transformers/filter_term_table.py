import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FilterTermRows(BaseEstimator, TransformerMixin):
    """
    Filters rows from the STDNT_TERM_INFO dataset by performing the following steps:
    1. Retains only the first term entry for each student
    2. Filters out rows with a year that is less than the specified parameter year
    3. Selects only undergraduate students and freshmen
    4. Selects only the rows for which the response variable 'y' has been computed.
    
    Params:
    min_year - integer, the year for filtering (rows with year equal or greater will be taken in the sample)
        
    Uses X - a dataframe with data from STDNT_TERM_INFO (the response variable must be already computed)
    """
    
    def __init__(self, min_year:int) -> None:        
        self.min_year = min_year
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame) -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        # Take only the first term entry
        transformed_X = transformed_X.groupby('STDNT_ID').nth(0)
        
        # Filter out rows with a year that is less than the specified parameter year
        transformed_X = transformed_X[transformed_X['YEAR'] >= self.min_year]
        
        # Select only undergraduate students and freshmen
        transformed_X = \
            transformed_X[(transformed_X['CRER_LVL_CD'] == 'U')&(transformed_X['ENTRY_TYP_SHORT_DES']=='Freshman')]
        
        # Select only the rows for which the response variable 'y' has been computed
        transformed_X = transformed_X[~transformed_X['y_ENROLLED_1_YEAR_LATER'].isnull()]
                
        return transformed_X