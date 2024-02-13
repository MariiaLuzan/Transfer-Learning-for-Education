import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureFamilyIncome(BaseEstimator, TransformerMixin):
    """
    Creates a new feature by grouping values of the field 'EST_GROSS_FAM_INC_DES'

    """
    
    def __init__(self) -> None:    
        pass
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        transformed_X['GROSS_FAM_INC'] = transformed_X['EST_GROSS_FAM_INC_DES']
        
        # Group values nan and ' '
        transformed_X['GROSS_FAM_INC'] = np.where(transformed_X['GROSS_FAM_INC'].isnull(),
                                                  ' ',
                                                  transformed_X['GROSS_FAM_INC'])
        
        # Group values $25,000 - $49,999, '$50,000 - $74,999', '$75,000 - $99,999'
        transformed_X['GROSS_FAM_INC'] = np.where((transformed_X['GROSS_FAM_INC']=='$25,000 - $49,999')|\
                                                  (transformed_X['GROSS_FAM_INC']=='$50,000 - $74,999')|\
                                                  (transformed_X['GROSS_FAM_INC']=='$75,000 - $99,999'),
                                                  '$25,000 - $99,999',
                                                  transformed_X['GROSS_FAM_INC'])
        
         # Group values '$150,000 - $199,999' and 'More than $200,000'
        transformed_X['GROSS_FAM_INC'] = np.where((transformed_X['GROSS_FAM_INC']=='$150,000 - $199,999')|\
                                                  (transformed_X['GROSS_FAM_INC']=='More than $200,000'),
                                                  'More than $150,000',
                                                  transformed_X['GROSS_FAM_INC'])
        
                
        return transformed_X