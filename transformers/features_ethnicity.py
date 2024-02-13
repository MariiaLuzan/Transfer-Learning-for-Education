import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEthnicGroupDummy(BaseEstimator, TransformerMixin):
    """
    Creates a new field called 'STDNT_ETHNC_NAN' for correct use of dummy variables:
    'STDNT_ETHNC_NAN' equals 1 when there's an absence of student ethnicity data.

    """
    
    def __init__(self) -> None:    
        pass
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        # Group values 'Native Amr' (5) and 'Hawaiian' (7)
        transformed_X['STDNT_ETHNC_GRP_CD'] = np.where(transformed_X['STDNT_ETHNC_GRP_CD']==7, 
                                                       5,
                                                       transformed_X['STDNT_ETHNC_GRP_CD'])
        
        
        # If all the fields indicating ethnic group equal 0, it implies that we lack information 
        # about the ethnicity of the student
        transformed_X['STDNT_ETHNC_NAN'] = np.where(transformed_X['STDNT_ASIAN_IND']+\
                                                    transformed_X['STDNT_BLACK_IND']+\
                                                    transformed_X['STDNT_HWIAN_IND']+\
                                                    transformed_X['STDNT_HSPNC_IND']+\
                                                    transformed_X['STDNT_NTV_AMRCN_IND']+\
                                                    transformed_X['STDNT_WHITE_IND']+\
                                                    transformed_X['STDNT_MULTI_ETHNC_IND']==0,
                                                    1,
                                                    0)
        
        transformed_X['STDNT_NTV_AMRCN_HWIAN_IND'] = np.where((transformed_X['STDNT_NTV_AMRCN_IND']==1)|\
                                                              (transformed_X['STDNT_HWIAN_IND']==1),
                                                              1,
                                                              0)
                
        return transformed_X