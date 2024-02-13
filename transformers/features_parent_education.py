import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureParentEducation(BaseEstimator, TransformerMixin):
    """
    Creates a new feature by grouping values of the field 'PRNT_MAX_ED_LVL_DES'

    """
    
    def __init__(self) -> None:    
        pass
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        transformed_X['PRNT_ED_LVL'] = transformed_X['PRNT_MAX_ED_LVL_DES']
        
        # Group values nan and "Not Indicated"
        transformed_X['PRNT_ED_LVL'] = np.where((transformed_X['PRNT_ED_LVL'].isnull())|\
                                                (transformed_X['PRNT_ED_LVL']==' '),
                                                "Not Indicated",
                                                transformed_X['PRNT_ED_LVL'])
        
        # Group values "Associate's degree" and "Nursing diploma"
        transformed_X['PRNT_ED_LVL'] = np.where((transformed_X['PRNT_ED_LVL']=="Associate's degree")|\
                                                (transformed_X['PRNT_ED_LVL']=="Nursing diploma"),
                                                "Associate or Nursing",
                                                transformed_X['PRNT_ED_LVL'])
        
        # Group values "Elementary School only", "High School diploma", "Less than High School"
        transformed_X['PRNT_ED_LVL'] = np.where((transformed_X['PRNT_ED_LVL']=="Elementary School only")|\
                                                (transformed_X['PRNT_ED_LVL']=="High School diploma")|\
                                                (transformed_X['PRNT_ED_LVL']=="Less than High School"),
                                                "High School and less",
                                                transformed_X['PRNT_ED_LVL'])
        
        # Group values "Doctorate", "Professional Doctorate", "Post Doctorate"
        transformed_X['PRNT_ED_LVL'] = np.where((transformed_X['PRNT_ED_LVL']=="Doctorate")|\
                                                (transformed_X['PRNT_ED_LVL']=="Professional Doctorate")|\
                                                (transformed_X['PRNT_ED_LVL']=="Post Doctorate"),
                                                "Doctorate and Higher",
                                                transformed_X['PRNT_ED_LVL'])
        
        return transformed_X