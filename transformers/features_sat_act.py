import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureSatAct(BaseEstimator, TransformerMixin):
    """
    Creates SAT and ACT features (float and categorical);
    SAT fields often contain missing values, and these gaps are filled 
    with the equivalent ACT scores when available.
    """
    
    # SAT - ACT converter
    # Source: https://www.princetonreview.com/college-advice/act-to-sat-conversion
    __SAT_ACT_convert = pd.read_csv('SAT_ACT_converter.csv', sep=';')
    
    def __init__(self, SAT_bins) -> None:    
        self.SAT_bins = SAT_bins
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        return self
    
    def SAT_features(self, X:pd.DataFrame)  -> pd.DataFrame:
        """
        The SAT changed its structure in March 2016, that's why
        scores before and after 2016 are in different columns.
        The function adds columns with SAT scores that integrate data 
        from these two columns (before and after March 2016).
    
        """
        
        transformed_X = X.copy()
    
        # Total SAT
        # MAX_SATI_TOTAL_CALC_SCR - before 2016 (Range: 400 - 1,600),
        # MAX_SATI_TOTAL_MSS_ERWS_SCR - after 2016 (Range: 400 - 1600)
        transformed_X['MAX_SAT_TOTAL_SCR'] = np.where(transformed_X['MAX_SATI_TOTAL_CALC_SCR'].isnull(),
                                                      transformed_X['MAX_SATI_TOTAL_MSS_ERWS_SCR'],
                                                      transformed_X['MAX_SATI_TOTAL_CALC_SCR'])
        transformed_X['MAX_SAT_TOTAL_PCTL'] = transformed_X['MAX_SATI_TOTAL_MSS_ERWS_PCTL']
    
        # Mathematics
        # MAX_SATI_MATH_SCR - before 2016 (Range: 200 - 800)
        # MAX_SATI_MSS_SCR - after 2016 (Range: 200- 800)
        transformed_X['MAX_SAT_MATH_SCR'] = np.where(transformed_X['MAX_SATI_MATH_SCR'].isnull(),
                                                     transformed_X['MAX_SATI_MSS_SCR'],
                                                     transformed_X['MAX_SATI_MATH_SCR'])
        # MAX_SATI_MATH_PCTL - before 2016
        # MAX_SATI_MSS_PCTL - after 2016
        transformed_X['MAX_SAT_MATH_PCTL'] = np.where(transformed_X['MAX_SATI_MATH_PCTL'].isnull(),
                                                      transformed_X['MAX_SATI_MSS_PCTL'],
                                                      transformed_X['MAX_SATI_MATH_PCTL'])
    
        # Clean missing percentile values
        # If percentile == 0, then "most likely an indication that the percentile was not available or not recorded." 
        # (from the data dictionary)
        transformed_X['MAX_SAT_TOTAL_PCTL'] = np.where(transformed_X['MAX_SAT_TOTAL_PCTL']==0,
                                                       np.NaN,
                                                       transformed_X['MAX_SAT_TOTAL_PCTL'])
    
        transformed_X['MAX_SAT_MATH_PCTL'] = np.where(transformed_X['MAX_SAT_MATH_PCTL']==0,
                                                      np.NaN,
                                                      transformed_X['MAX_SAT_MATH_PCTL'])
    
    
        return transformed_X

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        # Add SAT columns with integrated data (before and after March 2016)
        transformed_X = self.SAT_features(X)
        
        # Clean missing percentiles values for ACT
        # If percentile == 0, then "most likely an indication that the percentile was not available or not recorded." 
        # (from the data dictionary)
        transformed_X['MAX_ACT_COMP_PCTL'] = np.where(transformed_X['MAX_ACT_COMP_PCTL']==0,
                                                      np.NaN,
                                                      transformed_X['MAX_ACT_COMP_PCTL'])
    
        transformed_X['MAX_ACT_MATH_PCTL'] = np.where(transformed_X['MAX_ACT_MATH_PCTL']==0,
                                                      np.NaN,
                                                      transformed_X['MAX_ACT_MATH_PCTL'])
    
        # Add information for conversion
        ACT_SAT_convert = self.__SAT_ACT_convert.groupby('ACT').mean()
        ACT_SAT_convert.reset_index(inplace=True)
        transformed_X = transformed_X.merge(ACT_SAT_convert, how='left', left_on='MAX_ACT_COMP_SCR', right_on='ACT')
        transformed_X = transformed_X.merge(self.__SAT_ACT_convert, how='left', left_on='MAX_SAT_TOTAL_SCR', right_on='SAT')
    
        # Convert to ACT equivalent (Range: 01- 36)
        transformed_X['ACT_SAT_TOTAL_SCR'] = np.where(transformed_X['MAX_ACT_COMP_SCR'].isnull(),
                                                      transformed_X['ACT_y'],
                                                      transformed_X['MAX_ACT_COMP_SCR'])
    
        transformed_X['ACT_SAT_TOTAL_PCTL'] = np.where(transformed_X['MAX_ACT_COMP_PCTL'].isnull(),
                                                       transformed_X['MAX_SAT_TOTAL_PCTL'],
                                                       transformed_X['MAX_ACT_COMP_PCTL'])
    
        # Convert to SAT equivalent (Range: 400 - 1600)
        transformed_X['SAT_ACT_TOTAL_SCR'] = np.where(transformed_X['MAX_SAT_TOTAL_SCR'].isnull(),
                                                      transformed_X['SAT_x'],
                                                      transformed_X['MAX_SAT_TOTAL_SCR'])
    
        transformed_X['SAT_ACT_TOTAL_PCTL'] = np.where(transformed_X['MAX_SAT_TOTAL_PCTL'].isnull(),
                                                       transformed_X['MAX_ACT_COMP_PCTL'],
                                                       transformed_X['MAX_SAT_TOTAL_PCTL'])
    
        # Math PCTL
        # Convert to ACT equivalent
        transformed_X['ACT_SAT_MATH_PCTL'] = np.where(transformed_X['MAX_ACT_MATH_PCTL'].isnull(),
                                                      transformed_X['MAX_SAT_MATH_PCTL'],
                                                      transformed_X['MAX_ACT_MATH_PCTL'])
        # Convert to SAT equivalent
        transformed_X['SAT_ACT_MATH_PCTL'] = np.where(transformed_X['MAX_SAT_MATH_PCTL'].isnull(),
                                                      transformed_X['MAX_ACT_MATH_PCTL'],
                                                      transformed_X['MAX_SAT_MATH_PCTL'])
    
    
        # Binning SAT
        transformed_X['SAT_ACT_TOTAL_BIN'] = pd.cut(transformed_X['SAT_ACT_TOTAL_SCR'], self.SAT_bins)
        transformed_X['SAT_TOTAL_BIN'] = pd.cut(transformed_X['MAX_SAT_TOTAL_SCR'], self.SAT_bins)
    
    
        # Columns to drop
        cols = ['MAX_SATI_TOTAL_CALC_SCR', 'MAX_SATI_TOTAL_MSS_ERWS_SCR',
                'MAX_SATI_TOTAL_MSS_ERWS_PCTL', 'MAX_SATI_MATH_SCR', 'MAX_SATI_MSS_SCR',
                'MAX_SATI_MATH_PCTL', 'MAX_SATI_MSS_PCTL', 'MAX_ACT_COMP_SCR',
                'MAX_ACT_COMP_PCTL', 'MAX_ACT_MATH_PCTL', 'MAX_SAT_TOTAL_SCR',
                'MAX_SAT_TOTAL_PCTL', 'MAX_SAT_MATH_SCR', 'MAX_SAT_MATH_PCTL', 
                'ACT_x', 'SAT_x', 'SAT_y', 'ACT_y',]
    
        transformed_X = transformed_X.drop(columns=cols)
                
        return transformed_X