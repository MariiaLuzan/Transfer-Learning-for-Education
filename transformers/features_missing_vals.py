import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


from sklearn.pipeline import Pipeline
import category_encoders as ce

class ImputeMissing(BaseEstimator, TransformerMixin):
    """
    Creates new features with imputed missing values using mean strategy and interpolation

    """
    
    def __init__(self, cols:list, dic:dict) -> None:    
        # List of columns for imputation
        self.cols = cols
        # Dic of columns for imputation through interpolation
        self.dic = dic
    
    def fit(self, X:pd.DataFrame, y:pd.Series=None) -> 'CustomTransformer':
        # Calculate average values that will be used for imputation
        self.mean_values = X[self.cols].mean(axis=0)
        
        # Calculate interpolated values that will be used for imputation
        self.impute_vals = {}
        for col in self.dic:
            dropout_rates = X.groupby(self.dic[col])[[col, 'y_ENROLLED_1_YEAR_LATER']].mean()
            
            # Calculate dropout rates for missing data
            y_for_missing = X.loc[X[col].isnull(),'y_ENROLLED_1_YEAR_LATER'].mean()
            if y_for_missing >= dropout_rates['y_ENROLLED_1_YEAR_LATER'].max():
                self.impute_vals[col] = \
                    dropout_rates.loc[dropout_rates['y_ENROLLED_1_YEAR_LATER']==dropout_rates['y_ENROLLED_1_YEAR_LATER'].max(),
                                      col].values[0] 
            elif y_for_missing <= dropout_rates['y_ENROLLED_1_YEAR_LATER'].min():
                self.impute_vals[col] = \
                    dropout_rates.loc[dropout_rates['y_ENROLLED_1_YEAR_LATER']==dropout_rates['y_ENROLLED_1_YEAR_LATER'].min(),
                                      col].values[0]
            else: 
                dropout_rates['y_for_missing'] = y_for_missing
                dropout_rates[col+'_shift'] = dropout_rates[col].shift()
                dropout_rates['y_ENROLLED_1_YEAR_LATER_shift'] = dropout_rates['y_ENROLLED_1_YEAR_LATER'].shift()
                dropout_rates['y_diff'] = dropout_rates['y_ENROLLED_1_YEAR_LATER'] - dropout_rates['y_for_missing']
                dropout_rates['y_diff_shift'] = dropout_rates['y_diff'].shift()
                dropout_rates['y_estimate'] = \
                    (dropout_rates[col] * dropout_rates['y_diff_shift'] -\
                     dropout_rates[col+'_shift'] * dropout_rates['y_diff'])/ \
                    (dropout_rates['y_diff_shift'] - dropout_rates['y_diff'])

                self.impute_vals[col] = dropout_rates.loc[(dropout_rates['y_diff']<=0)&(dropout_rates['y_diff_shift']>=0), 'y_estimate'].iloc[0]
            
        return self

    def transform(self, X:pd.DataFrame)  -> pd.DataFrame:
       
        transformed_X = X.copy()
        
        for col in self.cols:
            transformed_X[col+'_IsMissing'] = np.where(transformed_X[col].isnull(), 1, 0)
            transformed_X[col+'_Imp'] = np.where(transformed_X[col].isnull(), 
                                                 self.mean_values[col],
                                                 transformed_X[col])
        
        for col in self.dic:
            transformed_X[col+'_Imp_Interp'] = np.where(transformed_X[col].isnull(), 
                                                        self.impute_vals[col],
                                                        transformed_X[col])
            
        return transformed_X
    
    

# Factors to impute missing values
impute_missing_cols = ['HS_GPA', 'SAT_ACT_TOTAL_SCR', 'ACT_SAT_TOTAL_SCR',
                       'ACT_SAT_MATH_PCTL', 'SAT_ACT_MATH_PCTL']

# Factors to impute missing values through interpolation
dic = {'HS_GPA': 'HS_GPA_BIN',
       'SAT_ACT_TOTAL_SCR': 'SAT_ACT_TOTAL_BIN'}

# Factors to impute WoE
impute_WoE_cols = ['STDNT_AGE_BIN', 'HS_GPA_BIN', 'CURR_GPA_BIN', 'STDNT_ETHNC_GRP_CD',
                   'GROSS_FAM_INC', 'SAT_ACT_TOTAL_BIN', 'PRNT_ED_LVL', 'ACAD_LOAD']

encoder_WoE = ce.WOEEncoder(cols=impute_WoE_cols)

imputation_pipeline = Pipeline([
    ('missing_values', ImputeMissing(impute_missing_cols, dic)),
    ('encoder_WoE', encoder_WoE),
  
])