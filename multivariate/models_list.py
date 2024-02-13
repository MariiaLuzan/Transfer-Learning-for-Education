import pandas as pd
import numpy as np

def create_list_of_models(possible_groups, groups_to_include, factors_df):
    """
    Generates a list of potential models through various combinations of factors.

    Args:
    possible_groups: A list of factor groups that could be included in the model
    groups_to_include: A list of factor groups that must be included
    factors_df - dataframe containing groups of factors and corresponding individual factors

    Returns:
    models_df: A DataFrame containing the models to be considered.
    """
    
    # Number of possible models (different combination of factors)
    num_of_models = 1
    dic_num_comb = {}
    include_nan_fact = {}
    for group in possible_groups:
        if group in groups_to_include:
            dic_num_comb[group] = len(factors_df.loc[factors_df['Group']==group, 'Factor'])
            include_nan_fact[group] = False
        else:
            dic_num_comb[group] = len(factors_df.loc[factors_df['Group']==group, 'Factor']) + 1
            include_nan_fact[group] = True
        
        num_of_models = num_of_models * dic_num_comb[group]     

    models_df = pd.DataFrame(index=range(num_of_models),columns=possible_groups)
    
    # Generate models
    num_of_rows = num_of_models
    for group in possible_groups:
        # Possible features in a group of features
        x = factors_df.loc[factors_df['Group']==group, 'Factor']
        x.index = range(1, len(x)+1)
    
        num_of_rows = int(num_of_rows / dic_num_comb[group])
        i = num_of_models / num_of_rows
        j = int(i / dic_num_comb[group])
    
        ind_start = 0
        ind_end = num_of_rows
        for l in range(1,j+1):
            if include_nan_fact[group]:
                x.loc[len(x)+1]=np.NaN
            for factor_ind in range(dic_num_comb[group]):
                if type(x.iloc[factor_ind])==list:
                    models_df.iloc[ind_start:ind_end, possible_groups.index(group)] = ','.join(x.iloc[factor_ind])
                else:
                    models_df.iloc[ind_start:ind_end, possible_groups.index(group)] = x.iloc[factor_ind]
                
                ind_start += num_of_rows
                ind_end += num_of_rows

    # Add remaining factors that will be always included in the models            
    models_df['REMAINING_FACTORS'] = "No_grades_at_all, Grade_Overall_I_for_1_and_more_courses, \
        Grade_W_for_1_course, Grade_W_for_2_courses,\
        Grade_W_for_3_and_more_courses, Grade_NR_for_1_and_more_courses"     
    
    # Create models as lists of factors in the column 'MODEL'
    models_df['MODEL'] = models_df.values.tolist()
    models_df['MODEL'] = models_df['MODEL'].astype(str).str.replace('\[|\]', '')
    models_df['MODEL'] = models_df['MODEL'].astype(str).str.replace("'", '')
    models_df['MODEL'] = models_df['MODEL'].astype(str).str.replace("nan,", '')
    models_df['MODEL'] = models_df['MODEL'].astype(str).str.replace("nan", '')
    models_df['MODEL'] = models_df['MODEL'].astype(str).str.replace(" ", '')
    models_df['MODEL'] = models_df['MODEL'].str.rstrip(',')
    models_df['MODEL'] = models_df['MODEL'].str.split(",")
    
    return models_df  