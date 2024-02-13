import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_dropout_rate(df, col, plot_name, sort=False, rotate=False, dic={}):
    """
    Plots the relationships between a factor and their corresponding dropout rates.

    Args:
    - df: a dataframe containing both response and independent variables;
    - col: a string representing the column name containing the factor values;
    - plot_name: a string specifying the plot's title;
    - sort: a boolean indicating whether to sort bars by dropout levels;
    - rotate: a boolean indicating whether to rotate x-axis labels;
    - dic: a dictionary mapping factor values to descriptive strings, e.g., {1: 'Asian'}.
    """
    
    df = df.copy()
    
    if dic != {}:
        df[col] = df[col].map(dic)
    
    df[col] = df[col].astype('string') 
    df[col] = np.where(df[col].isnull(), 'nan', df[col])
    
        
    bar = df.groupby(col)[['y_ENROLLED_1_YEAR_LATER']].agg(['mean', 'count'])
    bar.columns = bar.columns.droplevel()
    if sort:
        bar = bar.sort_values(by='mean', ascending=False)
    
    fig, ax = plt.subplots(1, 2, figsize=(10,4))
    bar1 = ax[0].bar(bar.index, bar['mean'])
    bar2 = ax[1].bar(bar.index, bar['count'])
    ax[0].title.set_text('Dropout rates')
    ax[1].title.set_text('Distribution of number of students')
    ax[0].bar_label(bar1, fmt=lambda x: '{:.1f}%'.format(x * 100), fontsize=9)
    ax[1].bar_label(bar2, fmt='{:,.0f}', fontsize=9)


    plt.suptitle(plot_name, fontsize=15, y=1.04)
    fig.subplots_adjust(wspace=0.5)  
    if rotate==True:
        ax[0].tick_params(axis='x', labelrotation=90)
        ax[1].tick_params(axis='x', labelrotation=90)