import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_proportion(sample_df, source_school, target_school, col, title):
    plt.plot(sample_df[sample_df['PRMRY_CRER_CD']==source_school].groupby('YEAR')[[col]].mean(),
             label=source_school)
    plt.plot(sample_df[sample_df['PRMRY_CRER_CD']==target_school].groupby('YEAR')[[col]].mean(),
             label=target_school)
    plt.title(title)
    plt.legend();

    
def plot_two_distr(sample_df, source_school, target_school, col, title):
    stat_source = sample_df.loc[sample_df['PRMRY_CRER_CD']==source_school, col].value_counts() / \
        len(sample_df[sample_df['PRMRY_CRER_CD']==source_school])
    plt.bar(stat_source.index, stat_source, label=source_school)

    stat_target = sample_df.loc[sample_df['PRMRY_CRER_CD']==target_school, col].value_counts() / \
        len(sample_df[sample_df['PRMRY_CRER_CD']==target_school])
    plt.bar(stat_target.index, stat_target, label=target_school, alpha=0.5)

    plt.xticks(rotation=45, ha='right')
    plt.title(title)
    plt.legend();

    
def plot_two_bars(sample_df, source_school, target_school, col, title):
    stat_source = sample_df.loc[sample_df['PRMRY_CRER_CD']==source_school, col].value_counts() / \
        len(sample_df[sample_df['PRMRY_CRER_CD']==source_school])
    x_axis = np.arange(len(stat_source.index))
    plt.bar(x_axis-0.2, stat_source, width=0.4, label='Source data - '+source_school)

    stat_target = sample_df.loc[sample_df['PRMRY_CRER_CD']==target_school, col].value_counts() / \
        len(sample_df[sample_df['PRMRY_CRER_CD']==target_school])
    plt.bar(x_axis+0.2, stat_target, width=0.4, label='Target data - '+target_school)
    
    plt.xticks(x_axis, stat_source.index)

    plt.xticks(rotation=45, ha='right')
    plt.title(title, fontsize=14)
    plt.legend();    
    

def plot_distribution(sample_df, source_school, target_school, col, n_bins, title):
    
    fig, ax = plt.subplots(2, 1, figsize=(7,5), sharex=True)
    sample_df.loc[sample_df['PRMRY_CRER_CD']==source_school, [col]].hist(ax=ax[0], bins=n_bins, histtype="bar")
    sample_df.loc[sample_df['PRMRY_CRER_CD']==target_school, [col]].hist(ax=ax[1], bins=n_bins, histtype="bar")

    ax[0].grid(False)
    ax[1].grid(False)

    ax[0].title.set_text('Source data - '+source_school)
    ax[1].title.set_text('Target data - '+target_school)

    plt.suptitle(title, fontsize=14);    
    
    
   
