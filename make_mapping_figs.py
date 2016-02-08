# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:14:14 2016

@author: James McFarland
Generates additional figures for showing loan features grouped by borrower location
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re
import pandas as pd
from LC_helpers import *
from sklearn.cross_validation import KFold

#%%
data_dir = 'static/data/'
fig_dir = 'static/images/'
data_name = 'all_loans_proc'

LD = pd.read_csv(data_dir + data_name, parse_dates=['issue_d'])


def make_group_cumdist_fig(LD,group_by,pcol):
    '''Make a plot showing the cumulative distribution of the within-group avg of a 
    specified response variable. compared to the cum dist expected by chance.
    INPUTS: 
        LD: pandas dataframe
        group_by: column to groupby for computing avgs.
        pcol: name of response variable column.
    OUTPUTS:
        fig: figure handle'''    

    name_legend_map = {'counts': 'Number of loans (thousands)',
				 'ROI': 'Average ROI (%)',
				  'int_rate': 'interest rate (%)',
				  'default_prob': 'default probability',
				  'dti': 'Debt-to-income ratio',
				  'emp_length': 'employment length (months)',
                        'annual_inc': 'annual income ($)'}

    min_group_loans = 100 #only use states with at least this many loans
    good_groups = LD.groupby(group_by).filter(lambda x: x[pcol].count() >= min_group_loans)
    n_groups = len(good_groups[group_by].unique())
    group_stats = good_groups.groupby(group_by)[pcol].agg(['mean','sem'])
    group_stats.sort_values(by='mean',inplace=True)
    ov_avg = good_groups[pcol].mean()
    
    #compute bootstrap estimates of null distribution of group-avgs
    boot_samps = 500 #number of bootstrap samples to use when estimating null dist
    shuff_avgs = np.zeros((boot_samps,n_groups))
    shuff_data = good_groups.copy()
    for cnt in xrange(boot_samps):
        shuff_data[pcol] = np.random.permutation(shuff_data[pcol].values)
        shuff_avgs[cnt,:] = shuff_data.groupby(group_by)[pcol].mean().sort_values()
    
    yax = np.arange(n_groups)
    rmean = np.mean(shuff_avgs,axis=0)
    rsem = np.std(shuff_avgs,axis=0)

    #plot avg and SEM of within-state returns
    fig, ax1 = plt.subplots(1,1,figsize=(6.0,5.0))
    if group_by == 'zip3':
        ax1.errorbar(group_stats['mean'].values,yax)
    else:
        ax1.errorbar(group_stats['mean'],yax,xerr=group_stats['sem'])
        plt.fill_betweenx(yax, rmean-rsem, rmean+rsem,facecolor='r',alpha=0.5,linewidth=0)
        plt.yticks(yax, group_stats.index,fontsize=6)
    
    ax1.plot(rmean,yax,'r')
    plt.legend(['Measured-avgs','Shuffled-avgs'],loc='best')
    if group_by == 'zip3':    
        plt.ylabel('Zip codes',fontsize=16)
    else:    
        plt.ylabel('States',fontsize=16)
    plt.xlabel(name_legend_map[pcol],fontsize=16)   
    plt.xlim(np.min(group_stats['mean']),np.max(group_stats['mean']))
    plt.ylim(0,n_groups)
    ax1.axvline(ov_avg,color='k',ls='dashed')
    plt.tight_layout()

    return fig

   
#%% make cumulative distribution figures for each specified grouping variable and response variable combo
grouping_vars = ['addr_state','zip3']
response_vars = ['ROI',
                 'int_rate',
                 'annual_inc',
                 'default_prob',
                 'dti',
                 'emp_length']

for group_var in grouping_vars:
    for resp_var in response_vars:
        fig = make_group_cumdist_fig(LD, group_var, resp_var)
        fname = '{}_{}_cdist.png'.format(group_var,resp_var)
        fig.savefig(fig_dir + fname, dpi=500, format='png')
        plt.close('all')
            
