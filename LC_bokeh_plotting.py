# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:24:29 2016

@author: James McFarland
"""
import pandas as pd
import seaborn as sns
import datetime as dt
import numpy as np
from bokeh.plotting import figure 
from bokeh.models import LinearAxis, Range1d
from bokeh.embed import components #to generate embedded html


tools = "pan,box_zoom,reset,resize" #set of tools to include in the Bokeh toolbar


def plot_agg_comparison(LD, value, groupby, sm=5, n_quantiles=10):
    '''Make Bokeh time series plot.
    INPUTS:
        LD: pandas dataframe with loan data
        value: name of column to plot as response variable.
        groupby: name of column to use for grouping data.
        sm: number of months to use for EWMA smoothing window (default 5).
        n_quantiles: number of quantiles to use if plotting response variable by quantiles.
    OUTPUTS:
        (script, div) html components of Bokeh plot.'''

    min_counts = 500 #min number of loans for each group (across all time points) for inclusion in plot
    if value == 'counts': #treat response variable 'counts' by setting that to the agg fnx.
        agg = 'count'
        value = 'ROI'
    else:
        agg = 'mean'    
    
    #map of response variables to axis titles.
    name_legend_map = {'counts': 'Number of loans (thousands)',
				 'ROI': 'ROI (%)',
				 'int_rate': 'interest rate (%)',
				 'default_prob': 'default probability',
				 'dti': 'Debt-to-income ratio',
				 'emp_length':'employment length (months)',
                        'annual_inc':'annual income ($)'}
    if groupby == 'loan_status': #specify order and pallete if plotting by loan-status
        group_set = ['Fully Paid', 'Current', 'In Grace Period','Late (16-30 days)', 
                     'Late (31-120 days)', 'Default', 'Charged Off']
        pal = sns.cubehelix_palette(n_colors=len(group_set)).as_hex()
    elif groupby == 'quantiles': #same for if plotting by quantiles
        quantiles = np.linspace(n_quantiles, 100-n_quantiles,n_quantiles).astype(float)/100.
        pal = sns.cubehelix_palette(n_colors=len(quantiles)).as_hex()        
    else: #otherwise, assume categorical grouping vars with no specified plotting order
        group_set = np.sort(LD[groupby].unique())
        pal = sns.color_palette("muted",n_colors=len(group_set)).as_hex()
    
    start_date = np.datetime64(dt.datetime(2009,01,01)) #only plot data for loans issued after this time
    if groupby == 'quantiles':
        time_avgs = LD[LD.issue_d > start_date].groupby(['issue_d'])[value].quantile(quantiles)       
        time_avgs = time_avgs.swaplevel(0,1) # swap quantile and time levels
        group_set = time_avgs.index.levels[0]
    else:
        time_avgs = LD[LD.issue_d > start_date].groupby([groupby,'issue_d'])[value].agg(agg)
        time_counts = LD[LD.issue_d > start_date].groupby([groupby,'issue_d'])[value].count()    
   
    time_avgs.index.names = [groupby,'issue_d']   
    
    ylabel = name_legend_map[value]    
    
    if sm > 0: #apply EWMA smoothing to time-grouped avgs 
        time_avgs = pd.ewma(time_avgs,span=sm)
    
    s1 = figure(x_axis_type="datetime", plot_width=800,
               plot_height=500, tools=tools,
               x_axis_label = 'Issue Date',
               y_axis_label = ylabel)

    for idx, group in enumerate(group_set):        
        if groupby != 'quantiles':        
            if time_counts.ix[group].sum() >= min_counts:
                group_avg = time_avgs.ix[group]
                s1.line(group_avg.index,
                        group_avg,
                        line_color=pal[idx],
                        line_width=4,
                        legend=str(group))
        else:
            group_avg = time_avgs.ix[group]
            s1.line(group_avg.index,
                    group_avg,
                    line_color=pal[idx],
                    line_width=4,
                    legend=str(group))
                            
    s1.legend.label_text_font_size = '13'  
    s1.legend.orientation = "bottom_right"    
    return components(s1)  


def agg_data(df_grouped, col_name):
    '''Helper function for computing groupby with specified response vars 
    (allowing for counts as a response var'''
    if col_name == 'counts':
        return df_grouped['ROI'].count()
    else:
        return df_grouped[col_name].mean()


def plot_2factor_comparison(LD, value1, value2, sm=5):
    '''Plot time series of two different loan features side-by-side, using two y-axes.
    INPUTS:
        LD: pandas dataframe with loan data.
        value1: string specifying first response var
        value2: string specifying second response var.
        sm: optional EWMA smoothing window in months (default 5).
    OUTPUTS:
        (script, div) HTML components of resulting plot.'''

    start_date = np.datetime64(dt.datetime(2008,01,01)) #only plot data after this point
    time_avgs = LD[LD.issue_d > start_date].groupby('issue_d')
    
    y1 = agg_data(time_avgs,value1)
    y2 = agg_data(time_avgs,value2)
    if sm > 0:
        y1 = pd.ewma(y1,span=sm)
        y2 = pd.ewma(y2,span=sm)
        
    pal = sns.color_palette("muted").as_hex()
    
    s1 = figure(x_axis_type="datetime", plot_width=1000,
               plot_height=600, tools=tools)
    s1.extra_y_ranges = {"foo": Range1d(start=y2.min(), end=y2.max())}
    
    s1.add_layout(LinearAxis(y_range_name="foo"), 'right')
    
    s1.line(y1.index, 
            y1,
            line_color=pal[0],  
            line_width=4,
            legend=value1)
            
    s1.set(y_range=Range1d(start=y1.min(), end=y1.max()))
    s1.line(y2.index,
            y2,
            line_color=pal[2],
            line_width=4,
            legend=value2, y_range_name="foo")
            
    return components(s1)                  
