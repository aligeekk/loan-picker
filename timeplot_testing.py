# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 13:48:05 2016

@author: james
"""
import pandas as pd
import seaborn as sns
from bokeh.io import output_file, show
from bokeh.plotting import figure
from bokeh.models import LinearAxis, Range1d
import datetime as dt
import numpy as np
data_dir = 'static/data/'
fig_dir = 'static/images/'

data_name = 'all_loans_proc'

LD = pd.read_csv(data_dir + data_name)
LD['issue_d'] = pd.to_datetime(LD['issue_d'],format = '%Y-%m-%d',unit='D')

#%%
output_file('/Users/james/Desktop/test.html')

def agg_data(df_grouped, col_name):
    if col_name == 'counts':
        return df_grouped['ROI'].count()
    else:
        return df_grouped[col_name].mean()

def plot_2factor_comparison(LD, value1, value2, sm=5):
    start_date = np.datetime64(dt.datetime(2008,01,01))
    time_avgs = LD[LD.issue_d > start_date].groupby('issue_d')
    
    y1 = agg_data(time_avgs,value1)
    y2 = agg_data(time_avgs,value2)
    if sm > 0:
        y1 = pd.ewma(y1,span=sm)
        y2 = pd.ewma(y2,span=sm)
        
    pal = sns.color_palette("muted").as_hex()
    
    s1 = figure(x_axis_type="datetime", plot_width=1000,
               plot_height=600)
    s1.extra_y_ranges = {"foo": Range1d(start=y2.min(), end=y2.max())}
    
    s1.add_layout(LinearAxis(y_range_name="foo"), 'right')
    
    s1.line(y1.index,
                      y1,
                      line_color=pal[0],
                      line_width=2,
                      legend='mean')
    s1.set(y_range=Range1d(start=y1.min(), end=y1.max()))
    s1.line(y2.index,
                      y2,
                      line_color=pal[2],
                      line_width=2,
                      legend='counts', y_range_name="foo")
                      
    show(s1)


def plot_agg_comparison(LD, value, groupby,agg='mean', sm=5):
    group_set = np.sort(LD[groupby].unique())
    pal = sns.color_palette("muted",n_colors=len(group_set)).as_hex()
    
    start_date = np.datetime64(dt.datetime(2009,01,01))
    time_avgs = LD[LD.issue_d > start_date].groupby([groupby,'issue_d'])[value].agg(agg)
        
    if sm > 0:
        time_avgs = pd.ewma(time_avgs,span=sm)
    
    s1 = figure(x_axis_type="datetime", plot_width=1000,
               plot_height=600)
    for idx, group in enumerate(group_set):
        group_avg = time_avgs.ix[group]
        s1.line(group_avg.index,
                      group_avg,
                      line_color=pal[idx],
                      line_width=2,
                      legend=str(group))
                      
    show(s1)
       

#%%
groupby = 'purpose'
value = 'dti'
agg = 'mean'
sm = 10
plot_agg_comparison(LD,value,groupby)
#%%
       
groupby = None
value1 = 'int_rate'
value2 = 'ROI'

sm = 5

plot_2factor_comparison(LD,value1,value2)




#%%
