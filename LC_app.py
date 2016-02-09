# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 09:59:19 2016

@author: James McFarland
Runs web app for Data Incubator capstone project analyzing Lending Club p2p data
"""

from flask import Flask, render_template, request, redirect, url_for, Markup
from wtforms import Form, RadioField, IntegerField, SelectMultipleField, IntegerField, BooleanField
from wtforms.validators import DataRequired, NumberRange
import matplotlib
# Force matplotlib to not use any Xwindows backend. Needed for generating figs on DO box
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import re
import os
from collections import defaultdict, namedtuple

import pandas as pd
from bs4 import BeautifulSoup
import LC_helpers as LCH
import LC_loading as LCL
import LC_bokeh_plotting as LC_bok
import LC_latest_predictions as LCP
from LC_forms import *
import dill
import json
import time

auth_keys = json.load(open("auth_keys.txt",'r'))

app_title = 'Loan Picker' 
refresh_loan_time = 3600 * 4 #refresh current loans every this many seconds

data_dir = 'static/data/'
fig_dir = 'static/images/'
data_name = 'all_loans_proc_short' #compact version of the loan data that can live on Heroku
map_name = 'USA_Counties_raw.svg' #base svg map of US with counties

# dict of axis labels corresponding to different features we might be plotting
name_legend_map = {'counts': 'Number of loans (thousands)',
			            'ROI': 'ROI (%)',
			            'int_rate': 'interest rate (%)',
			            'default_prob': 'default probability',
			            'dti': 'Debt-to-income ratio',
			            'emp_length':'employment length (months)',
                  'annual_inc':'annual income ($)'}

# dict for making a coarser grouping of loan purposes for plotting purposes
purpose_map = {'debt_consolidation':'debt',
              'credit_card':'debt',
              'home_improvement':'home improvement',
              'house':'home improvement',
              'major_purchase':'major purchase',
              'car':'major purchase',
              'wedding':'major purchase',
              'vacation':'major purchase',
              'moving':'major purchase',
              'small_business':'small business',
              'medical':'medical',
              'other':'other',
              'renewable_energy':'other',
              'educational':'other'}

app = Flask(__name__)
app.secret_key = auth_keys['app_key']

#%%
LD = pd.read_csv(data_dir + data_name, parse_dates=['issue_d'])
# LD['issue_d'] = pd.to_datetime(LD['issue_d'],format = '%Y-%m-%d',unit='D')
print('loaded {0} loans'.format(len(LD)))

#load lookup tables for converting zips and states to county FIPS codes for plotting
fips_to_zip = dill.load(open(data_dir + 'fips_to_zip.p', "rb" ) )
state_fips_dict = dill.load(open(data_dir + 'state_fips_dict.p',"rb"))

#get lat/long coordinates for each 3-digit zip
zip3_loc_path = os.path.join(data_dir,'zip3_loc_data.p')
with open(zip3_loc_path,'rb') as in_strm:
    zip3_loc_data = dill.load(in_strm)       

# precompute additional columns for convenience when plotting
LD['short_purpose'] = LD['purpose'].map(purpose_map)
LD['issue_year'] = LD['issue_d'].dt.year

# load base map and get state and county paths
app.base_map = LCL.load_base_map(fig_dir + map_name)
(app.county_paths,app.state_paths) = LCL.get_map_paths(app.base_map,fips_to_zip)

predictor = namedtuple('predictor', ['col_name', 'full_name', 'norm_type'])
model_data = LCP.load_pickled_models()
sim_lookup = LCP.get_validation_data()

#%%
use_grades = ['A','B','C','D','E','F']
load_time = time.time()
print('Grabbing loan data at {}'.format(load_time))
predictions = LCP.get_LC_loans(auth_keys['LC_auth_key'], model_data,
                               zip3_loc_data, use_grades)
                               
#%%
@app.route('/') #redirect to index page
def main():
    return redirect('/index')


# page where user selects desired stock features to plot
@app.route('/index',methods=['GET'])
def index():
    return render_template('home.html', app_title=app_title) #if request method was GET


# form page for making plots by borrower location
@app.route('/loan_mapping', methods=['GET','POST'])
def loan_mapping(map_rendered=False):
    mform = map_form(request.form)    
    if request.method == 'POST' and mform.validate():  #if user is posting, get form data and store it           
         map_rendered = True       
         # compute desired group-by/agg operation  
         data = LCH.compute_group_avgs(LD, mform.data['col_name'], 
                                       mform.data['grouping_var'], 
                                       mform.data['agg_fun'],
                                       state_fips_dict=state_fips_dict, 
                                       min_counts=50)
         data.name = mform.data['col_name']
         if mform.data['agg_fun'] == 'count':
             data.name = 'counts'
         # paint base map by county
         pal = LCH.paint_map(data, app.base_map, app.county_paths, fips_to_zip, 
                             color='cube', agg_fun=mform.data['agg_fun'])        
         # save colorbar for map as a png
         plt.savefig(fig_dir + 'map_cbar.png', dpi=500, format='png')
         plt.close()
         
    return render_template('loan_mapping.html', map_form=mform, svg=Markup(str(app.base_map)),
                rnum=np.random.randint(0,100000), map_rendered=map_rendered) 


# reset map and redirect to loan form page
@app.route('/reset_map',methods=['GET'])
def reset_map():
    print('resetting map')
    app.base_map = LCL.load_base_map(fig_dir + map_name)
    (app.county_paths,app.state_paths) = LCL.get_map_paths(app.base_map,fips_to_zip)
    return redirect('/loan_mapping') 


# page with analysis details
@app.route('/details',methods=['GET'])
def details():
    return render_template('details.html', rnum=np.random.randint(0,100000)) 


# page with modeling results
@app.route('/models',methods=['GET'])
def models():
    return render_template('models.html',rnum=np.random.randint(0,100000)) 


# page with form for generating time-based plots
@app.route('/time_series_form',methods=['GET','POST'])
def time_series_form():
    mform =  ts_form(request.form)    
    if request.method == 'POST' and mform.validate():  #if user is posting, get form data and store it           
        app.ts_form = mform
        return redirect('/time_series') #otherwise, go to the graph page
    
    return render_template('time_series_form.html', ts_form=mform) 


# results page for time-based plots
@app.route('/time_series',methods=['GET','POST'])
def time_series():
    sform =  smoothing_form(request.form)    
    col_name = app.ts_form.data['col_name']
    if request.method == 'GET':
        script, div = LC_bok.plot_agg_comparison(LD, col_name, 
                                                 app.ts_form.data['grouping_var'],
                                                 sm=app.ts_form.data['smooth_span'],
                                                 n_quantiles=app.ts_form.data['num_quantiles'])
    elif request.method == 'POST':
        script, div = LC_bok.plot_agg_comparison(LD, col_name, 
                                                 app.ts_form.data['grouping_var'],
                                                 sm=sform.data['smooth_span'],
                                                 n_quantiles=app.ts_form.data['num_quantiles'])
    return render_template('time_series.html', script=script, div=div, 
                           ts_form=app.ts_form, sform=sform, leg_map=name_legend_map) #if request method was GET


# page with form for generating time-based plots
@app.route('/current_loans',methods=['GET','POST'])
def current_loans():
    global predictions, load_time
    
    cur_time = time.time()
    if (cur_time - load_time) > refresh_loan_time:       
        load_time = cur_time        
        print('Grabbing loan data at {}'.format(load_time))
        predictions = LCP.get_LC_loans(auth_keys['LC_auth_key'], model_data,
                                       zip3_loc_data, use_grades)
    
    mform = cl_form(request.form)          
    if request.method == 'POST':
        app.cl_form = mform
        return redirect('/current_loans_results') 
       
    else:
        fig = LCP.make_dp_ret_figure(predictions, 0, predictions)
        plt.savefig(fig_dir + 'cl_dp_ret.png', dpi=500, format='png')
        plt.close()
            
        return render_template('current_loans.html', cl_form=mform, 
                               rnum=np.random.randint(0,100000),
                               tot_loans=len(predictions)) 


@app.route('/current_loans_results',methods=['GET','POST'])
def current_loans_results():
    mform = cl_form(request.form) 
    if request.method == 'GET':
        mform = app.cl_form

    bool_vals = {'use_A':'A','use_B':'B','use_C':'C','use_D':'D',
                 'use_E':'E','use_F':'F'}
    constraints = {}
    constraints['use_grades'] = [bool_vals[key] for key in bool_vals.keys() \
                                if mform.data[key]]
    constraints['max_dp'] = mform.data['max_dp']
    
    allowed_loans = predictions[(predictions.dp <= constraints['max_dp']) & \
                                (predictions.grades.isin(constraints['use_grades']))]
    pick_K = min([mform.data['port_size'], len(allowed_loans)])

    loan_ids_string = ', '.join(allowed_loans[:pick_K]['ids'].values.astype(str))
    fig = LCP.make_dp_ret_figure(predictions, pick_K, allowed_loans)
    plt.savefig(fig_dir + 'cl_dp_ret.png', dpi=500, format='png')
    plt.close()
    
    if pick_K > 0:
        fig = LCP.make_return_dist_fig(sim_lookup, allowed_loans, pick_K)
        plt.savefig(fig_dir + 'cl_ret_dist.png', dpi=500, format='png')
        plt.close()
        
    return render_template('current_loan_results.html', cl_form=mform, 
                           pick_K=pick_K, rnum=np.random.randint(0,100000),
                           loan_ids=loan_ids_string) 


if __name__ == '__main__':
#    app.run()
    app.run(host='0.0.0.0')
