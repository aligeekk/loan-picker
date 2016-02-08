# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 09:59:19 2016

@author: James McFarland
Runs web app for Data Incubator capstone project analyzing Lending Club p2p data
"""

from flask import Flask, render_template, request, redirect, url_for, Markup
from wtforms import Form, RadioField, IntegerField, SelectMultipleField, IntegerField
from wtforms.validators import DataRequired, NumberRange
import matplotlib
# Force matplotlib to not use any Xwindows backend. Needed for generating figs on DO box
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import re
import pandas as pd
from bs4 import BeautifulSoup
import LC_helpers as LCH
import LC_loading as LCL
import LC_bokeh_plotting as LC_bok
import dill

app_title = 'Loan Picker' 

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
app.secret_key = '\xb6\xbcr\xc4\xb5\x9cY\x03\xcdI\x15oR:\xdbJD\xb1c\x00+\x1c\x926'

class map_form(Form):
    '''Form for generating plots of different loan features based on the borrowers 
    geographic location (3-digit zip or state)'''

    grouping_vars = [('addr_state', 'by state'), 
                     ('zip3', 'by 3-digit zip')]
    group_dict = {key: val for key, val in grouping_vars}

    # tuple containing different agg functions (internal_name, human_readable)
    agg_funs = [('count', 'count'), 
                ('mean', 'mean'),
                ('std', 'std dev'),
                ('median', 'median')]               
    agg_dict = {key: val for key, val in agg_funs}

    # variable to plot. tuple as (internal_name, human_readable)
    response_vars = [('ROI', 'returns'), 
                     ('int_rate', 'interest rate'),
                     ('annual_inc', 'annual income'),
                     ('default_prob', 'default probability'),
                     ('dti', 'debt-to-income ratio'),
                     ('emp_length', 'employment length')]
    var_dict = {key: val for key,val in response_vars}
  
    grouping_var = RadioField('grouping', choices=grouping_vars,
                              validators=[DataRequired()],default='zip3')
    agg_fun = RadioField('agg', choices=agg_funs,
                         validators=[DataRequired()],default='mean')
    col_name = RadioField('response', choices=response_vars,
                          validators=[DataRequired()],
                          default='ROI')
  

class ts_form(Form):
    '''Make form for getting user input about how to generate a plot by loan issue date'''
    
    grouping_vars = [('loan_status','loan status'),
                     ('term','loan duration'),                    
                     ('home_ownership','home ownership'),
                     ('grade','loan grade'),
                     ('short_purpose','loan purpose'),
                     ('quantiles','quantiles')]
    group_dict = {key: val for key,val in grouping_vars}

    response_vars = [('ROI', 'returns'),
                     ('int_rate','interest rate'),
                     ('dti','debt-to-income ratio'),
                     ('default_prob','default probability'),
                     ('annual_inc','annual income'),
                     ('emp_length','employment length')]
    
    col_name = RadioField(choices=response_vars, validators=[DataRequired()], default='ROI')
    grouping_var = RadioField('Label', choices=grouping_vars, default='short_purpose')
    smooth_span = IntegerField('Smoothing', 
                               [NumberRange(min=0, max=24, message="Max smoothing 2 years")], default=10)                                
    num_quantiles = IntegerField('Number of quantiles', 
                               [NumberRange(min=1, max=20, message="Max 20 quantiles")], default=5)


class smoothing_form(Form):
    '''Make form for grabbing UI from the time-series plot page. At this point just lets you adjust the 
    amount of smoothing to apply'''
    smooth_span = IntegerField('Smoothing', 
                               [NumberRange(min=0, max=24, message="Max smoothing 2 years")], default=10)

#%%
LD = pd.read_csv(data_dir + data_name, parse_dates=['issue_d'])
# LD['issue_d'] = pd.to_datetime(LD['issue_d'],format = '%Y-%m-%d',unit='D')
print('loaded {0} loans'.format(len(LD)))

#load lookup tables for converting zips and states to county FIPS codes for plotting
fips_to_zip = dill.load(open(data_dir + 'fips_to_zip.p', "rb" ) )
state_fips_dict = dill.load(open(data_dir + 'state_fips_dict.p',"rb"))

# precompute additional columns for convenience when plotting
LD['short_purpose'] = LD['purpose'].map(purpose_map)
LD['issue_year'] = LD['issue_d'].dt.year

# load base map and get state and county paths
app.base_map = LCL.load_base_map(fig_dir + map_name)
(app.county_paths,app.state_paths) = LCL.get_map_paths(app.base_map,fips_to_zip)

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


if __name__ == '__main__':
#    app.run()
    app.run(host='0.0.0.0')
