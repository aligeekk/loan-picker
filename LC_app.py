# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 09:59:19 2016

@author: James McFarland
"""

from flask import Flask, render_template, request, redirect, url_for, Markup
from wtforms import Form, RadioField, IntegerField, SelectMultipleField, IntegerField
from wtforms.validators import DataRequired, NumberRange
#import matplotlib
# Force matplotlib to not use any Xwindows backend.
#matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
#import seaborn as sns
import re
import pandas as pd
from bs4 import BeautifulSoup
import LC_helpers as LCH
import LC_loading as LCL
import LC_bokeh_plotting as LC_bok
import pickle
#plt.ioff()

name_legend_map = {'counts': 'Number of loans (thousands)',
			  'ROI': 'ROI (%)',
			  'int_rate': 'interest rate (%)',
			  'default_prob': 'default probability',
			   'dti': 'Debt-to-income ratio',
			   'emp_length':'employment length (months)',
                    'annual_inc':'annual income ($)'}

#%%
app_title = 'Loan Explorer'
app = Flask(__name__)
app.secret_key = '\xb6\xbcr\xc4\xb5\x9cY\x03\xcdI\x15oR:\xdbJD\xb1c\x00+\x1c\x926'

class map_form(Form):
    
    grouping_vars = [('addr_state', 'by state'),
                     ('zip3','by 3-digit zip')]
    group_dict = {key: val for key,val in grouping_vars}

    agg_funs = [('count', 'count'),
                ('mean','mean'),
                ('std','std dev'),
                ('median','median')]
                
    agg_dict = {key: val for key,val in agg_funs}

    response_vars = [('ROI', 'returns'),
                     ('int_rate','interest rate'),
                     ('annual_inc','annual income'),
                     ('default_prob','default probability'),
                     ('dti','debt-to-income ratio'),
                     ('emp_length','employment length')]

    var_dict = {key: val for key,val in response_vars}
    
    
# makes a form for grabbing user input
    grouping_var = RadioField('Label', choices=grouping_vars,
                              validators=[DataRequired()],default='zip3')
    agg_fun = RadioField('Label3', choices=agg_funs,
                         validators=[DataRequired()],default='mean')
    col_name = RadioField('Label2', choices=response_vars,
                          validators=[DataRequired()],
                          default='ROI')
                          
    # color_name = RadioField('Label3', choices=[('blue','blue'),
    #                                            ('red','red'),
    #                                            ('green','green')],
    #                                             validators=[DataRequired()],
    #                                         default='blue')

class ts_form(Form):
    # makes a form for grabbing user input
    grouping_vars = [('loan_status','loan status'),
                     ('term','loan duration'),                    
                     ('home_ownership','home ownership'),
                     ('grade','loan grade'),
                     ('issue_year','issue year'),
                     ('short_purpose','loan purpose'),
                     ('quantiles','quantiles')]
                     
    group_dict = {key: val for key,val in grouping_vars}

    grouping_var = RadioField('Label', choices=grouping_vars, default='short_purpose')
    response_vars = [('ROI', 'returns'),
                     ('int_rate','interest rate'),
                     ('dti','debt-to-income ratio'),
                     ('default_prob','default probability'),
                     ('annual_inc','annual income'),
                     ('emp_length','employment length')]
                     
    col_name = RadioField(choices = response_vars, validators=[DataRequired()],
                                                               default='ROI')
    smooth_span = IntegerField('Smoothing', 
                               [NumberRange(min=0, max=24, message="Max smoothing 2 years")],
                                default=10)
                                
    num_quantiles = IntegerField('Number of quantiles', 
                               [NumberRange(min=1, max=20, message="Max 20 quantiles")],
                                default=5)

class smoothing_form(Form):
    smooth_span = IntegerField('Smoothing', 
                               [NumberRange(min=0, max=24, message="Max smoothing 2 years")],
                                default=10)

#%%
data_dir = 'static/data/'
fig_dir = 'static/images/'
data_name = 'all_loans_proc_short'
LD = pd.read_csv(data_dir + data_name)
LD['issue_d'] = pd.to_datetime(LD['issue_d'],format = '%Y-%m-%d',unit='D')

print('loaded {0} loans'.format(len(LD)))

fips_to_zip = pickle.load(open(data_dir + 'fips_to_zip.p', "rb" ) )
state_fips_dict = pickle.load(open(data_dir + 'state_fips_dict.p',"rb"))

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
LD['short_purpose'] = LD['purpose'].map(purpose_map)
LD['issue_year'] = LD['issue_d'].dt.year

app.base_map = LCL.load_base_map(fig_dir + 'USA_Counties_raw.svg')
(app.county_paths,app.state_paths) = LCL.get_map_paths(app.base_map,fips_to_zip)

#%%
@app.route('/') #redirect to index page
def main():
    return redirect('/index')

# page where user selects desired stock features to plot
@app.route('/index',methods=['GET'])
def index():
    return render_template('cover.html', app_title = app_title) #if request method was GET

@app.route('/home',methods=['GET'])
def home():
    return render_template('home.html', app_title = app_title) #if request method was GET

@app.route('/loan_mapping',methods=['GET','POST'])
def loan_mapping(map_rendered=False):
    mform =  map_form(request.form)    
    #map_rendered = False
    if request.method == 'POST' and mform.validate():  #if user is posting, get form data and store it           
         map_rendered = True         
         data = LCH.compute_group_avgs(LD, mform.data['col_name'], 
                                       mform.data['grouping_var'], 
                                       mform.data['agg_fun'],
                                       state_fips_dict = state_fips_dict, 
                                       min_counts = 50)
                       
         data.name = mform.data['col_name']
         if mform.data['agg_fun'] == 'count':
             data.name = 'counts'
         pal = LCH.paint_map(data, app.base_map, app.county_paths, fips_to_zip, 
                             color='cube', agg_fun=mform.data['agg_fun'])        
         
         plt.savefig(fig_dir + 'map_cbar.png', dpi=500, format='png')
         plt.close()
         
    return render_template('loan_mapping.html', map_form=mform, svg=Markup(str(app.base_map)),
                rnum=np.random.randint(0,100000), map_rendered = map_rendered) #if request method was GET

@app.route('/reset_map',methods=['GET'])
def reset_map():
    print('resetting map')
    app.base_map = LCL.load_base_map(fig_dir + 'USA_Counties_raw.svg')
    (app.county_paths,app.state_paths) = LCL.get_map_paths(app.base_map,fips_to_zip)
    return redirect('/loan_mapping') #if request method was GET


@app.route('/expected_returns',methods=['GET'])
def expected_returns():
    return render_template('expected_returns.html',rnum=np.random.randint(0,100000)) #if request method was GET

@app.route('/models',methods=['GET'])
def models():
    return render_template('models.html',rnum=np.random.randint(0,100000)) #if request method was GET


@app.route('/time_series_form',methods=['GET','POST'])
def time_series_form():
    mform =  ts_form(request.form)    
    if request.method == 'POST' and mform.validate():  #if user is posting, get form data and store it           
        app.ts_form = mform
        return redirect('/time_series') #otherwise, go to the graph page
    
    return render_template('time_series_form.html', ts_form = mform) #if request method was GET


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
