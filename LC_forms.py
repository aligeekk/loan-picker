# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:29:45 2016

@author: James McFarland
"""
from wtforms import Form, RadioField, IntegerField, SelectMultipleField, IntegerField, BooleanField, DecimalField
from wtforms.validators import DataRequired, NumberRange

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
    response_vars = [('ROI', 'annual returns'), 
                     ('int_rate', 'interest rate'),
                     ('annual_inc', 'annual income'),
                     ('default_prob', 'default probability'),
                     ('dti', 'debt-to-income ratio'),
                     ('emp_length', 'employment length'),
                     ('funded_amnt','loan amount')]
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
                     ('emp_length','employment length'),
                     ('funded_amnt','loan amount')]
    
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


class cl_form(Form):
    '''Make form for getting user input about constructing simulated loan portfolio'''
    port_size = IntegerField('Port_size', 
                             [NumberRange(min=1, max=10000, message="Exceeded max portfolio size")], default=100)                                
    max_dp = DecimalField('max_dp', 
                             [NumberRange(min=0, max=1, message="Outside allowable range")], default=1.0)                                

    use_A = BooleanField('use_A', default=True)                             
    use_B = BooleanField('use_B', default=True)                             
    use_C = BooleanField('use_C', default=True)                             
    use_D = BooleanField('use_D', default=True)                             
    use_E = BooleanField('use_E', default=True)                             
    use_F = BooleanField('use_F', default=True)                             
