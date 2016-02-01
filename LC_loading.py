# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:15:09 2015

@author: james
"""

import pandas as pd
import numpy as np
import sys

#%% MAPPING FUNCTIONS
def make_fips_dict(data_path, group_by='zip3'):
    """make dict for converting fips county codes into their corresponding 3-digit zip codes
    """    
    # data_path = '~/Data_Incubator/LC_analysis/Data/'
    zip_to_fips = pd.read_csv(data_path + 'zcta_county_rel_10.txt',
                              dtype={'ZCTA5':'string','STATE':'string','COUNTY':'string'},
                              usecols=['ZCTA5','STATE','COUNTY'])
                              
    zip_to_fips['zip3'] = zip_to_fips['ZCTA5'].apply(lambda x: x[:3]).astype('int')  #add column with just the first 3 digits
    zip_to_fips['ZCTA5'] = zip_to_fips['ZCTA5'].astype('int')
    zip_to_fips['fips'] = zip_to_fips['STATE'] + zip_to_fips['COUNTY']
    
#    fips_dict = {} #for each 3-digit zip this returns the set of counties contained in it
#    for zip3 in zip_to_fips['zip3']self.unique():
#        fips_dict[zip3] = set(zip_to_fips[zip_to_fips['zip3'] == zip3]['fips'])
        
    #get dict for mapping counties to 3-digit zips
    if group_by == 'zip3':
        fips_to_zip = zip_to_fips.groupby('fips')['zip3'].agg(lambda x:x.value_counts().index[0]) #find the mode of zip3 values for each county
    elif group_by == 'zip':
        fips_to_zip = zip_to_fips.groupby('fips')['ZCTA5'].agg(lambda x:x.value_counts().index[0]) #find the mode of zip3 values for each county
    fips_to_zip = pd.Series(fips_to_zip).to_dict()
    
    return fips_to_zip


def make_zip_dict(data_dir):
    """make dict for converting zip codes into fips codes
    """    
    # data_path = '~/Data_Incubator/LC_analysis/Data/'
    zip_to_fips = pd.read_csv(data_dir + 'zcta_county_rel_10.txt',
                              dtype={'ZCTA5':'string','STATE':'string','COUNTY':'string'},
                              usecols=['ZCTA5','STATE','COUNTY'])
                              
    zip_to_fips['ZCTA5'] = zip_to_fips['ZCTA5'].astype('int')
    zip_to_fips['fips'] = zip_to_fips['STATE'] + zip_to_fips['COUNTY']
    zip_to_fips.set_index('ZCTA5',inplace=True)
    zip_to_fips = pd.Series(zip_to_fips['fips']).to_dict()
    return zip_to_fips
    

def load_location_data(data_dir,group_by='zip3'):
    """helper function to load coordinates of each 3-digit zip code"""
    import us
    sys.path.append('/Users/james/data_science/pyzipcode')
    from pyzipcode import ZipCodeDatabase

    # Build pandas zip code database with lat/long coordinates
    zcdb = ZipCodeDatabase() #create zipcode database object
    zip_dict = dict([('zip_code',[]),('latitude',[]),('longitude',[]),('state',[])])
    for state in us.STATES:
        cur_zips = zcdb.find_zip(state='%s' % state.abbr)
        for cur_zip in cur_zips:
            zip_dict['zip_code'].append(cur_zip.zip)
            zip_dict['latitude'].append(cur_zip.latitude)
            zip_dict['longitude'].append(cur_zip.longitude)
            zip_dict['state'].append(cur_zip.state)
            
    zip_data = pd.DataFrame(zip_dict) #make dict into dataframe
    zip_data['zip3'] = zip_data['zip_code'].apply(lambda x: int(x[:3])) #add a column containing 3-digit zips
    zip_data['zip_code'] = zip_data['zip_code'].astype(int)
    zip_to_fips = make_zip_dict(data_dir)
    zip_data['fips'] = zip_data['zip_code'].map(zip_to_fips)
    if group_by == 'zip3':   
        zip_data = zip_data.groupby('zip3').agg({'latitude': np.mean,'longitude':np.mean})
    elif group_by == 'zip':
        zip_data.set_index('zip_code', inplace=True)
    elif group_by == 'fips':
        zip_data = zip_data.groupby('fips').agg({'latitude': np.mean,'longitude':np.mean})
        
    #compute average lat/long coordinates for each 3-digit zip 
    return zip_data
        
        
def load_census_data(census_cols,year=2011):
    """helper function to load desired data from census API"""
    from census import Census
    sys.path.append('/Users/james/data_science/census/')

    c = Census("f2d6925fe44b6e0740ca0cf1d883deb71f2e58d1") #census key
    income_data = pd.DataFrame(c.acs.get(census_cols.keys(), 
                    {'for': 'zip code tabulation area:*'}, year=year)) #pull data by zip code area
    income_data.columns = census_cols.values() + ['ZCTA'] #rename columns    
    income_data[census_cols.values()] = income_data[census_cols.values()].convert_objects(convert_numeric=True)
    
    # if calculating poverty rate   
    if ('hsholds_poverty' in income_data.columns) & ('total_hsholds' in income_data.columns):    
        income_data['poverty_rate'] = income_data['hsholds_poverty']/income_data['total_hsholds'] #calculate percentage of households below poverty level
    
    income_data['zip3'] = income_data['ZCTA'].apply(lambda x: x[:3]) #get 3-digit zip
    return income_data.groupby('zip3').mean() #compute avgs by 3-digit zip
   
   
def load_lending_data(file_names,keep_status=None,keep_terms=None,keep_grades=None):
    """helper function to load lending club data from a specified set of csv files"""
    # load in all files and complie into a dataframe    
    LD = []
    for file_name in file_names:
        if file_name.split('/')[-1] == 'LoanStats3d.csv':            
            cur_data = pd.read_csv(file_name,low_memory=False,skiprows=1) #random header line added to this file
        else:
            cur_data = pd.read_csv(file_name,low_memory=False)
        LD.append(cur_data)    
    LD = pd.concat(LD)
    
    LD = LD.dropna(subset=['zip_code']) # only use subset of data that has the 3-digit zip
    
    #subselect loans in the set of desired status categories
    if keep_status:
        LD = LD[LD['loan_status'].isin(keep_status)]
        
    #subselect loans with the desired term length
    LD['term'] = LD['term'].apply(lambda x: int(str(x.strip(' months'))))    
    if keep_terms:
        LD = LD[LD['term'].isin(keep_terms)]

    #subselect loans that are in desired set of grades
    if keep_grades:
        LD = LD[LD['grade'].isin(keep_grades)]

    LD['int_rate'] = LD['int_rate'].apply(lambda x: float(str(x).strip('%'))) #convert interest rate to float  
    LD['revol_util'] = LD['revol_util'].apply(lambda x: float(str(x).strip('%')))
    
    #process loan descriptions
    LD['num_add_desc'] = LD['desc'].str.count(r'Borrower added on') 
    LD['num_add_desc'].fillna(0, inplace=True)
    only_desc = LD['desc'].str.replace(r'Borrower added on [\S]{8} >','')
    only_desc.fillna('',inplace=True)
    LD['desc_length'] = only_desc.apply(lambda x: len(x))
    
    #convert date data to pandas datetimes
    month_dict = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07',
                  'Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12','nan':'nan'}
    date_cols = [cname for cname in LD.columns if cname[-2:] == '_d']  
    date_cols.append('earliest_cr_line')
    for date_col in date_cols:    
        LD[date_col] = LD[date_col].apply(lambda x: str(x).replace('-','/'))
        LD[date_col] = LD[date_col].apply(lambda x: month_dict[x[:3]] + x[3:] )
        LD[date_col] = pd.to_datetime(LD[date_col],format = '%m/%Y')    

    #calculate the number of months where payments were made
    issue_months = LD['issue_d'].dt.year*12 + LD['issue_d'].dt.month
    last_payment_months = LD['last_pymnt_d'].dt.year*12 + LD['last_pymnt_d'].dt.month
    num_pymnts = last_payment_months - issue_months
    num_pymnts[num_pymnts < 1] = 1 #floor at 1 to avoid undefined results 
    num_pymnts[np.isnan(num_pymnts)] = 1
    num_pymnts[num_pymnts > LD['term']] = LD[num_pymnts > LD['term']]['term'] #cap numpayments at term
    LD['num_pymnts'] = num_pymnts
    
    data_month,data_year = (9, 2015)  #date sampled from this (M,Y)
    LD['loan_age'] = data_month + data_year*12 - issue_months
    
    mth_col = ['mths_since_last_delinq','mths_since_last_major_derog',
               'collections_12_mths_ex_med','mths_since_last_record']
    LD[mth_col] = LD[mth_col].fillna(0)
        
    LD['cr_line_dur'] = LD['issue_d'] - LD['earliest_cr_line']
    LD['cr_line_dur'] = LD['cr_line_dur'].dt.days
    
    LD['last_credit_pull_dur'] = LD['issue_d'] - LD['last_credit_pull_d']    
    LD['last_credit_pull_dur'] = LD['last_credit_pull_dur'].dt.days
    
    #get issue date in days relatie to earliest loan
    times = (LD.issue_d - LD.issue_d.min()).dt.days.values
    LD['issue_day'] = times

    #convert employment length to ordinal values
    emp_dict = {'n/a':0, '< 1 year':0.5, '1 year':1, '2 years':2, '3 years':3,
                '4 years':4, '5 years':5, '6 years':6, '7 years':7,
                '8 years':8, '9 years':9, '10+ years':10}
    LD['emp_length'] = LD['emp_length'].map(emp_dict)
    
    LD['zip3'] = LD['zip_code'].apply(lambda x: x[:3])  #add column with just the first 3 digits
    
    ir = LD['int_rate']/12./100.
    P = LD['funded_amnt'] 
    direct_install = P*(ir + ir/((ir + 1)**LD['term'] - 1))
    install_err = (direct_install - LD['installment'])/direct_install*100.
    err_thresh = 0.05  
    bad = abs(install_err) > err_thresh
    print('Eliminating {} bad installment calcs'.format(np.sum(bad)))
    LD = LD[~bad]     
        
    bad = (LD.grade != 'A') & (LD.int_rate == 6.0)
    print('Eliminating {} loans with bad interest rates'.format(np.sum(bad)))
    LD = LD[~bad]     

    observed_labels = ['Charged Off','Fully Paid']
    LD['is_observed'] = LD.loan_status.isin(observed_labels)        
        
    return LD


#%%
from bs4 import BeautifulSoup
import re
def load_base_map(map_file, ax_xml = False): 
    '''Load in SVG base map'''
    with open(map_file, 'r') as F:
        if ax_xml:
            base_map = BeautifulSoup(F.read(),'xml')
        else:
            base_map = BeautifulSoup(F.read())
    return base_map
    
def get_map_paths(base_map,fips_to_zip):    
    '''get paths for counties and states'''
    county_ids = re.compile(r'[^State_Lines|separator]')
    county_paths = base_map.findAll('path', id=county_ids) # Find counties
    state_paths = base_map.findAll('path', id='State_Lines')  
    county_paths = [p for p in county_paths if p['id'] in fips_to_zip]
    return (county_paths,state_paths)
