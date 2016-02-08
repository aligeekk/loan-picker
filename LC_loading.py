# -*- coding: utf-8 -*-
"""
Created on Fri Dec 18 12:15:09 2015

@author: James McFarland
"""

import pandas as pd
import numpy as np
import sys
from bs4 import BeautifulSoup
import re
import datetime

def make_fips_to_zip_dict(data_path, group_by='zip3'):
    """Takes a csv file with 5-digit census zip-code tabulation areas, and makes a dict
    for converting fips county codes into their corresponding 3-digit zip codes. 
    Set group_by to zip to return 5-digit instead of 3-digit zip codes."""

    zip_to_fips = pd.read_csv(data_path + 'zcta_county_rel_10.txt',
                              dtype={'ZCTA5':'string','STATE':'string','COUNTY':'string'},
                              usecols=['ZCTA5','STATE','COUNTY'])
                              
    zip_to_fips['zip3'] = zip_to_fips['ZCTA5'].apply(lambda x: x[:3]).astype('int')  #add column with just the first 3 digits
    zip_to_fips['ZCTA5'] = zip_to_fips['ZCTA5'].astype('int')
    zip_to_fips['fips'] = zip_to_fips['STATE'] + zip_to_fips['COUNTY']
            
    #get dict for mapping counties to 3-digit zips
    if group_by == 'zip3':
        fips_to_zip = zip_to_fips.groupby('fips')['zip3'].agg(lambda x:x.value_counts().index[0]) #find the mode of zip3 values for each county
    elif group_by == 'zip':
        fips_to_zip = zip_to_fips.groupby('fips')['ZCTA5'].agg(lambda x:x.value_counts().index[0]) #find the mode of zip3 values for each county
    fips_to_zip = pd.Series(fips_to_zip).to_dict()
    
    return fips_to_zip


def make_zip_to_fips_dict(data_dir):
    """make dict for converting from zip codes into fips codes. Uses a csv file of county fips codes 
    for each census zcta"""    
    zip_to_fips = pd.read_csv(data_dir + 'zcta_county_rel_10.txt',
                              dtype={'ZCTA5':'string','STATE':'string','COUNTY':'string'},
                              usecols=['ZCTA5','STATE','COUNTY'])
                              
    zip_to_fips['ZCTA5'] = zip_to_fips['ZCTA5'].astype('int')
    zip_to_fips['fips'] = zip_to_fips['STATE'] + zip_to_fips['COUNTY']
    zip_to_fips.set_index('ZCTA5',inplace=True)
    zip_to_fips = pd.Series(zip_to_fips['fips']).to_dict()

    return zip_to_fips
    

def load_location_data(data_dir,group_by='zip3'):
    """Helper function to load lat/long coordinates of each 3-digit zip code.
    Returns data grouped by either 3-digit zip, 5-digit zip, or county FIPS"""
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
    zip_to_fips = make_zip_to_fips_dict(data_dir)
    zip_data['fips'] = zip_data['zip_code'].map(zip_to_fips)
    if group_by == 'zip3':   
        zip_data = zip_data.groupby('zip3').agg({'latitude': np.mean,'longitude':np.mean})
    elif group_by == 'zip':
        zip_data.set_index('zip_code', inplace=True)
    elif group_by == 'fips':
        zip_data = zip_data.groupby('fips').agg({'latitude': np.mean,'longitude':np.mean})
        
    return zip_data
        

def load_lending_data(file_names,keep_status=None,keep_terms=None,keep_grades=None):
    """Load Lending Club data into pandas dataframe from a specified set of csv files.
    INPUTS:
        file_names: list of csv file name paths.
        keep_status: optional list of loan status values to keep. Default is all
        keep_terms: optional list of loan terms [36,60] to keep. Default is all.
        keep_grades: optional list of loan grades to keep. Default is A-F (ignoring G).
    OUTPUTS:
        LD: Dataframe with compiled loan data."""
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
    LD['revol_util'] = LD['revol_util'].apply(lambda x: float(str(x).strip('%'))) #convert revol_util to float
    
    '''Process loan descriptions. Additional comments can be added at later times by the borrowers
    and these get flagged. Many loans dont have any description available initially.
    I calculate the number of later-additions to the loan description, as well as the length (in characters)
    of the ORIGINAL loan description.'''
    LD['num_add_desc'] = LD['desc'].str.count(r'Borrower added on') 
    LD['num_add_desc'].fillna(0, inplace=True)
    only_desc = LD['desc'].str.replace(r'Borrower added on [\S]{8} >','')
    only_desc.fillna('',inplace=True)
    LD['desc_length'] = only_desc.apply(lambda x: len(x))
    
    #convert date data to pandas datetimes
    month_dict = {'Jan':'01','Feb':'02','Mar':'03','Apr':'04','May':'05','Jun':'06','Jul':'07',
                  'Aug':'08','Sep':'09','Oct':'10','Nov':'11','Dec':'12','nan':'nan'}
    date_cols = [cname for cname in LD.columns if cname[-2:] == '_d']  
    date_cols.append('earliest_cr_line') #this is also a date column]
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
    
    data_month, data_year = (9, 2015)  #date sampled from this (M,Y)
    LD['loan_age'] = data_month + data_year*12 - issue_months #months since issue data from time data was compiled
    
    '''Process columns given by number of months since some event (which may or may not have happened)
    For cases where values are missing (ie the event in question didnt happen, I take an ad-hoc approach
    of setting those values to about 1SD above the max observed value (ish). This could obviously be improved,
    but also shouldnt matter much'''
    mth_col = ['mths_since_last_delinq','mths_since_last_major_derog','mths_since_last_record']
    LD[mth_col] = LD[mth_col].fillna(200) #this is about 1 SD above the max non-nan value.
    LD['collections_12_mths_ex_med'] = LD['collections_12_mths_ex_med'].fillna(0) #may want to think about how to handle these better
        
    #duration in months from loan issue to when credit line was opened
    LD['cr_line_dur'] = LD['issue_d'] - LD['earliest_cr_line']
    LD['cr_line_dur'] = LD['cr_line_dur'].dt.days
    
    #months from loan issue to when credit info was last pulled
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
    
    '''Calculate installment and check if this agrees with provided numbers. If not
    just discard those (very rare) loans where some number is wrong'''
    ir = LD['int_rate']/12./100.
    P = LD['funded_amnt'] 
    direct_install = P*(ir + ir/((ir + 1)**LD['term'] - 1))
    install_err = (direct_install - LD['installment'])/direct_install*100.
    err_thresh = 0.05  
    bad = abs(install_err) > err_thresh
    print('Eliminating {} bad installment calcs'.format(np.sum(bad)))
    LD = LD[~bad]     
        
    #There are some loans where the interest rate is set to 6.0 obviously erroneously
    bad = (LD.grade != 'A') & (LD.int_rate == 6.0)
    print('Eliminating {} loans with bad interest rates'.format(np.sum(bad)))
    LD = LD[~bad]     

    #store a bool of whether the loan outcome is 'observed' or not.
    observed_labels = ['Charged Off','Fully Paid']
    LD['is_observed'] = LD.loan_status.isin(observed_labels)        
        
    return LD


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


#%% Functions for parsing loan data pulled from the LC API
def get_cr_line_dur(date_string):
    '''Get number of days since credit line was opened'''
    cur_date = datetime.date.today()
    date_string = re.findall(r'[^T]+',date_string)[0]
    cr_start = datetime.datetime.strptime(date_string, '%Y-%m-%d').date()    
    return (cur_date - cr_start).days
    
def get_desc_length(desc):
    '''Return length of description if any'''
    if desc:
        only_desc = desc.replace(r'Borrower added on [\S]{8} >','')  
        return len(only_desc)
    else: 
        return 0

def get_num_descs(desc):
    '''Return number of descriptions added'''
    if desc:
        return len(re.findall(r'Borrower added on',desc))
    else: 
        return 0
        
def get_mnths_since(mths_since):
    '''Handle features that are months since some event that might not have happened'''
    if mths_since:
        return mths_since
    else:
        return 200. #this is just an arbitrary value to fill in for Nans, better than 0.
        
def get_emp_length(emp_length):
    '''Handle possible missing employment length data'''
    if emp_length:
        return emp_length/12. # in years
    else:
        return 0
        
def get_zip_loc(addrZip, zip3_data, loc_type):
    '''Get lat or long for zip3'''
    zip3 = int(addrZip[:3])
    if zip3 in zip3_data:
        return zip3_data.ix[zip3][loc_type]
    else:
        nearest_zip = min(zip3_data.index.values, key=lambda x:abs(x-zip3))
        return zip3_data.ix[nearest_zip][loc_type]

