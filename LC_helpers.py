# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 16:05:51 2016

@author: James McFarland
"""
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import matplotlib.colorbar as cbar

def get_amortization_schedules(LD, term, calc_int = True):
    '''Calculate loan amortization schedules for all loans in dataframe with a specified term length.
    INPUTS: 
        LD: pandas dataframe with loan data.
        term: int specifying loan term in months.
        calc_int: boolean specifying whether or not to calculate interest schedule as well (default True).
    OUTPUTS:
        Tuple containing (prin_sched, prin_sched_obs, [int_sched])
            prin_sched: [n_loans x term+1] array of scheduled outstanding principal.
            prin_sched_obs: [n_loans x term+1] array of observed outstanding principal by month.
            [int_sched]: [n_loans x term+1] array of scheduled interest payments by month'''

    n_loans = len(LD)
    assert LD['term'].unique() == [term]
    term_array = np.tile(np.arange(term+1),(n_loans,1)) #array of vectors going from 0 to term
    #array of copies of the number of payments for each loan
    num_payments_array = LD['num_pymnts'][:,np.newaxis] * np.ones((n_loans,term+1)) 
    monthly_int = LD['int_rate'][:,np.newaxis]/12./100. #monthly interest rate for each loan
    
    #get amortization schedule of outstanding principal
    prin_sched = ((1 + monthly_int) ** term_array - 1) / ((1 + monthly_int) ** term - 1)
    prin_sched = LD['funded_amnt'][:,np.newaxis]*(1 - prin_sched)

    #compute schedule of payments actually received
    prin_sched_obs = prin_sched.copy() 
    prin_sched_obs[num_payments_array <= term_array] = 0 #havent received future payments yet
   
    if not calc_int:
        return (prin_sched, prin_sched_obs)
    else: #calc schedule of interest payments if requested
        int_sched = LD['installment'][:,np.newaxis] + np.diff(prin_sched,axis=1)    
        #include a month0, for loans that default before they make any payments
        int_sched = np.concatenate((np.zeros((n_loans,1)),int_sched),axis=1) 
        return (prin_sched, prin_sched_obs, int_sched)
        

def get_best_returns(LD, term, service_rate = 0.01):
    """compute best possible returns (ie if no loans default). 
        INPUTS: 
            LD: dataframe of loan data
            term: loan term (scalar) for the data in LD
            service_rate: percent of payments LC takes as service charge
        RETURNS: 
            best_NAR as a pandas series"""

    tot_pymnts = LD['term'] * LD['installment']
    tot_sc = tot_pymnts * service_rate #total paid in service fees to LC
    net_gains = tot_pymnts - tot_sc - LD['funded_amnt']

    (p_sched, _) = get_amortization_schedules(LD, term, calc_int = False)
    csum_prnc = np.sum(p_sched,axis=1) #get summed cumsum of outstanding princ 

    net_returns = net_gains/LD['funded_amnt']
    mnthly_returns = net_gains/csum_prnc #avg monthly return weighted by outstanding prncp
    best_NAR = (1 + mnthly_returns) ** 12 - 1 

    return best_NAR
 

def get_NARs(LD, term, service_rate = 0.01):
    """compute net-annualized returns (NARs) for all loans in dataframe LD. 
        INPUTS: 
            LD: dataframe of loan data
            term: loan term (scalar) for the data in LD
            service_rate: percent of payments LC takes as service charge
        RETURNS: 
            tuple containing (NAR, net_returns, prnc_weight) 
            NAR: net-annualized return (as fraction)
            net_returns: Net gains normalized by loan amount
            prnc_weight: Summed value of cumulative sum of outstanding prnc (denom in LC NAR calc)    
    """
      
    #net gain is total payment recieved less principal less collection recovery fees
    tot_gain = LD['total_pymnt'] - LD['total_rec_prncp'] - LD['collection_recovery_fee']
    
    #compute loss for charged off loans 
    tot_loss = LD['funded_amnt'] - LD['total_rec_prncp']
    tot_loss[LD['loan_status'] != 'Charged Off'] = 0 #set loss to 0 for loans that arent charged off
    
    '''for loans where the principal is charged off and then later fully recovered, set the 
    loan duration to be the loan term. This is a hack, but we dont know when it's actually 
    recovered, and this at least prevents these loans from appearing like they provide gigantic
    returns. This also doesnt get rid of all these corner case, just the most obvious ones.'''
    post_CO_rec = (LD['loan_status'] == 'Charged Off') & (LD['recoveries'] > LD['funded_amnt'])
    LD.ix[post_CO_rec,'num_pymnts'] = LD.ix[post_CO_rec,'term']        
          
    (p_sched, p_sched_obs) = get_amortization_schedules(LD, term, calc_int = False)
    
    csum_prnc = np.sum(p_sched_obs,axis=1) #get summed cumsum or outstanding princ 
    prnc_weight = csum_prnc/LD['funded_amnt'] #normalized value of the denominator in LC NAR calc

    #compute total service charge fee for each loan
    service_charge = service_rate * LD['total_pymnt']
    #find set of loans that are paid off in full after less than a year
    early_repay = (LD['total_pymnt'] == LD['funded_amnt']) & \
                    (LD['num_pymnts'] <= 12) 
    #LC has a max service charge in these cases 
    max_sc = LD['num_pymnts'] * LD['installment'] * service_rate
    service_charge[early_repay] = max_sc[early_repay] #cap service charge
    
    #take interest made, less lossed principal less total service fees to get net gains
    net_gains = (tot_gain - tot_loss - service_charge)
    net_returns = net_gains/LD['funded_amnt']
    mnthly_returns = net_gains/csum_prnc #avg monthly return weighted by outstanding prncp
    NAR = (1 + mnthly_returns) ** 12 - 1 #annualize

    return (NAR, net_returns, prnc_weight)

    
def get_expected_NARs(LD, term, hazard_funs, service_rate = 0.01):
    """compute NARs for all loans in dataframe LD. Uses precomputed hazard functions 
    to estimate expected NARs given loan status for immature loans.
        INPUTS: 
            LD: dataframe of loan data
            term: loan term (scalar) for the data in LD
            hazard_funs: array of hazard functions for each loan grade
            service_rate: percent of payments LC takes as service charge
        RETURNS: 
            tuple containing (exp_NAR, tot_default_prob, exp_num_pymnts, exp_net_returns, exp_prnc_weight)  
            exp_NAR: expected net-annualized return.
            tot_default_prob: estimated probability of defaulting at any point.
            exp_num_pymnts: expected number of payments recieved.
            exp_net_returns: expected net gains normalized by loan amount
            exp_prnc_weight: expected value of denom in LC NAR calc        
    """
    
    #https://www.lendingclub.com/info/demand-and-credit-profile.action
    outcome_map = { #expected principal recovery given status, view this is prob of all princ being charged off
        'Current': 0,
        'Fully Paid': 0,
        'In Grace Period': 28,
        'Late (16-30 days)': 58,
        'Late (31-120 days)': 74,
        'Default': 89,
        'Charged Off': 100}

    (p_sched, p_sched_obs, int_sched) = get_amortization_schedules(LD, term)
    term_array = np.tile(np.arange(term+1),(len(LD),1)) #array of vectors going from 0 to term
    num_payments_array = LD['num_pymnts'][:,np.newaxis] * np.ones((len(LD),term+1)) #array of copies of the number of payments for each loan

    interest_paid = np.cumsum(int_sched, axis=1) # total interest received at time t
    #your net gain if you only received payments up to time t
    gains_at_time = interest_paid - term_array * LD['installment'][:,np.newaxis] * service_rate - p_sched
    princ_csum = np.cumsum(p_sched, axis=1) #cumulative sum of outstanding principal at time t   
    monthly_returns = gains_at_time / princ_csum #avg monthly returns if default at t

    #make array containing grade-conditional hazard fnxs for each loan             
    grades = np.sort(LD.grade.unique()) #set of unique loan grades
    grade_map = {grade: idx for idx,grade in enumerate(grades)} #dict mapping grade letters to index values in the hazard_funs arrays
    grade_index = LD['grade'].apply(lambda x: grade_map[x]).astype(int)
    def_probs = hazard_funs[grade_index.values,:] #default probabilities at each time 

    def_probs[term_array < num_payments_array] = 0 #prob of defaulting for made payments is 0
    def_probs[(LD.loan_status == 'Fully Paid').values,:] = 0 #set default probs for paid loans to 0
    
    #handle charged off loans
    CO_set = (LD.loan_status == 'Charged Off').values #set of charged off loans
    CO_probs = def_probs[CO_set,:] #get default probs for those loans
    CO_probs[num_payments_array[CO_set,:] == term_array[CO_set,:]] = 1. # default prob is 1 at time where default occurred
    CO_probs[num_payments_array[CO_set,:] < term_array[CO_set,:]] = 0.
    def_probs[CO_set,:] = CO_probs 
    
    #conditional probability of being charged off 
    prob_CO = pd.to_numeric(LD['loan_status'].replace(outcome_map), errors='coerce')/100.

    #find loans that are 'in limbo', and handle them separately
    in_limbo_labels = ['In Grace Period', 'Late (16-30 days)', 'Late (31-120 days)', 'Default']
    in_limbo = LD.loan_status.isin(in_limbo_labels).values    
    dp_limbo = def_probs[in_limbo,:]
    #set default prob at current time based on map of recov probs
    dp_limbo[num_payments_array[in_limbo,:] == term_array[in_limbo,:]] = prob_CO[in_limbo] 
    
    '''assume that if the loan doesnt end up getting written off then we return to normal 
    default probs for remaining time. This means that we have to weight the contribution of 
    these guys by the prob that you didnt default given delinquent status'''
    new_weight_mat = (1 - prob_CO[in_limbo, np.newaxis]) * dp_limbo    
    later_time = num_payments_array[in_limbo,:] < term_array[in_limbo,:]
    dp_limbo[later_time] = new_weight_mat[later_time] 
    
    def_probs[in_limbo,:] = dp_limbo #store in-limbo values back in full array
    
    tot_default_prob = np.sum(def_probs, axis=1) #total default probs
    pay_prob = 1 - tot_default_prob #total payment prob
    
    #expected returns is average monthly returns given default at each time, weighted by conditional probs of defaulting
    exp_mnthly_returns = np.sum(def_probs * monthly_returns,axis=1) + pay_prob * monthly_returns[:,-1]
    exp_net_gains = np.sum(def_probs * gains_at_time,axis=1) + pay_prob * gains_at_time[:,-1]
    exp_net_returns = exp_net_gains / LD['funded_amnt'] # get relative returns
    exp_csum = np.sum(def_probs * princ_csum, axis=1) + pay_prob * princ_csum[:,-1] #expected value of cumulative sum weight factor
    exp_prnc_weight = exp_csum/LD['funded_amnt'] # normalize weight factor by loan amount
    exp_num_pymnts = np.sum((term_array + 1) * def_probs, axis=1) + pay_prob*term #expected number of payments
    exp_NAR = (1 + exp_mnthly_returns) ** 12 - 1
    
    return (exp_NAR, tot_default_prob, exp_num_pymnts, exp_net_returns, exp_prnc_weight)
    
    
def extract_fips_coords(county_paths):
    '''Get the geographic coordinates (AU) of each county (fips) in the map.
    Useful for filling in missing data based on nearest neighbors
    INPUTS:
        county_paths: set of path objects from bs4 for each county.
    OUTPUTS:
        dataframe of coordinate values for each fips'''
    coord_dict = {}
    re_path = re.compile(r'path d=\"(.*)\" id')
    re_ML = re.compile(r'[MLz]')
    for county in county_paths:
        path = re.findall(re_path,str(county))[0]
        coord_list = []
        numpair_set = re.split(re_ML,path)
        for numpair in numpair_set:
            if len(numpair) > 1:
                coord_list.append([float(el) for el in numpair.split(',')])
        avg_coords = np.mean(np.array(coord_list),axis=0)
        coord_dict[county['id']] = avg_coords
    coord_df = pd.DataFrame(coord_dict).transpose()
    coord_df.index.name = 'fips'
    coord_df.columns = ['x_coord','y_coord']
    return coord_df


def fill_missing_fips(ktree,map_coords,fips_data):
    '''Take a dataframe indexed by fips coordinates and fill in data for
    missing fips using nearest neighbor lookup.
    INPUTS:
        ktree: pre-computed ktree object.
        map_coords: dataframe of coordinates for each fips.
        fips_data: dataframe keyed by fips that we want to fill in.
    OUTPUTS:
        new_data: data frame with missing FIPS filled in.'''

    missing_fips = [fips for fips in map_coords.index.values 
                    if fips not in fips_data.index.values]
    new_data = {}
    for fips in missing_fips:
        knn_ds, knn_ids = ktree.query(map_coords.ix[fips].values,2)
        closest_fips = str(map_coords.index[knn_ids[1]])
        cnt = 3
        while closest_fips not in fips_data.index.values: #if nearest neighbor doesn't exist
            knn_ds, knn_ids = ktree.query(map_coords.ix[fips].values,cnt)
            closest_fips = str(map_coords.index[knn_ids[-1]])
            cnt += 1

        closest_data = fips_data.ix[closest_fips,'preds']
        new_data[fips] = closest_data

    new_df = pd.DataFrame({'preds':new_data})    
    new_df.index.name = 'fips'
    new_data = pd.concat([fips_data,new_df],axis=0)
    return new_data
 
 
def paint_map(data, soup_map, county_paths = None, fips_to_zip = None, 
              color = 'blue', get_cbar = True, name_legend_map=None, agg_fun='mean'):
    '''paints the data onto an svg base map based on either zip3 or states.
    INPUTS:
        data: pandas series of data, indexed by the geographic grouping variable.
        soup_map: bs4 object containing the svg base map.
        county_paths: pre-extracted county paths.
        fips_to_zip: dict for mapping from fips to zip codes (3 or 5 digit).
        color: base color scheme.
        get_cbar: bool specifying whether to export a figure containing the color bar.
        agg_fun: aggregation function to use.
    OUTPUTS:
        cbar_fig: optional handle to the colobar.'''
    
    #set base path style properties
    if (data.index.name == 'zip3') | (data.index.name == 'zip_code') | (data.index.name == 'fips'):
        county_path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:1.0;fill-opacity:1.0;stroke-width:0.2;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'
    elif data.index.name == 'state_fips':
        county_path_style = 'font-size:12px;fill-rule:nonzero;stroke:#FFFFFF;stroke-opacity:0.0;fill-opacity:1.0;stroke-width:0.2;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:'
    
    #make color palette
    if color == 'cube':
        pal = sns.cubehelix_palette(as_cmap=True) 
        missing_path_style = 'font-size:12px;fill-rule:nonzero;stroke:#737373;stroke-opacity:1.0;fill-opacity:1.0;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:#737373'
    else:
        pal = sns.light_palette(color, as_cmap=True) 
        missing_path_style = 'font-size:12px;fill-rule:nonzero;stroke:#000000;stroke-opacity:1.0;fill-opacity:1.0;stroke-width:0.1;stroke-miterlimit:4;stroke-dasharray:none;stroke-linecap:butt;marker-start:none;stroke-linejoin:bevel;fill:#000000'
    
    #color color map properties
    cNorm  = colors.Normalize(vmin = data.quantile(0.05), vmax = data.quantile(0.95))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=pal)
        
    #apply color map to scalar values to get a series of hex color codes
    data = data.apply(scalarMap.to_rgba)
    data = data.apply(colors.rgb2hex)
    
    for p in county_paths: #paint mapping
        if (data.index.name == 'zip3') | (data.index.name == 'zip_code'):      
            lookup = fips_to_zip[p['id']]
        elif data.index.name == 'state_fips':
            lookup = p['id'][:2]
        elif data.index.name == 'fips':
            lookup = p['id']
            
        if lookup in data:
            p['style'] = county_path_style + data[lookup]
        else: 
            p['style'] = missing_path_style
    
    if get_cbar: #make color bar if desired
        cbar_fig,ax=plt.subplots(1,1,figsize=(6,1))
        cb1 = cbar.ColorbarBase(ax,cmap=pal, norm=cNorm, orientation='horizontal')  
        if agg_fun == 'count':
            label = 'Number of loans (thousands)'
        else:
            agg_fun_map = {'mean':'avg. ',
                           'median':'median ',
                           'std':'SD of '}
            label = agg_fun_map[agg_fun] + name_legend_map[data.name]
        cb1.set_label(label)
        plt.tight_layout()
           
        return cbar_fig 
    
    
def compute_group_avgs(data, col_name, group_by, agg_fun='mean', 
                       state_fips_dict = None, min_counts = 50):
    """Get series of within-group-stats based on either addr_state or zip3.
    INPUTS:
        data: pandas dataframe.
        col_name: column name of response variable.
        group_by: column name of grouping variable (either zip3 or addr_state).
        agg_fun: aggregation function to use (default 'mean').
        state_fips_dict: dict to map state names to fips if needed.
        min_counts: min number of loans per group to include.
    OUTPUTS:
        group_avgs: pandas series with results."""
    group_counts = data.groupby(group_by)['int_rate'].count()
    if agg_fun=='count': #if we want counts just replace avgs with this
        group_avgs = group_counts/1000 #keep in thousands
    elif agg_fun=='mean':
        group_avgs = data.groupby(group_by)[col_name].mean()
    elif agg_fun=='median':
        group_avgs = data.groupby(group_by)[col_name].median()
    elif agg_fun=='std':
        group_avgs = data.groupby(group_by)[col_name].std()
    if agg_fun != 'count':
        group_avgs = group_avgs[group_counts >= min_counts] 
    
    if group_by == 'addr_state':
        group_avgs.index = group_avgs.index.map(lambda x: state_fips_dict[x])
        group_avgs.index.name = 'state_fips'
        
    return group_avgs   
