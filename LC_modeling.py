
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MaxAbsScaler, StandardScaler, MinMaxScaler
import sklearn.base
import numpy as np

# HELPER FUCNTIONS
def annualize_returns(mnthly_returns):
    '''Compute annualized returns given avg monthly returns'''
    ann_ret = ((mnthly_returns + 1)**12 - 1) * 100
    return ann_ret



def pick_K_returns(y_pred, returns, weights, K_list, n_boots=500, sub_marg=True):
    '''Compute avg returns (NAR) on portfolios of different sizes selected
    by taking the top K loans based on model-predicted ROI.
    INPUTS:
        y_pred: vector of model predicted ROIs.
        returns: vector of net_returns (observed/expected) for each loan
        weights: vector of weight factors (denominator in LC NAR calc).
        K_list: list of different loan portfolio sizes to test
        n_boots: number of bootstrap resampling iterations
        sub_marg: boolean indicating whether to subtract off marginal returns
    OUTPUTS:
        pick_k_avgs: array of portfolio returns size: (n_boots, len(K_list))
        '''
    y_comb = pd.DataFrame({'returns': returns, 'pred':y_pred,
                        'weights': weights})

    pick_k_avgs = np.zeros((n_boots,len(K_list)))
    for n in xrange(n_boots):
        y_samp = y_comb.sample(frac=1, replace=True) #resample with replacement
        y_samp = y_samp.sort_values(by='pred',ascending=False) #sort descending by predicted value
        for idx, K in enumerate(K_list):
            picked_loans = y_samp.head(K) #take top K based on predicted
            pick_k_avgs[n,idx] = picked_loans['returns'].sum() / picked_loans['weights'].sum()
        if sub_marg:
            marg_returns = y_samp['returns'].sum() / y_samp['weights'].sum()
            pick_k_avgs[n,:] = pick_k_avgs[n,:] - marg_returns
    return pick_k_avgs


def pick_K_returns_by_grade(y_pred, returns, weights, grades, K, n_boots=100):
    '''Same as pick_K_returns but picks portfolios separately from each loan
    grade class.
    INPUTS:
        y_pred: vector of model-predicted loan returns.
        returns: measured/expected net returns
        weights: vector of weights (denominator of LC NAR calc)
        grades: vector of loan grades
        K: portfolio size (scalar)
        n_boots: number of bootstrap samples
    OUTPUTS:
        pick_k_avgs: array of portfolio returns size: (n_boots, len(grades))'''
    y_comb = pd.DataFrame({'return': returns, 'pred':y_pred, 
                        'grade':grades, 'weight': weights})
    pick_k_avgs = np.zeros((n_boots, len(y_comb['grade'].unique())))
    for n in xrange(n_boots):
        y_samp = y_comb.sample(frac=1, replace=True) #resample with replacement
        picked_loans = y_samp.groupby('grade').apply(lambda g: g.sort_values(by='pred', ascending=False).head(K))
        picked_loans = picked_loans.groupby('grade')
        port = picked_loans['return'].sum() / picked_loans['weight'].sum()
        pick_k_avgs[n,:] = port.sort_index().values
    return pick_k_avgs 


def get_choice_grade_makeup(preds, grades, unique_grades, K):
    '''Calculate the distribution of loan grades among the top K picked loans.
    INPUTS:
        preds: vector of model predictions
        grades: vector of grades
        unique_grades: specify list of possible grades (in case were looking at a subset that
            doesnt have all examples.
        K: portfolio size (scaler).
    OUTPUTS:
        pandas series of counts, keyed by grade'''
    temp_df = pd.DataFrame({'pred':preds, 'grade':grades})
    temp_df.sort_values(by='pred',ascending=False,inplace=True)
    # initialize a series to ensure that all grades have a count (even if it's zero)
    base_series = pd.DataFrame({'values':np.zeros(len(unique_grades))}, index=unique_grades)
    base_series['values'] = temp_df.head(K).groupby('grade').count().sort_index()                   
    base_series.fillna(0,inplace=True)     
    return base_series.values.squeeze() #return a numpy array of counts, ordered by indices in unique_grades


# MODEL CLASSES
class zip_avg_mod(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''define model that just takes the within-zip3 avgs on the training set'''
    def __init__(self, min_counts = 100):
        self.zip_avgs = None
        self.min_counts = min_counts

    def fit(self, X, y):
        X['response'] = y
        zip_stats = X.groupby('zip3')['response'].agg(['mean','count'])       
        marg_mean = X['response'].mean() 
        self.marg_mean = marg_mean
        zip_stats.ix[zip_stats['count'] < self.min_counts, 'mean'] = marg_mean #if less than min_counts force to prior mean
        self.zip_avgs = zip_stats['mean']
        return self

    def predict(self, X):
        prediction = self.zip_avgs.ix[X['zip3']]
        prediction[prediction.isnull()] = self.marg_mean
        return prediction.values

    def transform(self, X):
        return self.predict(X)

    def score(self, X, y):
        return sklearn.metrics.r2_score(y, self.predict(X))

    
class state_avg_mod(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''define model that takes the within-state avgs on the training set'''
    def __init__(self, min_counts = 100):
        self.state_avgs = None
        self.min_counts = min_counts

    def fit(self, X, y=None):
        X['response'] = y
        self.marg_mean = X['response'].mean()
        state_stats = X.groupby('addr_state')['response'].agg(['mean','count'])
        state_stats.ix[state_stats['count'] < self.min_counts,'mean'] = self.marg_mean
        self.state_avgs = state_stats['mean']
        return self

    def predict(self, X):
        prediction = self.state_avgs[X['addr_state']]
        prediction[prediction.isnull()] = self.marg_mean
        return prediction.values

    def transform(self, X):
         return self.predict(X)

    def score(self, X, y):
        return sklearn.metrics.r2_score(y, self.predict(X))

    
class marg_avg_mod(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''define model that just takes the overall avg on the training set'''
    def __init__(self):
        self.marg_avg = None

    def fit(self, X, y=None):
        self.marg_avg = X['response'].mean()
        return self

    def predict(self, X):
        prediction = np.ones(len(X),) * self.marg_avg
        return prediction

    def score(self, X, y):
        return sklearn.metrics.r2_score(y, self.predict(X))


class rand_pick_mod(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    '''define a null model that just makes random predictions'''
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        n_pts = len(X)
        prediction = np.random.rand(n_pts)
        return prediction

    def score(self, X, y):
        return sklearn.metrics.r2_score(y, self.predict(X))


class make_dummies(sklearn.base.BaseEstimator,
                       sklearn.base.TransformerMixin):
    '''uses pandas to transform categorical variables into one-hot encoding'''
    def __init__(self, dummy_cols):
        self.dummy_cols = dummy_cols
        self.dv = DictVectorizer()

    def fit(self, X, y=None):
        self.dv.fit(X[self.dummy_cols].to_dict(orient='records'))
        return self

    def transform(self, X):
        return self.dv.transform(X[self.dummy_cols].to_dict(orient='records'))


class col_selector(sklearn.base.BaseEstimator,
                       sklearn.base.TransformerMixin):
    '''uses pandas to select subset of columns'''
    def __init__(self, col_list, col_dict):
        col_idx = []
        [col_idx.extend(col_dict[col_name]) for col_name in col_list]
        self.col_idx = col_idx

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:,self.col_idx]


class log_minmax(sklearn.base.BaseEstimator,
                       sklearn.base.TransformerMixin):
    '''Transformer that first takes log1p(X) then calls the minMaxScaler transformer'''
    def __init__(self):
        self.mm_tran = MinMaxScaler()
    
    def fit(self, X, y=None):
        self.mm_tran.fit(np.log1p(X),y)       
        return self

    def transform(self, X):
        Xt = self.mm_tran.transform(np.log1p(X))
        return Xt
        
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

#%%
