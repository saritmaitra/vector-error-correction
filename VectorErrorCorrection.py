#!/usr/bin/env python
# coding: utf-8

# ## Error correction model
# - vector error correction model (VECM) is a restricted VAR designed for use with nonstationary series that are known to be cointegrated
# - cointegration relations built into the specification so that it restricts the long-run behavior of the endogenous variables to converge to their cointegrating relationships while allowing for short-run adjustment dynamics
# -  cointegration term is known as the error correction term since the deviation from long-run equilibrium is corrected gradually through a series of partial short-run adjustments.
# - Our series have nonzero means and deterministic trends as well as stochastic trends

# In[1]:

import os
from os import listdir
from os.path import isfile, join
get_ipython().system('pip install pyforest')
import datetime, pickle, copy, warnings
import glob
from pyforest import *
plt.style.use('dark_background')


# In[2]:


df = pd.read_csv("April_data_6series.csv")
df.sample(5)


# In[3]:


df.drop(columns=['Unnamed: 0'], axis=1, inplace=True)
df.tail()


# In[4]:


df = df.set_index('timestamp')
df.index = pd.to_datetime(df.index)
df.info()


# In[5]:


X = df[:15000] # subset of data
X


# In[86]:

from pandas import DataFrame, merge, concat
plt.style.use('dark_background')
def plot_vars(train, levels, color, leveltype):
    
    """
    Displays historical trends of variables
    And see if it's sensible to just select levels instead of differences
    """
    fig, ax = plt.subplots(1, 6, figsize=(16,2.5), sharex=True)
    for col, i in dict(zip(levels, list(range(6)))).items():
        X[col].plot(ax=ax[i], legend=True, linewidth=1.0, color=color, sharex=True)     
    
    fig.suptitle(f"Historical trends of {leveltype} variables", 
                 fontsize=12, fontweight="bold")
    
plot_vars(X.values, levels = X.columns, color="red", leveltype="levels")
plt.tight_layout()


# In[6]:


# Plot
plt.style.use('dark_background')
fig, axes = plt.subplots(nrows=3, ncols=2, dpi=120, figsize=(10,6))

for i, ax in enumerate(axes.flatten()):
    dataset = X[X.columns[i]]
    ax.plot(dataset, color='red', linewidth=1)
    # Decorations
    ax.set_title(X.columns[i])
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines['top'].set_alpha(0)
    ax.tick_params(labelsize=6)
    plt.tight_layout();


# We have seen earler (VAR model) that all the series are unit root non-stationary, they may be co-integrated. This extension of unit root concept to multiple time series means that a liner combination of two or more series is stationary and hence, mean reverting. We also have seen that, VAR model is not equipped to handle this case without differencing. So, we will use here Vector Error Correction Model (VECM). We will explore here cointegratyion becase , if present and assumed to pressist, it can be leveraged for trading strategy.
# 
# However, the concept of an integrated multivariate series is complicated by the fact that, all the component series of the process may be individually integrated but the process is not jointly integrated in the sense that one or more linear combinations of the series exist that produce a new stationary series.
# 
# To simplify, a combination of two co-integrated series has a stable mean to which this linear combination reverts. A multivariate series with this characteristics is said to be cointegrated.
# 
# ### Test for cointegartion
# To verify further the relevance of the model, there is need to test for cointegration. In order to assume a long run relationship in the model despite the fact that the series are drifting apart or trending either upward or downward? having a long-run equilibrium.
# 
# Note: Cointegration test should be performed on the level form of the variables and not on their first difference. Both tests are performed as shown below.
# 
# There are two major approaches to test for cointegration:
# - Engle-Granger twio step method
# - Johansen procedure
# 
# Engle-Granger method involves regressing one series over another and then applying ADF unit root test to the regression residual. If H0 (null hypothesis) can be rejected, then we can assume that residuals are stationary and the series are cointegrated. A benefot of this approach is that, the regression cofficient represents  the multiplier that renders the combinationn stationary i.e. mean reverting.
# 
# Johansen procedure tests the restrictions imposed by cointegration on a VAR model. 
# 
# ### Johansen co-integration on level data:
# Johansen test assesses the validity of a cointegrating relationship, using a maximum likelihood estimates (MLE) approach.
# 
# Two types of Johansen’s test:
# 
# - one uses trace (from linear algebra),
# - the other a maximum eigenvalue approach (an eigenvalue is a special scalar; when we multiply a matrix by a vector and get the same vector as an answer, along with a new scalar, the scalar is called an eigenvalue).
# - Both forms of the test will determine if cointegration is present. The hypothesis is stated as:
# 
# Null hypothesis (H0) for both: no cointegrating equations.
# 
# The difference is in the alternate hypothesis (H1): the trace test alternate hypothesis is simply that the number of cointegrating relationships is at least one (shown by the number of linear combinations).
# 
# Rejecting the null hypothesis is basically stating there is only one combination of the non-stationary variables that gives a stationary process.

# In[6]:


from statsmodels.tsa.vector_ar.vecm import VECM, select_order
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar.vecm import select_coint_rank
from statsmodels.tsa.vector_ar.vecm import CointRankResults


# In[7]:


nobs = 15
train_ecm, test_ecm = X[0:-nobs], X[-nobs:]

# Check size
print(train_ecm.shape)  
print(test_ecm.shape)


# In[8]:


# VECM model fitting
from statsmodels.tsa.vector_ar import vecm
# pass "1min" frequency
train_ecm.index = pd.DatetimeIndex(train_ecm.index).to_period('1min')
model = vecm.select_order(train_ecm, maxlags=8)
print(model.summary())


# In[10]:


pd.options.display.float_format = "{:.2f}".format
"""definition of det_orderint:
-1 - no deterministic terms; 0 - constant term; 1 - linear trend"""
pd.options.display.float_format = "{:.2f}".format
model = coint_johansen(endog = train_ecm, det_order = 1, k_ar_diff = 3)
print('Eigen statistic:')
print(model.eig) 
print()
print('Critical values:')
d = DataFrame(model.cvt)
d.rename(columns = {0:'90%', 1: '95%', 2:'99%'}, inplace=True)
print(d); print()
print('Trace statistic:')
print(DataFrame(model.lr1)) 


# Here, we see that trace statistics (76.86033757  44.90556245  24.43779121  11.11437692) are < critical values @95% (79.24, 55.24, 35.01, 18.39).
# 
# Interpreting Johansen Cointegration Test Results
# - output releases two statistics, Trace Statistic and Max-Eigen Statistic
# - Rejection criteria is at 0.05 level
# - Reject (H0) null hypothesis if the p-value <= 0.05 means there is no cointegrating equations.

# ### Eigen and Trace statistic (Johansent co-integration)

# In[10]:


pd.options.display.float_format = "{:.2f}".format
model = coint_johansen(endog = df, det_order = 0, k_ar_diff = 8)

print("\nnormalized eigenvector 0\n", model.evec[:,0] / model.evec[:,0][0])
print("\nnormalized eigenvector 1\n", model.evec[:,1] / model.evec[:,1][0])
print("\ntest statistics\n", model.lr1[0], model.lr1[1]) # Trace statistic
print("\ncritical values\n", model.cvt[0], model.cvt[1]) # Maximum eigenvalue statistic
print("\neig\n", model.eig)  # Eigenvalues of VECM coefficient matrix
print("\nevec\n", model.evec)  # Eigenvectors of VECM coefficient matrix
print("\nlr1\n", model.lr1)  # dim = (n,) Trace statistic
print("\nlr2\n", model.lr2) 
print("\ncvt\n", model.cvt) # Critical values (90%, 95%, 99%) of trace statistic
print("\ncvm\n", model.cvm) # Critical values (90%, 95%, 99%) of maximum eigenvalue statistic
print("\nind\n", model.ind) # Order of eigenvalues


# In[19]:


# simplifying the above test statotistic in tabular format as below
# trace test

out = coint_johansen(endog = df, det_order = 1, k_ar_diff = 8)
d = {'0.90':0, '0.95':1, '0.99':2}
traces = out.lr1
cvts = out.cvt[:, d['0.95']]
def adjust(val, length= 6): return str(val).ljust(length)
print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
for col, trace, cvt in zip(df.columns, traces, cvts):
    print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)


# We can see that in 4 equations, we can safely reject null hypotheses and alternate hypotheses of cointegration exists are accepted.

# In[21]:


# eigen test

out = coint_johansen(endog = df, det_order = 1, k_ar_diff = 8)
d = {'0.90':0, '0.95':1, '0.99':2}
eigen = out.eig
cvts = out.cvt[:, d['0.95']]
def adjust(val, length= 6): return str(val).ljust(length)
print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
for col, eig, cvt in zip(df.columns, eigen, cvts):
    print(adjust(col), ':: ', adjust(round(eig,2), 9), ">", adjust(cvt, 8), ' =>  ' , eig > cvt)


# - Implies that the series in question are related and therefore can be combined in a linear fashion.
# - That is, even if there are shocks in the short run, which may affect movement in the individual series, they would converge with time (in the long run).
# - Estimate both long-run and short-run models.
# - moving together in such a way that their linear combination results in a stationary time series
# - sharing an underlying common stochastic trend.
# 
# Here we can safely use of vector autoregressive (VAR) model and vector error correction model (VECM) for analysis. Cointegration implies that time series will be connecting through an error correction model. The error correction model is important in time series analysis because it allows us to better understand long-run dynamics.

# ### Engle Granger cointegration test:
# 
# This uses the augmented Engle-Granger two-step cointegration test. Constant or trend is included in 1st stage regression, i.e. in cointegrating equation.
# 
# The output returns
# 
# - t-statistic of unit-root test on residuals.
# - asymptotic p-value based on MacKinnon (1994).
# - critical values for the test statistic at the 1 %, 5 %, and 10 % levels 
# 
# The Null hypothesis is that there is no cointegration, the alternative hypothesis is that there is cointegrating relationship. If the pvalue is small, below a critical size, then we can reject the hypothesis that there is no cointegrating relationship.

# In[13]:


import statsmodels.tsa.stattools as ts 

print('EURUSD <-> GBPUSD')
print(ts.coint(df.eur_usd, df.gbp_usd))
print('_________________')
print('EURUSD <-> USDJPY')
print(ts.coint(df.eur_usd, df.usd_jpy))
print('_________________')
print('EURUSD <-> GC')
print(ts.coint(df.eur_usd, df.gc))
print('_________________')
print('EURUSD <-> NQ')
print(ts.coint(df.eur_usd, df.nq))
print('_________________')
print('EURUSD <-> ES')
print(ts.coint(df.eur_usd, df.es))
print('_________________')


# ###### p-value in all cases of 0.98/0.72/0.98/0.90/0.78 > 0.05 means that the null hypothesis (H0) of no cointegration cannot be rejected at the 5% level of significance. 

# In[14]:


print('USDJPY <-> GBPUSD')
print(ts.coint(df.usd_jpy, df.gbp_usd))
print('_________________')
print('USDJPY <-> EURUSD')
print(ts.coint(df.usd_jpy, df.eur_usd))
print('_________________')
print('USDJPY <-> GC')
print(ts.coint(df.usd_jpy, df.gc))
print('_________________')
print('USDJPY <-> NQ')
print(ts.coint(df.usd_jpy, df.nq))
print('_________________')
print('USDJPY <-> ES')
print(ts.coint(df.usd_jpy, df.es))
print('_________________')


# ##### p-values in all 5 cases of 0.21/0.18/0.29/0.10 > 0.05 means that the null hypothesis (H0) of no cointegration cannot be rejected at the 5% level of significance. In one case (USDJPY <-> NQ, we find p (0.01) < 0.05 means that H1 is accepted means cointegrating relationships in these series.

# In[15]:


print('GBPUSD <-> EURUSD')
print(ts.coint(df.gbp_usd, df.eur_usd))
print('_________________')
print('GBPUSD <-> USDJPY')
print(ts.coint(df.gbp_usd, df.usd_jpy))
print('_________________')
print('GBPUSD <-> GC')
print(ts.coint(df.gbp_usd, df.gc))
print('_________________')
print('GBPUSD <-> NQ')
print(ts.coint(df.gbp_usd, df.nq))
print('_________________')
print('GBPUSD <-> ES')
print(ts.coint(df.gbp_usd, df.es))
print('_________________')


# ##### Here too, in one case (GBPUSD<-> GC, we find p-value (0.03) < 0.05, which means that H1 is accepted and cointegrating relationships in these series.

# In[16]:


print('NQ <-> EURUSD')
print(ts.coint(df.nq, df.eur_usd))
print('_________________')
print('NQ <-> USDJPY')
print(ts.coint(df.nq, df.usd_jpy))
print('_________________')
print('NQ <-> GC')
print(ts.coint(df.nq, df.gc))
print('_________________')
print('NQ <-> GBPUSD')
print(ts.coint(df.nq, df.gbp_usd))
print('_________________')
print('NQ <-> ES')
print(ts.coint(df.nq, df.es))
print('_________________')


# In[60]:


nobs = 15
train_ecm, test_ecm = X[0:-nobs], X[-nobs:]

# Check size
print(train_ecm.shape)  
print(test_ecm.shape)


# # Linear model assumptions:
# - model in linear in parameters
# - regressors are fixed, non-stochastic
# - expected value of error term is 0
# - homoscedastic or constant variance of errors
# - no auto-correlation
# - no multi-colinearity; no perfect linear relationships among Xs
# - no specification bias
# 
# Gaus-Markov theorem:
# - given all above assumptions
#   - estimators are liner functions of the dependent variable (y)
#   - estimators are unbiased, in repeated samples, the estimators approach their true value
#   - in the class of linear estimators, OLS estimators have minimum variance; i.e they are efficient, or the best estimators.

# In[61]:


# VECM model fitting
from statsmodels.tsa.vector_ar import vecm
# pass "1min" frequency
train_ecm.index = pd.DatetimeIndex(train_ecm.index).to_period('1min')
model = vecm.select_order(train_ecm, maxlags=8)
print(model.summary())


# ## Error correction model:
# 
# - shows the long-run equilibrium relationships of variables.
# - includes a short-run dynamic adjustment mechanism that describes how variables adjust when they are out of equilibrium.
# - uses adjustment coefficients to measure the forces that push the relationship towards long-run equilibrium.
# 
# ### Cointegration rank of a VECM

# In[19]:


from statsmodels.tsa.vector_ar.vecm import select_coint_rank


# In[68]:


from statsmodels.tsa.vector_ar.vecm import select_coint_rank
rank1 = select_coint_rank(train_ecm, det_order = 1, k_ar_diff = 3,
                                   method = 'trace', signif=0.01)
print(rank1.summary())


# - first column in the table shows the rank which is the number of cointegrating relationships for the dataset, while the second reports the number of equations in total. 
# - λtrace statistics in the third column, together with the corresponding critical values. 
# - first row of the table tests the null hypothesis of at most one cointegrating vector, against the alternative hypothesis that the number of cointegrating equations is strictly larger than the number assumed under the null hypothesis, i.e., larger than one. 
# - test statistic of 254.9 considerably exceeds the critical value (117.0) and so the null of at most one cointegrating vector is rejected. 
# - test statistic (89.92) also exceeds the critical value (87.77), so the null of at most two cointegrating vectors is rejected at the 1% level
# - test statistic (48.59) do not exceeds the critical value (62.52), so the null of at most three cointegrating vectors cannot be rejected at the 1% level
# 
# Below test statistic on maximum eigen value:
# 
# Maximum-eigenvalue statistic assumes a given number of r cointegrating relations under the null hypothesis and tests this against the alternative that there are r + 1 cointegrating equations.

# In[67]:


rank2 = select_coint_rank(train_ecm, det_order = 1, k_ar_diff = 3, 
                              method = 'maxeig', signif=0.01)

print(rank2.summary())


# In[69]:


from statsmodels.tsa.vector_ar.vecm import VECM
# VECM
vecm = VECM(train_ecm, k_ar_diff=3, coint_rank = 3, deterministic='ci')
"""estimates the VECM on the prices with 3 lags, 3 cointegrating relationship, and 
a constant within the cointegration relationship"""
vecm_fit = vecm.fit()
print(vecm_fit.summary())


# 
# ## Checking residual auto-correlation

# In[70]:


from statsmodels.stats.stattools import durbin_watson
out = durbin_watson(vecm_fit.resid)
for col, val in zip(train_ecm.columns, out):
    print((col), ':', round(val, 2))


# ## Impact of changes in one variable on the others at different horizons:
# 
# ### Impulse Response Analysis
# IRAs the estimated responses to a unit impulse in one of the variables. They are computed in practice using the MA(\infty) representation of the VAR(p) process:

# In[87]:


# stores and prints the impulse response functions
plt.style.use('ggplot')
irf = vecm_fit.irf(15)
irf.plot(orth = False)
plt.show()


# #### Objectives: 
# - how variable x effects variable y
# - when the maximum impact is experienced
# - how long the effect lasts
# 
# One standard deviation shock/increase to x causes significant increases (decreases) in y (impulse variable) for m periods (determined by the length of period for which the SE bands are above 0 or below 0 in case of decrease) after which the effect dissipates. The increase peaks in period k. 
# 
# Each of the IRF graphs tells us how a shock to one variable affects another (or the same) variable.

# In[92]:


X.tail(1)


# In[89]:


plt.style.use('ggplot')
irf.plot(impulse='eurusd')
plt.show()


# In[90]:


plt.style.use('ggplot')
irf.plot(impulse='gbpusd')
plt.show()


# In[91]:


plt.style.use('ggplot')
irf.plot(impulse='usdjpy', orth = True)
plt.show()


# In[28]:


plt.style.use('ggplot')
irf.plot(impulse='es')
plt.show()


# In[29]:


plt.style.use('ggplot')
irf.plot(impulse='gc')
plt.show()


# In[30]:


plt.style.use('ggplot')
irf.plot(impulse='nq')
plt.show()


# In[32]:


eg = vecm_fit.test_granger_causality(caused="eur_usd", causing="gbp_usd")
eg.summary()


# In[33]:


je = vecm_fit.test_granger_causality(caused="usd_jpy", causing="eur_usd")
je.summary()


# In[34]:


jnq = vecm_fit.test_granger_causality(caused="usd_jpy", causing="nq")
jnq.summary()


# In[35]:


jes = vecm_fit.test_granger_causality(caused="usd_jpy", causing="es")
jes.summary()


# In[71]:


pd.options.display.float_format = "{:.2f}".format
forecast, lower, upper = vecm_fit.predict(nobs, 0.05)
print("lower bounds of confidence intervals:")
print(DataFrame(lower.round(2)))
print("\npoint forecasts:")
print(DataFrame(forecast.round(2)))
print("\nupper bounds of confidence intervals:")
print(DataFrame(upper.round(2)))


# In[72]:


pd.options.display.float_format = "{:.2f}".format
forecast = DataFrame(forecast, index= test_ecm.index, columns= test_ecm.columns)
forecast.rename(columns = {'eurusd':'eurusd_pred', 'gbpusd':'gbpusd_pred', 'usdjpy':'usdjpy_pred',
                    'gc':'gc_pred', 'nq':'nq_pred', 'es':'es_pred'}, inplace = True)
forecast


# In[73]:


combine = concat([test_ecm, forecast], axis=1)
pred = combine[['eurusd', 'eurusd_pred', 'gbpusd', 'gbpusd_pred', 'usdjpy', 
                   'usdjpy_pred', 'gc', 'gc_pred', 'nq', 'nq_pred', 'es', 'es_pred']]
def highlight_cols(s):
    color = 'yellow'
    return 'background-color: %s' % color

pred.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['eurusd_pred', 'gbpusd_pred', 'usdjpy_pred',
                                                               'gc_pred', 'nq_pred', 'es_pred']])


# In[81]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

# score eur_usd
mae = mean_absolute_error(pred.eurusd, pred['eurusd_pred'])
mse = mean_squared_error(pred.eurusd, pred.eurusd_pred)
rmse = np.sqrt(mse)
sum = DataFrame(index = ['Mean Absolute Error', 'Mean squared error', 'Root mean squared error'])
sum['Accuracy metrics :    EURUSD'] = [mae, mse, rmse]

# score gbp_usd
mae = mean_absolute_error(pred.gbpusd, pred['gbpusd_pred'])
mse = mean_squared_error(pred.gbpusd, pred.gbpusd_pred)
rmse = np.sqrt(mse)
sum['GBPUSD'] = [mae, mse, rmse]

# score usd_jpy
mae = mean_absolute_error(pred.usdjpy, pred['usdjpy_pred'])
mse = mean_squared_error(pred.usdjpy, pred.usdjpy_pred)
rmse = np.sqrt(mse)
sum['USDJPY'] = [mae, mse, rmse]

# score nq
mae = mean_absolute_error(pred.nq, pred['nq_pred'])
mse = mean_squared_error(pred.nq, pred.nq_pred)
rmse = np.sqrt(mse)
sum['NQ'] = [mae, mse, rmse]

# score usd_jpy
mae = mean_absolute_error(pred.es, pred['es_pred'])
mse = mean_squared_error(pred.es, pred.es_pred)
rmse = np.sqrt(mse)
sum['ES'] = [mae, mse, rmse]

# score usd_jpy
mae = mean_absolute_error(pred.gc, pred['gc_pred'])
mse = mean_squared_error(pred.gc, pred.gc_pred)
rmse = np.sqrt(mse)
sum['GC'] = [mae, mse, rmse]
sum


# In[82]:


train_ecm, test_ecm = X[0:-nobs], X[-nobs:]

plt.style.use('dark_background')
get_ipython().run_line_magic('matplotlib', 'inline')
for i in ['eurusd','gbpusd','usdjpy','gc','nq','es']:
    plt.rcParams["figure.figsize"] = [15,6]
    plt.plot(train_ecm[str(i)], label='Train '+str(i))
    plt.plot(test_ecm[str(i)], label='Test '+str(i))
    plt.plot(forecast[str(i) + '_pred'], label='Predicted '+ str(i))
    plt.legend(loc='best')
    plt.show()


# In[83]:


plt.style.use('seaborn')
fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    forecast[col+'_pred'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    test_ecm[col].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# In[ ]:





# In[55]:


from statsmodels.tsa.vector_ar.vecm import VECM

# pass "1min" frequency
train_ecm.index = pd.DatetimeIndex(train_ecm.index).to_period('1min')

# VECM
vecm_1 = VECM(train_ecm, k_ar_diff=2, coint_rank = 4, deterministic='ci')

"""estimates the VECM on the prices with 8 lags, 2 cointegrating relationship, and 
a constant within the cointegration relationship"""

vecm_fit_1 = vecm_1.fit()

pd.options.display.float_format = "{:.2f}".format
fcast_1, lower, upper = vecm_fit_1.predict(nobs, 0.05)
print("lower bounds of confidence intervals:")
print(lower.round(2))
print("\npoint forecasts:")
print(fcast_1.round(2))
print("\nupper bounds of confidence intervals:")
print(upper.round(2))


# In[59]:


pd.options.display.float_format = "{:.2f}".format
fcast_1 = DataFrame(fcast_1, index= test_ecm.index, columns= test_ecm.columns)
fcast_1.rename(columns = {'eur_usd':'eur_usd_pred', 'gbp_usd':'gbp_usd_pred', 'usd_jpy':'usd_jpy_pred',
                    'gc':'gc_pred', 'nq':'nq_pred', 'es':'es_pred'}, inplace = True)
fcast_1


# In[60]:


join = concat([test_ecm, fcast_1], axis=1)
pred_1 = combine[['eur_usd', 'eur_usd_pred', 'gbp_usd', 'gbp_usd_pred', 'usd_jpy', 
                   'usd_jpy_pred', 'gc', 'gc_pred', 'nq', 'nq_pred', 'es', 'es_pred']]
def highlight_cols(s):
    color = 'yellow'
    return 'background-color: %s' % color

pred_1.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['eur_usd_pred', 'gbp_usd_pred', 'usd_jpy_pred',
                                                               'gc_pred', 'nq_pred', 'es_pred']])


# In[74]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

# score eur_usd
r_squared = r2_score(pred_1.eur_usd, pred_1.eur_usd_pred)
mae = mean_absolute_error(pred_1.eur_usd, pred_1['eur_usd_pred'])
mse = mean_squared_error(pred_1.eur_usd, pred_1.eur_usd_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_1.eur_usd, pred_1.eur_usd_pred)
sum_df = DataFrame(index = ['R-squared', 'Mean Absolute Error', 'Mean squared error', 'Root mean squared error',
                            'Mean squared log Error'])
sum_df['Accuracy metrics :    EURUSD'] = [r_squared, mae, mse, rmse, msle]

# score gbp_usd
r_squared = r2_score(pred_1.gbp_usd, pred_1.gbp_usd_pred)
mae = mean_absolute_error(pred_1.gbp_usd, pred_1['gbp_usd_pred'])
mse = mean_squared_error(pred_1.gbp_usd, pred_1.gbp_usd_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_1.gbp_usd, pred_1.gbp_usd_pred)
sum_df['GBPUSD'] = [r_squared, mae, mse, rmse, msle]

# score usd_jpy
r_squared = r2_score(pred_1.usd_jpy, pred_1.usd_jpy_pred)
mae = mean_absolute_error(pred_1.usd_jpy, pred_1['usd_jpy_pred'])
mse = mean_squared_error(pred_1.usd_jpy, pred_1.usd_jpy_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_1.usd_jpy, pred_1.usd_jpy_pred)
sum_df['USDJPY'] = [r_squared, mae, mse, rmse, msle]

# score nq
r_squared = r2_score(pred_1.nq, pred_1.nq_pred)
mae = mean_absolute_error(pred_1.nq, pred_1['nq_pred'])
mse = mean_squared_error(pred_1.nq, pred_1.nq_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_1.nq, pred_1.nq_pred)
sum_df['NQ'] = [r_squared, mae, mse, rmse, msle]

# score usd_jpy
r_squared = r2_score(pred_1.es, pred_1.es_pred)
mae = mean_absolute_error(pred_1.es, pred_1['es_pred'])
mse = mean_squared_error(pred_1.es, pred_1.es_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_1.es, pred_1.es_pred)
sum_df['ES'] = [r_squared, mae, mse, rmse, msle]

# score usd_jpy
r_squared = r2_score(pred_1.gc, pred_1.gc_pred)
mae = mean_absolute_error(pred_1.gc, pred_1['gc_pred'])
mse = mean_squared_error(pred_1.gc, pred_1.gc_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_1.gc, pred_1.gc_pred)
sum_df['GC'] = [r_squared, mae, mse, rmse, msle]
sum_df


# In[78]:


# VECM
vecm_2 = VECM(train_ecm, k_ar_diff=8, coint_rank = 2, deterministic='cili')

"""estimates the VECM on the prices with 8 lags, 2 cointegrating relationship, and 
a constant within the cointegration relationship"""

vecm_fit_2 = vecm_2.fit()

pd.options.display.float_format = "{:.2f}".format
fcast_2, lower, upper = vecm_fit_2.predict(nobs, 0.05)
print("lower bounds of confidence intervals:")
print(lower.round(2))
print("\npoint forecasts:")
print(fcast_2.round(2))
print("\nupper bounds of confidence intervals:")
print(upper.round(2))


# In[79]:


pd.options.display.float_format = "{:.2f}".format
fcast_2 = DataFrame(fcast_2, index= test_ecm.index, columns= test_ecm.columns)
fcast_2.rename(columns = {'eur_usd':'eur_usd_pred', 'gbp_usd':'gbp_usd_pred', 'usd_jpy':'usd_jpy_pred',
                    'gc':'gc_pred', 'nq':'nq_pred', 'es':'es_pred'}, inplace = True)
join = concat([test_ecm, fcast_2], axis=1)
pred_2 = combine[['eur_usd', 'eur_usd_pred', 'gbp_usd', 'gbp_usd_pred', 'usd_jpy', 
                   'usd_jpy_pred', 'gc', 'gc_pred', 'nq', 'nq_pred', 'es', 'es_pred']]
def highlight_cols(s):
    color = 'yellow'
    return 'background-color: %s' % color

pred_2.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['eur_usd_pred', 'gbp_usd_pred', 'usd_jpy_pred',
                                                               'gc_pred', 'nq_pred', 'es_pred']])


# In[80]:


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error, mean_squared_log_error

# score eur_usd
r_squared = r2_score(pred_2.eur_usd, pred_2.eur_usd_pred)
mae = mean_absolute_error(pred_2.eur_usd, pred_2['eur_usd_pred'])
mse = mean_squared_error(pred_2.eur_usd, pred_2.eur_usd_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_2.eur_usd, pred_2.eur_usd_pred)
sum_df = DataFrame(index = ['R-squared', 'Mean Absolute Error', 'Mean squared error', 'Root mean squared error',
                            'Mean squared log Error'])
sum_df['Accuracy metrics :    EURUSD'] = [r_squared, mae, mse, rmse, msle]

# score gbp_usd
r_squared = r2_score(pred_2.gbp_usd, pred_2.gbp_usd_pred)
mae = mean_absolute_error(pred_2.gbp_usd, pred_2['gbp_usd_pred'])
mse = mean_squared_error(pred_2.gbp_usd, pred_2.gbp_usd_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_2.gbp_usd, pred_2.gbp_usd_pred)
sum_df['GBPUSD'] = [r_squared, mae, mse, rmse, msle]

# score usd_jpy
r_squared = r2_score(pred_2.usd_jpy, pred_2.usd_jpy_pred)
mae = mean_absolute_error(pred_2.usd_jpy, pred_2['usd_jpy_pred'])
mse = mean_squared_error(pred_2.usd_jpy, pred_2.usd_jpy_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_2.usd_jpy, pred_2.usd_jpy_pred)
sum_df['USDJPY'] = [r_squared, mae, mse, rmse, msle]

# score nq
r_squared = r2_score(pred_2.nq, pred_2.nq_pred)
mae = mean_absolute_error(pred_2.nq, pred_2['nq_pred'])
mse = mean_squared_error(pred_2.nq, pred_2.nq_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_2.nq, pred_2.nq_pred)
sum_df['NQ'] = [r_squared, mae, mse, rmse, msle]

# score usd_jpy
r_squared = r2_score(pred_2.es, pred_2.es_pred)
mae = mean_absolute_error(pred_2.es, pred_2['es_pred'])
mse = mean_squared_error(pred_2.es, pred_2.es_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_2.es, pred_2.es_pred)
sum_df['ES'] = [r_squared, mae, mse, rmse, msle]

# score usd_jpy
r_squared = r2_score(pred_2.gc, pred_2.gc_pred)
mae = mean_absolute_error(pred_2.gc, pred_2['gc_pred'])
mse = mean_squared_error(pred_2.gc, pred_2.gc_pred)
rmse = np.sqrt(mse)
#mape = mean_absolute_percentage_error(pred_1.eur_usd - pred_1.eur_usd_pred)
msle = mean_squared_log_error(pred_2.gc, pred_2.gc_pred)
sum_df['GC'] = [r_squared, mae, mse, rmse, msle]
sum_df


# In[84]:


# VECM
vecm_3 = VECM(train_ecm, k_ar_diff=4, coint_rank = 4)

"""estimates the VECM on the prices with 8 lags, 2 cointegrating relationship, and 
a constant within the cointegration relationship"""

vecm_fit_3 = vecm_3.fit()

pd.options.display.float_format = "{:.2f}".format
fcast_3, lower, upper = vecm_fit_3.predict(nobs, 0.05)
#print("lower bounds of confidence intervals:")
#print(lower.round(2))
#print("\npoint forecasts:")
#print(fcast_3.round(2))
#print("\nupper bounds of confidence intervals:")
#print(upper.round(2))

pd.options.display.float_format = "{:.2f}".format
fcast_3 = DataFrame(fcast_3, index= test_ecm.index, columns= test_ecm.columns)
fcast_3.rename(columns = {'eur_usd':'eur_usd_pred', 'gbp_usd':'gbp_usd_pred', 'usd_jpy':'usd_jpy_pred',
                    'gc':'gc_pred', 'nq':'nq_pred', 'es':'es_pred'}, inplace = True)
join = concat([test_ecm, fcast_3], axis=1)
pred_3 = combine[['eur_usd', 'eur_usd_pred', 'gbp_usd', 'gbp_usd_pred', 'usd_jpy', 
                   'usd_jpy_pred', 'gc', 'gc_pred', 'nq', 'nq_pred', 'es', 'es_pred']]
def highlight_cols(s):
    color = 'yellow'
    return 'background-color: %s' % color

pred_3.style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['eur_usd_pred', 'gbp_usd_pred', 'usd_jpy_pred',
                                                               'gc_pred', 'nq_pred', 'es_pred']])


# In[85]:


plt.style.use('seaborn')
fig, axes = plt.subplots(nrows=int(len(df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
for i, (col,ax) in enumerate(zip(df.columns, axes.flatten())):
    pred_3[col+'_pred'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
    test_ecm[col].plot(legend=True, ax=ax);
    ax.set_title(col + ": Forecast vs Actuals")
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.spines["top"].set_alpha(0)
    ax.tick_params(labelsize=6)

plt.tight_layout();


# In[ ]:




