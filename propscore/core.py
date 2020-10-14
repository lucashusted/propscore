#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from statsmodels.api import Logit
from scipy.stats import ttest_ind as ttest
from numpy import linspace
from pandas import Series
import warnings

class PropensityScore:
    """
    Parameters
    ----------
    outcome : str
        This should be the name of the binary variable to predict.
    test_vars : list
        A list of the variables to test.
    df : DataFrame
        The pandas DataFrame that contains all of the data.
    init_vars : str or list, optional
        Variables to always have included in the propensity score. The default is None.
    add_cons : Boolean, optional
        Select this to add a constant to model. The default is True.
    disp : Boolean, optional
        Display the final model including dropped variables. The default is True.
    cutoff_ord1 : Numeric, optional
        The log gain cutoff for first order covariates. The default is 1.
    cutoff_ord2 : Numeric, optional
        The log gain cutoff for second order covariates. The default is 2.71.
    t_strata : Numeric, optional
        The cutoff for the t-statistic for the calculated strata. The default is 1.
    n_min_strata : Int or 'auto'
        The minimum number of units in each strata. The default is 'auto' in which case
        the number is the number of covariates tested in the propensity score (just linear ones)
        plus 3. If not auto, the input needs to be an integer.

    Raises
    ------
    ValueError
        If variables are improperly defined, this prints out warnings.

    Returns
    -------
    self.data : DataFrame
        This includes a new frame of just the outcome and potential covariates.
    self.dropped_vars : list
        The variables that did not make the cut for singularity reasons.
    self.model : sm.Logit.fit() model
        This is the raw model on the final set of variables from Statsmodels
    self.propscore : Series
        This is the propensity score as calculated by self.model.fittedvalues.
        This may not match dimension of data due to dropped missing values,
        but index will align properly.
    self.strata : Series
        The calculated strata. Missing propensity scores and values outside of
        min of treated group or max of control group are coded as NaN.
    self.logodds : Series
        The linearized propensity score. Will be the same dimension as propscore.
    self.test_vars_ord2: list
        The full list of tested second order variables for reference.
    self.trim_range : tuple
        The result of calculating the optimal trim min and max propensity score values.
    self.in_trim : Series (True/False)
        An array where True means that the propensity score falls within the 
        trim min/max range.
    """
    def __init__(self, outcome, test_vars, df, init_vars=None, add_cons=True, disp=True,
                 cutoff_ord1 = 1, cutoff_ord2 = 2.71, t_strata = 1, n_min_strata='auto'):

        # double checking some inputs
        if type(outcome)!=str:
            raise ValueError('y must be a string variable name in the DataFrame.')
        if type(test_vars)!=list:
            raise ValueError('X must be a list of covariates to test.')

        self.outcome = outcome
        self.test_vars = test_vars
        self.add_cons = add_cons
        self.init_vars = init_vars

        if init_vars and type(init_vars)==str:
            covs = [init_vars] + test_vars
        elif init_vars and type(init_vars) == list:
            covs = init_vars + test_vars
        else:
            covs = test_vars

        if n_min_strata == 'auto':
            n_min_strata = len(covs)+3

        if 'propscore' in covs + [outcome] or 'logodds' in covs + [outcome]:
            raise ValueError('You cannot have variables labeled "propscore" or "logodds"')


        data = df[[outcome]+covs].copy()

        ord2_vars = []
        dropped_vars = []
        # looping through covariates
        for idx,cc in enumerate(covs):
            # first a gut check to make sure all the variables aren't singular
            if len(data[cc].dropna().unique()) == 1:
                raise ValueError('{} only takes on one value'.format(cc))

            # for all variables generate the interaction terms
            if idx<len(covs):
                for jj in covs[idx+1:]:
                     testvar = data[cc]*data[jj]
                     if (not testvar.equals(data[cc]) and
                    not testvar.equals(data[jj]) and
                    len(testvar.dropna().unique()) > 1):
                         data.loc[:,'X'.join([cc,jj])] = testvar
                         ord2_vars.append('X'.join([cc,jj]))
                     else:
                         dropped_vars.append('X'.join([cc,jj]))

            # for continuous variables, generate squared term
            if not data[cc].equals(data[cc]**2):
                data.loc[:,'{}_sq'.format(cc)] = data[cc]**2
                ord2_vars.append('{}_sq'.format(cc))
            else:
                dropped_vars.append('{}_sq'.format(cc))

        if add_cons:
            data.loc[:,'_cons'] = 1

        self.data = data
        self.dropped_vars = dropped_vars
        self.test_vars_ord2 = ord2_vars

        # =====================================================================
        # Actually calculating propensity score
        # =====================================================================
        linear = self.model_from_group(self.test_vars,cutoff = cutoff_ord1,
                                            init_vars = self.init_vars)

        squared = self.model_from_group(ord2_vars,cutoff = cutoff_ord2,
                                             init_vars = linear)

        if add_cons:
            self.model = Logit(self.data[self.outcome],
                        self.data[squared+['_cons']],missing='drop').fit(disp=False)
        else:
            self.model = Logit(self.data[self.outcome],
                        self.data[squared],missing='drop').fit(disp=False)

        self.logodds = self.model.fittedvalues.rename('logodds')
        self.propscore = Series(self.model.predict(),index=self.logodds.index,name='propscore')
        self.trim_range = self.calc_trim(self.propscore)
        self.in_trim = (self.propscore.ge(self.trim_range[0]) & 
                        self.propscore.le(self.trim_range[1])).rename('in_trim')
        self.strata = self.stratify(self.data[self.outcome],self.logodds,
                                    t_max=t_strata, n_min = n_min_strata)

        if disp:
            print(self.model.summary())
            print('The following vars were infeasible: {}'.format(', '.join(self.dropped_vars)))
            print('Stratification produced {} strata'.format(len(self.strata.dropna().unique())))

    def best_in_group(self, newvars, basevars=None):
        ''' Get the best variable for score among a set of new variables '''

        if not basevars and self.add_cons:
            basevars = ['_cons']
        elif basevars and self.add_cons:
            basevars = basevars + ['_cons']
        elif not basevars and not self.add_cons:
            raise ValueError('Must specify at least one covariate for baseline model')

        origmod = Logit(self.data[self.outcome],
                        self.data[basevars],missing='drop').fit(disp=False)
        list_llf = []
        for cc in newvars:
            try:
                newmod = Logit(self.data[self.outcome],
                               self.data[basevars+[cc]],missing='drop').fit(disp=False)
                if origmod.nobs/origmod.nobs<.95:
                    warnings.warn('Using {} causes more than 5% '\
                                  'of the sample to be dropped'.format(cc))
                list_llf.append(newmod.llf)
            except:
                if cc not in self.dropped_vars:
                    self.dropped_vars.append(cc)
                list_llf.append(origmod.llf)
        idx = list_llf.index(max(list_llf))

        return newvars[idx], 2*(list_llf[idx]-origmod.llf)

    def model_from_group(self,test_vars,cutoff,init_vars=None):
        ''' Iterate through a list over and over until no more contribution '''
        remaining = test_vars.copy()

        if init_vars and type(init_vars)==str:
            final = [init_vars].copy()
            init_vars = [init_vars]
        elif init_vars and type(init_vars) == list:
            final = init_vars.copy()
        else:
            final = []

        while len(remaining)>0:
            temp, gain_add = self.best_in_group(remaining,basevars=final)
            if gain_add > cutoff:
                final.append(temp)
                remaining.remove(temp)
            else:
                break

        return final

    # we will define a static method so that we can call this on any generic series
    @staticmethod
    def stratify(outcome, logodds, n_min, t_max = 1):
        """
    Calculate strata from a given outcome variable and log-odds. Specify the cutoff
    for the t-statistic in t_max, or the minimum number of observations for
    each strata in n_min.
    Parameters
    ----------
    outcome : Series
        Binary variable denoting treatment outcome
    logodds : Series
        The calculated log-odds for that (transformation of propensity score).
    t_max : Float
        The maximum t-statistic value acceptable in a strata before splitting.
        Default is 1.
    n_min : Int
        The minimum number of observations per strata.

    Returns
    -------
    strata : Series
        The calculated strata. Missing propensity scores and values outside of
        min of treated group or max of control group are coded as NaN.
        """

        if type(outcome)!=Series or type(logodds)!=Series:
            raise ValueError('Expecting pandas series as inputs')

        # helper function to facilitate indexing
        def above_med(x):
            return (x>=x.median()).astype(int)

        outcome = outcome.rename('outcome').to_frame()
        df = outcome.join(logodds)
        minmax = df.groupby('outcome')['logodds'].agg(['max','min'])
        df = df.loc[df.logodds.ge(minmax.loc[1,'min']) &
                            df.logodds.le(minmax.loc[0,'max']) &
                            df.logodds.notnull()]

        # initialize the strata, potential blocks, and the change while loop
        df.loc[:,'strata'] = 0
        df.loc[:,'block'] = 0
        change = True

        while change == True:
            # get the medians of the strata
            df.loc[:,'medgrp'] = df.groupby('strata')['logodds'].apply(above_med)
            for ii in df.strata.unique():
                # simplify the notation
                sub = df.loc[df.strata.eq(ii),:].copy()

                # calculate t-stat and a grouper with number of groups
                t_test = ttest(sub.loc[sub.outcome.eq(1),'logodds'],
                               sub.loc[sub.outcome.eq(0),'logodds'],
                               nan_policy='omit').statistic
                n = sub.groupby(['medgrp','outcome'])['logodds'].count()

                # make new blocks
                if t_test>t_max and min(n)>2 and min(n.groupby('medgrp').sum())>n_min:
                    df.loc[df.strata.eq(ii),'block'] = df.loc[df.strata.eq(ii),'medgrp']

            if df.block.sum()==0:
                change = False
            else:
                # getting ready for next loop
                df.strata = df.groupby(['strata','block']).ngroup()
                df.block = 0

        return outcome.join(df.strata).strata

    # we will define a static method so that we can call this on any generic series
    @staticmethod
    def calc_trim(propscore):
        y = 1/(propscore*(1-propscore))
        
        if y.max() <= (2/y.count())*(y.sum()):
            return 0
        
        for gamma in linspace(y.max(),0,10000):
            lhs_estimand = (gamma/y.count())*(y.le(gamma).sum())
            rhs_estimand = (2/y.count())*((y.le(gamma)*y).sum())
            if lhs_estimand < rhs_estimand:
                break
        
        alpha = .5-((.25-(1/gamma))**.5)
        
        return alpha,1-alpha

