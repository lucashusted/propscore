#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from statsmodels.api import Logit
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
    disp : TYPE, optional
        Display the final model including dropped variables. The default is True.
    cutoff_ord1 : TYPE, optional
        The log gain cutoff for first order covariates. The default is 1.
    cutoff_ord2 : TYPE, optional
        The log gain cutoff for second order covariates. The default is 2.71.

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
    self.test_vars_ord2: list
        The full list of tested second order variables for reference.

    """
    def __init__(self, outcome, test_vars, df, init_vars=None, add_cons=True, disp=True,
                 cutoff_ord1 = 1, cutoff_ord2 = 2.71):

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

        if disp:
            print(self.model.summary())
            print('The following vars were infeasible: {}'.format(', '.join(self.dropped_vars)))


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
