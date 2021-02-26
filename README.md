# Propensity Score Calculator

Estimate the Propensity Score in Python following [Imbens and Rubin (2015a)](https://doi.org/10.1017/CBO9781139025751.014). Several by-products are calculated including:

- Strata based on the estimated propensity score [Imbens and Rubin (2015a)](https://doi.org/10.1017/CBO9781139025751.014)

- Suggested Maximum and Minimum values of the propensity score to maintain covariate balance through trimming [Imbens and Rubin (2015b)](https://doi.org/10.1017/CBO9781139025751.017)


## Installation
Use `pip` to install:
```
pip install propscore
```

## Description
This code is a work in progress, but allows one to estimate the propensity score (the probability of being in the treated group) following the general methodology laid out in [Imbens and Rubin (2015a)](https://doi.org/10.1017/CBO9781139025751.014).

Support currently exists for first and second order terms. The method estimates in 3 steps. The first is done by the user, the remaining are done by the code.

### Step 1
Choose which covariates you think are relevant and should always be included in the propensity score equation. These will be in `init_vars` in the code.

### Step 2
Add additional linear terms that will be tested. These will be selected one-by-one according to which gives the largest overall gain in log odds in the propensity score calcuation. In each step we take the max such that the gain is greater than a predetermined value (the default is 1). Once no remaining variable gives a gain of at least one, the linear portion terminates.

### Step 3
Quadratic and interaction terms are automatically generated from the `init_vars` and the `test_vars` and these are compared in the same way as the linear terms except the log odds must increase by a separate amount (default is 2.71).

## Example Use
The following is sample code to illustrate the use (note: this example will not run as I have not bothered to find a suitable dataset for testing, this is still a work in progress.
```
from propscore import PropensityScore

# Imagine you have pandas DataFrame, df, with columns "outcome" an "var1"-"var5".
# We wish to always include var2 and var3, and want to test the relevance
# of the other variables and higher order terms.

output = PropensityScore('outcome', ['var1','var4','var5'], df, init_vars=['var2','var3'])

# The propensity score values are given in the pandas Series:
output.propscore


# To see the different strata calculated, you can reference
output.strata

# However, you can also calculate strata based on log-odds as follows (with minimum 20 observations):
# This may be useful if you want to use the package to calculate strata once you have propensity scores
strata = PropensityScore.stratify(df.outcome,output.logodds,n_min_strata=20)

# Finally, the program automatically calculates a suitable min and max value of the propensity score:
output.trim_range

# This is also a static method (like stratify), so you can calculate the suitable min/max of the propensity score as:
in_trim = PropensityScore.calc_trim(output.propscore)

```

## Output

- `self.data`: DataFrame. This includes a new frame of just the outcome and potential covariates.

- `self.dropped_vars`: list. The variables that did not make the cut for singularity reasons.

- `self.logodds`: Series. This is the log-odds ratio or the linearized propensity score. Equivalent to `self.model.fittedvalues`. This may not match dimension of data due to dropped missing values, but index will align properly on nonmissing values.

- `self.model`: The fitted model from `Statsmodels`. This is the raw model on the final set of variables from Statsmodels

- `self.propscore`: Series. This is the propensity score as calculated by `self.model.predict()`. This may not match dimension of data due to dropped missing values, but index will align properly on nonmissing values.

- `self.test_vars_ord2`: list. The full list of tested second order variables for reference.

- `self.strata`: Series. The calculated strata (index starting at 0) that result from the propensity score, also following the same chapter. Missing values in outcome variable and propensity scores that fall outside of the allowable values (larger than max of control group and smaller than min of treatment group) are coded as NaN.

- `self.trim_range`: tuple. The result of calculating the optimal trim min and max propensity score values.

- `self.in_trim`: Series (True/False). An array where True means that the propensity score falls within the trim min/max range.

## References

Imbens, G., & Rubin, D. (2015a). Estimating the Propensity Score. In Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction (pp. 281-308). Cambridge: Cambridge University Press. doi:10.1017/CBO9781139025751.014

Imbens, G., & Rubin, D. (2015b). Trimming to Improve Balance in Covariate Distributions. In Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction (pp. 359-374). Cambridge: Cambridge University Press. doi:10.1017/CBO9781139025751.017
