# Propensity Score Calculator

Estimate the Propensity Score in Python following Imbens and Rubin (2015)

## Description
This code is a work in progress, but allows one to estimate the propensity score (the probability of being in the treated group) following the general methodology laid out in [Imbens and Rubin (2015)](https://doi.org/10.1017/CBO9781139025751.014).

Support currently exists for first and second order terms. The method estimates in 3 steps. The first is done by the user, the remaining are done by the code.

### Step 1
Choose which covariates you think are relevant and should always be included in the propensity score equation. These will be in `init_vars` in the code.

### Step 2
Add additional linear terms that will be tested. These will be selected one-by-one according to which gives the largest overall gain in log odds in the propensity score calcuation. In each step we take the max such that the gain is greater than a predetermined value (the default is 1). Once no remaining variable gives a gain of at least one, the linear portion terminates.

### Step 3
Quadratic and interaction terms are automatically generated from the `init_vars` and the `test_vars` and these are compared in the same way as the linear terms except the log odds must increase by a separate amount (default is 2.71).

## Output

- `self.data`: DataFrame. This includes a new frame of just the outcome and potential covariates.

- `self.dropped_vars`: list. The variables that did not make the cut for singularity reasons.

- `self.model`: The fitted model from `Statsmodels`. This is the raw model on the final set of variables from Statsmodels

- `self.propscore`: Series. This is the propensity score as calculated by self.model.fittedvalues. This may not match dimension of data due to dropped missing values, but index will align properly.

- `self.test_vars_ord2`: list. The full list of tested second order variables for reference.

## References

Imbens, G., & Rubin, D. (2015). Estimating the Propensity Score. In Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction (pp. 281-308). Cambridge: Cambridge University Press. doi:10.1017/CBO9781139025751.014
