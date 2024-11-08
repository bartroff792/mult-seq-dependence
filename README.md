# Sequential Testing of Multiple Hypotheses

A Python package for sequential testing of multiple hypotheses under arbitrary joint distributions by [Mike Hankin](https://github.com/meh2135) and [Jay Batroff](https://github.com/bartroff792).

This package serves two main purposes:

* A library to run sequential step-down and sequential step-up tests on user provided data/data-streams.
* A simulation environment for examining the behavior of these procedures under a variety of configurations and data generating processes.

## Package Structure

### Library

All major python modules are in the `Code/utils` dir.

* `cutoff_funcs.py`: functions for building vectors of p-value cutoffs and log likelihood ratio cutoffs for sequential testing procedures
* `multseq.py`: actual testing procedure
* `data_funcs.py`: defines interfaces for streaming data.

### Simulation


* `data_funcs.py`: Functions for reading drug data, generating fake data, generating hypotheses, and computing llr paths.
* `simulation_orchestration.py`: This module contains functions for higher level simulation for sequential testing of multiple hypotheses (beyond just generating the observations), and executing the SPRT procedures on it.
* `docker_main.py`: launches a simulation run on google cloud platform. Located outside of the `utils` dir.

### Demo Data

* `AmnesiaRateClean.csv`: a table of drugs, each with the (annual) rate at which they've "generated" amnesia side effect reports, as well as the rate at which they've generated non-amnesia side effect reports. We recommend using their total side effect generation rate as a proxy for their usage.
* `GoogleSearchHitData.csv`: a table of drugs search popularity, and the proportion of those searches that include "amnesia". The popularity and naming schemes of drugs differ, so some of these may be of higher than expected variance. Further, many drugs' search rates weren't available.
* `YellowcardData.csv`: a bit too raw... contains number of total side effects, fatal side effects, amnesia reports, etc for each drug.

## Usage

### Simulation

To launch a simluation from the command line using the docker_main.py, and drop the results into a mysql database running at DBHOST, run something like the following:

```
python docker_main.py --alpha=0.1 --beta=0.05 --m_null=7 --m_alt=3 --hyp_type=binom --theta0=0.01 --theta1=0.05 --extra_params='n=3' --host=DBHOST --sim_reps=1000
```

The simulation results are stored in three MySQL tables:
- `simulation_metadata`: Tracks execution details like start time and number of repetitions
- `simulation_params`: Records all simulation parameters (alpha, beta, hypothesis counts, etc.)
- `simulation_results`: Stores results for each Monte Carlo iteration, including false discovery proportions, rejection counts, and sample numbers


### Library


This package handles both finite and infinite horizon testing. In the finite horizon case, type 1 error is directly controlled, and only rejective cutoffs are used. In the infinite horizon case, both type 1 and type 2 error are controlled, and cutoffs are used to both reject and accept hypotheses.

The package currently only fully supports simple hypotheses. Using infinite horizon testing of simple hypotheses, we can employ Wald's approximation to choose log likelihood ratio cutoffs. However, for finite horizon testing, even of simple hypotheses, we must rely on simulations to choose the cutoffs.


#### Running The Procedure

If you already know the cutoffs and have the test statistics (be they log likelihood ratio or any other arbitrary statistic) packaged as either a pandas DataFrame or an online data stream (implementing the `utils.data_funcs.online_data` interface), then you can just pass them to the msprt function, and analyze the results:

```
utils.msprt(
    statistics: pd.DataFrame,
    cutoffs: cutoff_funcs.CutoffDF,
    record_interval: int = 100,
    stepup: bool = False,
    rejective: bool = False,
    verbose: bool = True,
)
```

This returns an object of type `utils.multseq.MSPRTOut`, which is well documented in the source code (be sure to check the docstrings for the classes that comprise its attributes).

#### Developing Cutoffs

The first step is to decide on the basic structure of the error probabilities. That is, there are a range of shapes of error levels that, when scaled, yield the same FDR level but different power characteristics. Benjamini-Hochberg is the most classical, but we default to Benjamini-Liu (described in detail below).

```python
alpha_vec = cutoff_funcs.construct_base_pvalue_cutoffs(cut_type, m_total, 1.0 / (10.0 * m_total))
```
This will get you a vector of p-value cutoffs that are the same shape as the FDR level you've chosen, though they do NOT control FDR at that level under arbitrary dependence.

```python
alpha_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(
            alpha,
            alpha_vec,
            m0=m0,
        )
```

This snippet will scale the p-value cutoffs to control FDR at the level specified by alpha.

In the case of finite horizon testing, we can then use Wald's approximation to choose the log likelihood ratio cutoffs:
```python
cutoff_df = cutoff_funcs.calculate_mult_sprt_cutoffs(alpha, beta)
```

Note: pFDR control for infinite horizon testing is currently being rebuilt.

If we're going to rejective testing, we can use the following:

```python
A_vec = cutoff_funcs.estimate_finite_horizon_rejective_llr_cutoffs(
            params0=params0,
            params1=params1,
            hyp_type=hyp_type,
            n_periods=n_periods,
            alpha_levels=alpha,
            k_reps=k_reps,
            imp_sample=fh_cutoff_imp_sample,
            imp_sample_prop=fh_cutoff_imp_sample_prop,
        )
```
The first 3 arguments detail the null and alternative hypotheses; they're followed by the number of periods the FDR control level, the number of Monte Carlo repetitions used to estimate the cutoffs, and parameters that control importance sampling in that Monte Carlo estimation, used to ensure llr cutoffs for small p-values are not negative.

## In the weeds

### Theory

[Main Theorems](MainThms.md)

### BL scaling

Taking

* $m_{0}$ to be the number of true null hypotheses
* $m_{1}$ to be the number of false null hypotheses
* $\vec{\alpha}=(\alpha_{1}, \alpha_{2}, ... \alpha_{m_{0}+ m_{1}})$ to be the vector of p-value cutoffs such that $\alpha_{j}\leq \alpha_{j+1}$

Then define the Guo+Rao FDR bound for a stepdown procedure to be

$$
D(m_{0},m_{1},\vec{\alpha})=m_{0}(\sum_{j=1}^{m_{1}+1}\frac{\alpha_{j}-\alpha_{j-1}}{j}+\sum_{j=m_{1}+2}^{m}\frac{m_{1}(\alpha_{j}-\alpha_{j-1})}{j(j-1)})
$$

when $m_{0}$ (and $m_{1}$) are known, and

$$
D(\vec{\alpha}) = \max_{m_{0}\in \{1,...,m\}} D(m_{0}, m - m_{0}, \vec{\alpha})
$$

when they're unknown.

