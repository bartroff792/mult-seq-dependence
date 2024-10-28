# Sequential Testing of Multiple Hypotheses

This package serves two main purposes:

* Can be used as a library to run sequential stepdown and sequential step-up tests on user provided data/data-streams.
* Can be used to run a wide array simulations of applying the procedures to simple, synthetic data generating processes where the true parameter values are known.

## Package Structure

### Library

All major python modules are in the `Code/utils` dir.

* `cutoff_funcs.py`: functions for building vectors of p-value cutoffs and log likelihood ratio cutoffs for sequential testing procedures
* `multseq.py`: actual testing procedure
* `data_funcs.py`: defines interfaces for streaming data.

### Simulation


* `data_funcs.py`: Funcs for reading drug data, generating fake data, generating hypotheses, and computing llr paths.
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

### Library


#### Cutoffs known

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

#### Cutoffs Unknown

....WIP

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

