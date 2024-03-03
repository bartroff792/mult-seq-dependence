# Qs for Jay
## General
* When would we use MC to get the cutoffs for infinite horizon? That's totally unnecessary right?
* just need stepdown right?
* in a bunch of my funcs theres the option to replace the LLR with the historical cumulative max of the LLR... wtf?

## Deliverables
### Synthetic Data
* Scenarios
    * Control trypes
        * FDR controlled finite horizon
        * FDR and FNR controlled infinite horizon
        * pFDR controlled finite horizon
        * pFDR and pFNR controlled infinite horizon
    * Sim types
        * Poisson from H0 and H1
        * Binomial from H0 and H1
        * Do we want poisson grid? extra work there to visualize
        * are the 3 cuts of 10 total hypotheses sufficient?
    * Correlation structure
        * Independence
        * Gaussian copula with $Corr_{i,j}=\rho^{\vert i - j \vert}$ for rho pos and neg?
* Metrics
    * Average sample number before termination for infinite
        * is there an equivalet for finite horizon
    * FDR, FNR, pFDR, pFNR
    * SEs for all the above

### Yellowcard
* Sim under independence
* what stats?
* Far out: get embeddings from bio arxiv for corr groups


# MH fixes
* detailed comments on how n_period flows through simulation_orchestration.calc_sim_cutoffs namely that inf/fin is determined totally by beta, and passing n_periods overrides any sample size guessing AND that passing None gets 1000 for finite horizon