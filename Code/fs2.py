#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:33:18 2017

@author: mhankin
"""
# %% Arg imports
import argparse, argcomplete, os
# Parallelization
parser = argparse.ArgumentParser()
parser.add_argument('--singlecore', action='store_true', help="Dont parallelize main sims")
parser.add_argument('--skiprej', action='store_true', help="skip rejective procedures")
parser.add_argument('--skipgen', action='store_true', help="skip general procedures")
parser.add_argument('--doviz', action='store_true', help="do visualizations")
parser.add_argument('--shelvepath', default="~/Dropbox/Research/MultSeq/Data/binpois.shelve", help="shelve record path")
parser.add_argument('--vizpath', default="~/Dropbox/Research/MultSeq/Data/binpois.html", help="Visualization html path")
parser.add_argument('--cfgpath', default="~/Dropbox/Research/MultSeq/Data/sim.cfg", help="Simulation configuration file")
parser.add_argument('--cfgsect', default="binpois", help="Simulation configuration section")
parser.add_argument('--usesect', action='store_true', help="Use default roots with cfgsect name")


argcomplete.autocomplete(parser)
cmd_args = parser.parse_args()

if cmd_args.usesect:
    cmd_args.vizpath = "~/Dropbox/Research/MultSeq/Data/{0}.html".format(cmd_args.cfgsect)
    cmd_args.shelvepath = "~/Dropbox/Research/MultSeq/Data/{0}.shelve".format(cmd_args.cfgsect)
    
    
    
from pylab import *
from scipy.stats import binom, poisson
from scipy.special import factorial
import multseq
from multseq import step_down_elimination
from utils.data_funcs import (assemble_fake_drugs, simulate_correlated_reactions, assemble_llr,
                              assemble_fake_binom, simulate_correlated_binom, assemble_binom_llr,
                              assemble_fake_pois, simulate_correlated_pois, assemble_pois_llr)
from utils.cutoff_funcs import (create_fdr_controlled_alpha, llr_term_moments, llr_binom_term_moments, 
                                llr_pois_term_moments)
from utils import data_funcs
from scipy.stats import norm as normal_var
import pandas
import tqdm
import statsmodels.formula.api as sm


#p0 = .05
#p1 = .045
alpha = .25
rho = -.6
m_null= 5
n_periods = 25
period_test_points = 10
base_periods = 400
period_max_exp = 6.0
reps_per_period_point = 100
scale_alpha=True
max_magnitude = 4.0

def single_binom_test(x, n, p0, p1):
    if p1>p0:
        return (1 - binom.cdf(x-1, n, p0))
    else:
        # p1 < p0
        return binom.cdf(x, n, p0)
    
def single_pois_test(x, lam0, lam1):
    if lam1>lam0:
        return (1 - poisson.cdf(x-1, lam0))
    else:
        # p1 < p0
        return poisson.cdf(x, lam0)

def get_oc_range(p0, p1, 
                 alpha = alpha, rho = rho, m_null = m_null, n_periods = n_periods,
    period_test_points = period_test_points, base_periods = base_periods, period_max_exp = period_max_exp,
    reps_per_period_point = reps_per_period_point, scale_alpha=scale_alpha, max_magnitude=max_magnitude, 
                 hyp_type="drug"):

    n_period_vec = (base_periods * (2.0 ** linspace(0.0, period_max_exp, period_test_points))).astype(int)
    
    
    #################    
    
    
    if (hyp_type is None) or (hyp_type=="drug"):
        dar, dnar, ground_truth = assemble_fake_drugs(max_magnitude, m_null, False, p0, p1)
        drr = dar + dnar
    elif hyp_type == "binom":
        dar, ground_truth = assemble_fake_binom(m_null, False, p0, p1)
        drr = pandas.Series(ones(len(dar)), index=dar.index)
        dnar = None
    elif hyp_type == "pois":
        dar, ground_truth = assemble_fake_pois(m_null, False, p0, p1)
        drr = pandas.Series(ones(len(dar)), index=dar.index)
        dnar = None
    else:
        raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))


    # Calculate alpha cutoffs
    N_drugs = len(dar)

    alpha_vec_raw = alpha * arange(1, 1+N_drugs) / float(N_drugs)
    if scale_alpha:
        scaled_alpha_vec = create_fdr_controlled_alpha(alpha, alpha_vec_raw)
    else:
        scaled_alpha_vec = alpha_vec_raw
    
    # Set up records
    rej_rec = pandas.DataFrame(zeros((reps_per_period_point, 2 *m_null)).astype(float), columns=dar.index)
    fdp_rec = zeros(reps_per_period_point)
    fnp_rec = zeros(reps_per_period_point)
    fdr_vec = zeros(period_test_points)
    fnr_vec = zeros(period_test_points)
    
    for j in tqdm.tqdm(range(period_test_points)):
        n_periods = n_period_vec[j]
        for i in range(reps_per_period_point):
            
            
            #######
            # Generate data
            if (hyp_type is None) or (hyp_type=="drug"):
                amnesia_ts, nonamnesia_ts = simulate_correlated_reactions(n_periods * dar, n_periods * dnar, 2, rho)
                llr_ts = assemble_llr(amnesia_ts, nonamnesia_ts, p0, p1)
                llr = llr_ts.iloc[-1]
                amnesia = amnesia_ts.iloc[0]
                tot_reacts = nonamnesia_ts.iloc[0] + amnesia
                nrm_approx = llr_term_moments(drr, p0, p1) * n_periods

                Z_scores = (llr - nrm_approx["term_mean"]) / sqrt(nrm_approx["term_var"])
                p_val = pandas.Series(1 - normal_var.cdf(Z_scores), index=dar.index)
            elif hyp_type == "binom":
                amnesia_ts = data_funcs.simulate_correlated_binom(dar, n_periods, rho)
                amnesia = pandas.DataFrame(amnesia_ts).iloc[-1]
                p_val = pandas.Series(dict([(drug_name, single_binom_test(amnesia_val, n_periods, p0, p1)) for drug_name, amnesia_val in amnesia.items()]))
                #llr = data_funcs.assemble_binom_llr(amnesia, p0, p1).iloc[0]
                #nrm_approx = llr_binom_term_moments(p0, p1) * n_periods
            elif hyp_type == "pois":
                amnesia_ts = data_funcs.simulate_correlated_pois(dar, n_periods, rho)
                amnesia = pandas.DataFrame(amnesia_ts).iloc[-1]
                
                p_val = pandas.Series(dict([(drug_name, single_pois_test(amnesia_val, n_periods * p0, n_periods * p1)) for drug_name, amnesia_val in amnesia.items()]))
                #llr = data_funcs.assemble_pois_llr(amnesia, p0, p1).iloc[0]
                #nrm_approx = llr_pois_term_moments(p0, p1) * n_periods
            else:
                raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
            #########
            
#             Universal code()
#             print(amnesia, type(amnesia), amnesia.shape)
#             print(p_val.round(3), type(p_val), p_val.shape)
            rej_idx, num_rej = step_down_elimination(p_val, scaled_alpha_vec, 0, 'low')
            rej_rec.loc[i, rej_idx] = 1.0
            fdp_rec[i] = ground_truth[rej_idx].sum() / max((1, num_rej))
            fnp_rec[i] = ((~ground_truth[~ground_truth.index.isin(rej_idx)]).sum()) / max((1, 2 * m_null - num_rej))
            
        
        fdr_vec[j] = fdp_rec.mean()
        fnr_vec[j] = fnp_rec.mean()
#         break

    return pandas.DataFrame({"fdr":fdr_vec, "fnr":fnr_vec, 
                             "logfdr":log(fdr_vec), "logfnr":log(fnr_vec), 
                             "samplesize":n_period_vec})
    

    
oc_range_pois = get_oc_range(p0=1.5, p1=2.0, hyp_type="pois", base_periods=2)
oc_range_pois.plot(kind="line", x="samplesize", y=["fdr", "fnr"])
oc_range_pois.plot(kind="line", x="samplesize", y=["logfdr", "logfnr"])
pois_ols_result = sm.ols(formula="logfnr ~ 1 + samplesize", data=oc_range_pois).fit()
print("Poisson")
print(pois_ols_result.params)
print(oc_range_pois)

oc_range_binom = get_oc_range(p0=0.05, p1=0.15, hyp_type="binom", base_periods=2)
oc_range_binom.plot(kind="line", x="samplesize", y=["fdr", "fnr"])
oc_range_binom.plot(kind="line", x="samplesize", y=["logfdr", "logfnr"])
binom_ols_result = sm.ols(formula="logfnr ~ 1 + samplesize", data=oc_range_binom).fit()
print("Binomial")
print(binom_ols_result.params)
print(oc_range_binom)

show()


