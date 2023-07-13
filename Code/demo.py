# -*- coding: utf-8 -*-
"""
Created on Tue May 24 19:40:42 2016

@author: mike
"""
from pylab import *
import pandas
import numpy
import numpy.random
import seaborn as sns
import multseq
import visualizations
import string
from tqdm import tqdm
from utils import cutoff_funcs
from utils.cutoff_funcs import (finite_horizon_rejective_cutoffs, 
                                create_fdr_controlled_alpha, guo_rao_stepdown_fdr_level)
from utils.data_funcs import (read_drug_data, simulate_reactions, assemble_llr, 
                              simulate_correlated_reactions, 
                              assemble_fake_drugs)
from utils.simulation_funcs import calc_sim_cutoffs
from utils import data_funcs, cutoff_funcs
import time
import logging, traceback
import multiprocessing 
import itertools
import traceback

EXTREME_STAT_PATH = pandas.DataFrame({'H1':array([0.0, 2.6, 3.5, 17.0, -30, 12]), 
                             'H2':array([0.0, 2.3, 1.5, -1.5, 1.5, 2.5]),
                             'H3':array([0.0, 0.5, -0.5, -1.5, -2.5, -1.5])})

REASONABLE_STAT_PATH = pandas.DataFrame({'H1':array([0.0, 2.6, 3.5, 6.0, -3, 4]), 
                             'H2':array([0.0, 2.3, 1.5, -1.5, 1.5, 2.5]),
                             'H3':array([0.0, 0.5, -0.5, -1.5, -2.5, -1.5])})
def general(stepup=False, use_extreme=True, do_annotation=False):
    if use_extreme:
        llr = EXTREME_STAT_PATH
    else:
        llr = REASONABLE_STAT_PATH
    A_vec = array([3.0, 2.0, 1.0])
    B_vec = -array([2, 1, .5])
#    llr.plot(style=".--")
#    hlines(A_vec, .5, 7.5, linestyles="solid")
#    hlines(B_vec, .5, 7.5, linestyles="dotted")
#    ylim(-4,4)
    
    
    #visualizations.plot_multseq_llr(llr.copy(), A_vec, B_vec, stepup=stepup)
    return visualizations.plot_multseq_llr(llr.copy(), A_vec, B_vec, 
                                    pandas.Series([False, False, True], index=llr.columns),
                                    do_annotation=do_annotation, ghost_lines=do_annotation, 
                                    jitter_mag=0.25, stepup=stepup) 

# TODO: replace all the long arg strings with a single test_config object
class test_config(object):
    def __init__(self, alpha, beta, n_periods, BH, undershoot_prob, min_am, min_tot, sim_reps):
        self.alpha = alpha
        self.beta = beta
        self.n_periods = n_periods
        self.BH = BH
        self.undershoot_prob = undershoot_prob
        self.min_am = min_am
        self.min_tot = min_tot
        self.sim_reps = sim_reps
        
        
def normal_plot(alpha=.05, beta=.1, n_periods = 14, stepup=False, m0=3, m1=2, 
                cut_type="BL", scale_fdr=True, ghost_lines=False, 
                do_annotation=False, cummax=False, use_streaming=True, rho=-0.5):
    mu0 = 1.0
    mu1 = 2.0
    mean_vec, sd_vec, ground_truth = data_funcs.assemble_fake_gaussian(3.0, m0, mu0, mu1, m_alt=m1)
    idx = pandas.Index(["stream"+str(i) for i in range(m0+m1)])
    mean_vec.index = idx
    sd_vec.index = idx
    ground_truth.index = idx
#    if beta is not None:
#        
#        _, temp_A_B = cutoff_funcs.calc_bh_alpha_and_cuts(alpha, beta, m0 + m1)
#    else:
#        alpha_levels = cutoff_funcs.create_fdr_controlled_alpha(alpha, arange(1.0, m0 + m1 +1, dtype=float)/ (m0 + m1))
#        A_vec = cutoff_funcs.finite_horizon_rejective_cutoffs(sd_vec, mu0, mu1, alpha_levels, 
#                                     n_periods, 1000, hyp_type="gaussian", sleep_time=1)
#        temp_A_B = (A_vec, None)
        
    A_vec, B_vec, _ = calc_sim_cutoffs(sd_vec, alpha, beta, scale_fdr, cut_type, 
                     p0=mu0, p1=mu1, stepup=stepup, m0_known=False,
                     m_total=m0+m1, n_periods=n_periods, undershoot_prob=.1,
                     hyp_type="gaussian", fin_par=True, fh_sleep_time=2)
    numpy.random.seed()
#    n_periods = cutoff_funcs.est_sample_size(temp_A_B[0], temp_A_B[1], drr, p0, p1)   
    
    
    samps_noncum = data_funcs.simulate_correlated_gaussian_noncum(mean_vec, sd_vec, n_periods, rho)
    llr = data_funcs.assemble_gaussian_llr(samps_noncum, mu0, mu1)
    if cummax:
        llr = llr.cummax()
    if use_streaming:
        gen_llr_params = dict(dar=mean_vec, dnar=sd_vec, n_periods=n_periods, 
                              rho=rho, hyp_type="gaussian", p0=mu0, p1=mu1, 
                              m1=m1, rho1=None, rand_order=False, cummax=cummax)
        finite_horizon = beta is None
        if finite_horizon:
            llr_df = data_funcs.generate_llr(**gen_llr_params)
            llr = data_funcs.df_generator(llr_df)
            dgp = llr._dgp
        else:
            dgp = data_funcs.infinite_dgp_wrapper(gen_llr_params, False)
            llr = data_funcs.online_data(mean_vec.index, dgp)

#        llr = data_funcs.df_generator(llr) #online_data(llr.columns, data_funcs.df_dgp_wrapper(llr))
        data_arg = dgp.get_data_record
    else:
        data_arg = None
    outviz = visualizations.plot_multseq_llr(llr, 
                                             A_vec, B_vec, ground_truth, 
                                             do_annotation=do_annotation, 
                                             ghost_lines=ghost_lines, 
                                             jitter_mag=0.01, 
                                             stat_data_func=data_arg, stepup=stepup)
    xlim(0.0, len(llr) +2)
    return outviz
    
    

        

     

def infinite_horizon_seq_stepdown_plot(alpha=.1, beta=.2, 
                                       BH=True, record_interval=100, 
                                       stepup=False, skip_main_plot=False,
                                       n_periods=1000, rho=None):

    dar, dnar, meta_data = data_funcs.read_drug_data()
    p0, p1 = data_funcs.am_prop_percentile_p0_p1(dar, dnar, .5, .9)
    drr_raw = (dar + dnar).values
    # Screen rare drugs and drugs with no amnesia SE
    drug_mask = array( (drr_raw > 10) & (dar > 1) )
    drr = drr_raw
    N_drugs =  len(drr)
#    _, temp_A_B = cutoff_funcs.calc_bh_alpha_and_cuts(alpha, beta, N_drugs)
#    est_periods = cutoff_funcs.est_sample_size(temp_A_B[0], temp_A_B[1], drr, p0, p1)
#    if est_periods < 1000:
#        est_periods = 1000
#    print(est_periods)
#    raise ValueError()
    #p0 = .001
    #p1 = .01
#    n_periods = 10
    print("p0: {0}".format(p0))
    print("p1: {0}".format(p1))
    # Generate data
    amnesia, nonamnesia = simulate_correlated_reactions(dar, 
                                                        dnar, 
                                                        n_periods,
                                                        rho)
    print(amnesia)
    llr = assemble_llr(amnesia, nonamnesia, p0, p1)

    # Calculate cutoffs
    N_drugs = len(llr.columns)                       
    A_vec, B_vec = cutoff_funcs.calculate_mult_sprt_cutoffs( alpha * arange(1, 1+N_drugs) / float(N_drugs),
                                                             beta * arange(1, 1+N_drugs) / float(N_drugs))
    
    # Perform testing procedure

    return visualizations.plot_multseq_llr(llr.copy(), A_vec, B_vec, stepup=stepup, skip_main_plot=skip_main_plot)

def finite_horizon_seq_stepdown_plot(alpha=.1, n_periods=1000,BH=True, record_interval=100,
                                       stepup=False, skip_main_plot=False, do_scale=True,
                                       rho=None, imp_sample_prop=0.85, k_reps=1000, do_parallel=True,
                                       sleep_time=30.0):
    """Perform finite horizon sequential stepdown procedure on drug data.
    
    args:
        alpha: (float)
        BH: (bool)
        record_interval: (int)

    return:
	
    """
    dar, dnar, meta_data = data_funcs.read_drug_data()
    p0, p1 = data_funcs.am_prop_percentile_p0_p1(dar, dnar, .5, .9)
    drr_raw = (dar + dnar).values
    # Screen rare drugs and drugs with no amnesia SE
    drug_mask = array( (drr_raw > 10) & (dar > 1) )
    drr = drr_raw
    N_drugs =  len(drr)
#    _, temp_A_B = cutoff_funcs.calc_bh_alpha_and_cuts(alpha, beta, N_drugs)
#    est_periods = cutoff_funcs.est_sample_size(temp_A_B[0], temp_A_B[1], drr, p0, p1)
#    if est_periods < 1000:
#        est_periods = 1000
##    print(est_periods)
##    raise ValueError()
#    #p0 = .001
#    #p1 = .01
##    n_periods = 10
#    n_periods = est_periods
    # Generate data
    amnesia, nonamnesia = simulate_correlated_reactions(dar, 
                                                        dnar, 
                                                        n_periods,
                                                        rho)
#    print(amnesia)
    llr = assemble_llr(amnesia, nonamnesia, p0, p1)

    # Calculate cutoffs
    N_drugs = len(llr.columns)    
    # First come up with p-value cutoffs
    # BH
    if BH:
        alpha_vec_raw = alpha * arange(1, 1+N_drugs) / float(N_drugs)
    # Holm
    else:        
        alpha_vec_raw = alpha / (float(N_drugs) - arange(N_drugs))
    if do_scale:
        scaled_alpha_vec = create_fdr_controlled_alpha(alpha, alpha_vec_raw)
    else:
        scaled_alpha_vec = alpha_vec_raw

    # Next calculate llr cutoffs
    min_alpha_diff = min(diff(scaled_alpha_vec))
    print("Approx number of MC reps, unscaled: {0}".format(int(1.0/min_alpha_diff)))
    
    #raise Exception(str(max(int(1.0/min_alpha_diff), 100)))
#    k_reps = 1000
    A_vec = finite_horizon_rejective_cutoffs(drr, p0, p1, scaled_alpha_vec, 
                                                  n_periods, k_reps, imp_sample_prop=imp_sample_prop,
                                                  do_parallel=do_parallel,
                                                  sleep_time=sleep_time)
    
    # Perform testing procedure

    return visualizations.plot_multseq_llr(llr.copy(), A_vec, None, stepup=stepup, skip_main_plot=skip_main_plot)

    
if __name__=="__main__":
    normal_plot()
    show()