# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 23:00:19 2016

@author: mike
"""
import numpy as np
from numpy import (log, exp, arange, hstack, linalg, zeros, log10, percentile, 
                   diff, array, mod, ones, linspace, argmax, vstack, newaxis, 
                   isnan, interp, argsort)
from scipy.optimize import minimize
import numpy
from pandas import Series
import pandas
import datetime
import warnings
from tqdm import tqdm, trange
from .data_funcs import simulate_reactions
from . import data_funcs
from .common_funcs import log_odds, sigmoid, chunk_mc
import multiprocessing 
import time
import logging
import shelve
import itertools
import traceback
from scipy.stats import t as tdist
from scipy.stats import norm as gaussian
    
# FDR controlled pvalue cutoffs and related functions
FloatArray = np.typing.NDArray[np.float32]

def fdr_helper(alpha_vec, m0=None, get_max_m0=False):
    """Returns FDR bound for a given set of alphas and the number of true nulls.
    
    Assumes alpha_vec[0] = alpha_1 <= alpha_vec[1] = alpha_2 <= alpha_vec[m-1] = alpha_m    
    Args:
        alpha_vec: increasing array of p-value/significance cutoffs
        m0: number of true null hypotheses.
        get_max_m0: If m0 is None and a search is to be performed, passing True
            to this argument will return both the max fdr and the m0 at which
            it was achieved.
    Return:
        Tight upper bound on FDR, when m0 is known.
    """
    # Verify alpha vec
    assert np.any(np.diff(alpha_vec) < 0.0), "Alpha vector is not monotone increasing."
    assert np.any(0.0>=alpha_vec), "Alpha vector contains values less than or equal to 0."
    if np.any(1.0<= alpha_vec):
        warnings.warn("Alpha vector contains values greater than or equal to 1.")
        
    if m0 is None:
        fdr_vec = [fdr_helper(alpha_vec, m0) for m0 in range(1, len(alpha_vec) + 1)]
        if get_max_m0:
            return max(fdr_vec), argmax(fdr_vec)
        else:
            return max(fdr_vec)
        
    # Otherwise proceed
    # Total number of hypotheses
    m = len(alpha_vec)
    # Number of false hypotheses
    m1 = m - m0
    index_vec = arange(1, m+1)
    # Pad the alpha vector with 0 and take its first order difference so that 
    # \alpha_{1} - \alpha_{0} = \alpha_{1}
    # Now diff_vec[i] = \alpha_{i+1} - \alpha_{i}, so to make reconciling this
    # with the formulas more efficient, we use index_vec.
    # diff_vec[i] = \alpha_{index_vec[i]} - ...
    diff_vec = diff(hstack(([0],alpha_vec)))
    # \sum_{i=1}^{m1+1} (\alpha_i - \alpha_{i-1}) / i
    term1 = (diff_vec[index_vec<=m1+1]/index_vec[index_vec<=m1+1]).sum()
    # \sum_{i=m1+2}^{m} m1 *(\alpha_i - \alpha_{i-1}) / (i * (i-1))
    term2 = m1 * (diff_vec[index_vec > m1+1]/(index_vec[index_vec > m1+1]*(index_vec[index_vec > m1+1]-1))).sum()
    return m0 * (term1 + term2)

def fdr_func(alpha_vec):
    """DEPRECATED: Calculates FDR bound for set of significance cutoffs when number of true nulls is unknown.
    Maximizes D(\vec{\alpha}) over m0
    Use fdr_helper without a second argument instead.
    
    Args:
        alpha_vec: increassing array of p-values/significance cutoffs.
    Returns:
        Scalar FDR bound when the alpha_vec is used in a sequential stepdown.
    """
    warnings.warn("fdr_func is deprecated. Use fdr_helper without a second argument instead.")
    fdr_vec = [fdr_helper(alpha_vec, m0) for m0 in range(1, len(alpha_vec) + 1)]
    return max(fdr_vec)

def fdr_scaling_func(fdr_level, alpha_vec):
    """
    Returns a constanst lambda such that using cutoffs (alpha_1*lambda, ... alpha_m*lambda) will control fdr at fdr_level
    
    Args:
        fdr_level: level at which FDR must be controlled.
        alpha_vec: initial, unscaled vector of increasing alpha cutoffs.
    Returns:
        A scaling factor which, when multiplied by alpha_vec, will return a 
        proportional cutoff vector that controls alpha at the correct level.
    """
    return fdr_level / fdr_helper(alpha_vec)
    
def create_fdr_controlled_alpha(fdr_level, alpha_vec):
    return fdr_scaling_func(fdr_level, alpha_vec) * alpha_vec

def create_fdr_controlled_bh_alpha_indpt(fdr_level, m_hyps):
    return fdr_level * arange(1, m_hyps + 1, dtype=float)  / float(m_hyps)
    
    
def create_fdr_controlled_bh_alpha(fdr_level, m_hyps):
    alpha_vec_raw = create_fdr_controlled_bh_alpha_indpt(fdr_level, m_hyps)
    return create_fdr_controlled_alpha(fdr_level, alpha_vec_raw)
    
def create_fdr_controlled_bl_alpha_indpt(fdr_level: FloatArray, m_hyps: int, hedge: bool=True):
    """Get FDR controlled alpha cutoffs for the Benjamini-Liu stepdown procedure w/independence.
    
    See Benjamini Liu (1999). Stepdown cutoffs for independent hypotheses. 
    alpha_i = 1 - (1 - min(1, m * alpha / (m - i + 1))) ** (1 / (m - i + 1))
    """
    # Create a matrix for the cases 
    tempmat = np.ones((2, m_hyps))
    tempmat[1, :] = m_hyps * fdr_level / (m_hyps - np.arange(m_hyps)).astype(float)
    casevec = tempmat.min(0)
    # Construct alpha vector
    alpha_vec = 1.0 - (1.0 - casevec) ** (1.0 / (m_hyps - arange(m_hyps)).astype(float))
    
    
    # If there are low-significance cutoffs equal to 1.0 and the hedge option
    # is requested, shrink them slightly
    num_sure_rejects = (alpha_vec == 1.0).sum()
    if hedge and (num_sure_rejects > 0):
        # Choose a value to shrink the most significant cutoff thats currently
        # equal to 1.0 down to by either choosing the least significant cutoff
        # less than 1.0 or 1- .5 / m
        low1 = 1.0 - (min(1.0 - alpha_vec[-(num_sure_rejects +1 )], 1.0 / m_hyps) / 2.0)
        # Interpolate from there to 1.0, leaving off the 1.0
        oneadjvec =  linspace(low1, 1.0, num_sure_rejects + 1)
        alpha_vec[-num_sure_rejects:] = oneadjvec[:-1]
    
    return alpha_vec
    
def create_fdr_controlled_bl_alpha(fdr_level, m_hyps, indpt_fdr_level=None):
    """a la Benjamini Liu (1999), scaled with Guo Rao for arbitrary joint. 
    See create_fdr_controlled_bl_alpha_indpt.
    """
    if indpt_fdr_level is None:
        indpt_fdr_level = fdr_level
    alpha_vec_raw = create_fdr_controlled_bl_alpha_indpt(indpt_fdr_level, m_hyps)
    # print(delvec)
    return create_fdr_controlled_alpha(fdr_level, alpha_vec_raw)
                                       
def calc_bh_alpha_and_cuts(fdr_level, fnr_level, N_drugs):
    alpha_vec = create_fdr_controlled_bh_alpha(fdr_level, N_drugs)
    beta_vec = create_fdr_controlled_bh_alpha(fnr_level, N_drugs)
    return (alpha_vec, beta_vec), calculate_mult_sprt_cutoffs(alpha_vec, beta_vec)

def cutoff_truncation(cut_vec):
    """Prevents negative cutoffs.
    """
    cut_vec = cut_vec.copy()
    lowest_pos = cut_vec[cut_vec>0].min()
    num_neg = (cut_vec <= 0).sum()
    if num_neg > 0:
        warnings.warn("Truncating cutoffs")
    cut_vec[cut_vec <= 0] = linspace(lowest_pos, 0.0, num_neg + 2)[1:-1]
    return cut_vec

def calculate_mult_sprt_cutoffs(alpha_vec, beta_vec, rho=.583, do_trunc=True):
    """Uses Wald approx to calculate llr cutoffs from type 1 and 2 error thresholds.
    A > 0 > B    
    Reject H0 when Lambda > A
    Accept H0 when Lambda < B
    Args:
        alpha_vec: increasing vector of type 1 error cutoffs.
        beta_vec: increasing vector of type 2 error cutoffs.
        rho: scalar adjustment factor for the Wald approximations.
    Returns:
        2-tuple of vectors A_vec and B_vec
        A_vec: increasing rejection cutoffs **for log likelihood ratio**
        B_vec: decreasing acceptance cutoffs **for log likelihood ratio**
    """
    alpha1 = alpha_vec[0]
    beta1 = beta_vec[0]
    A_vec = log((1 - alpha1 - beta1 * (1 - alpha_vec)) / 
                    (alpha_vec * (1 - alpha1))) - rho    
    B_vec = log(beta_vec * (1 - beta1) / 
                (1 - beta1 - alpha1 * (1 - beta_vec))) + rho
                
    if (A_vec < 0).any():
        num_neg =  (A_vec < 0).sum()
        
        warnings.warn("{0} A_vec Cutoff levels are negative: from {1} to {2}".format(
                num_neg, A_vec.min(), A_vec.max()))
        
        if do_trunc:
            A_vec = cutoff_truncation(A_vec)
        
    if (B_vec > 0).any():
        num_pos =  (B_vec > 0).sum()
        
        warnings.warn("{0} B_vec Cutoff levels are positive: from {1} to {2}".format(
                num_pos, B_vec.min(), B_vec.max()))
        
        if do_trunc:
            B_vec = -cutoff_truncation(-B_vec)
    
    return A_vec, B_vec

# TODO: fix this. calculating for reversed statistics
def get_pvalue_cutoffs(A_vec, B_vec, rho=.583):
    """Inverts Wald approx to get type 1/2 error cutoffs from llr cutoffs.
    """
    alpha_vec = zeros(A_vec.shape)
    beta_vec = zeros(B_vec.shape)
    highest_sig = linalg.solve(array([[exp(B_vec[0]+rho), 1],
                                      [exp(A_vec[0]-rho), 1]]),
                              array([1, exp(A_vec[0]-rho)]))
    alpha_vec[0] = highest_sig[0]
    beta_vec[0] = highest_sig[1]
    
    alpha_vec[1:] = (1 - highest_sig.sum()) / ((1-highest_sig[0]) * exp(B_vec[1:] + rho) - highest_sig[1])
    beta_vec[1:] = (1 - highest_sig.sum()) / ((1-highest_sig[1]) * exp(-A_vec[1:] + rho) - highest_sig[0])
    return alpha_vec, beta_vec
    

# TODO: fix this. calculating for reversed statistics    
def pfdr_pfnr_cutoffs(alpha_raw_vec, beta_raw_vec, pfdr, pfnr, m0, epsilon=10.0**-8):
    m = len(alpha_raw_vec)
    m1 = m - m0
    alpha_vec0 = alpha_raw_vec
    beta_vec0 = beta_raw_vec
    pfdrx = fdr_helper(alpha_vec0, m0) / (1 - beta_vec0[-1])
    pfnrx = fdr_helper(beta_vec0, m1) / (1 - alpha_vec0[-1])
    alpha_vec1 = pfdr * alpha_vec0 / pfdrx
    beta_vec1 = pfnr * beta_vec0 / pfnrx
    while( max(abs(alpha_vec1 - alpha_vec0))>epsilon  or  max(abs(beta_vec1 - beta_vec0))>epsilon):
        alpha_vec0 = alpha_vec1[:]
        beta_vec0 = beta_vec1[:]
        pfdrx = fdr_helper(alpha_vec0, m0) / (1 - beta_vec0[-1])
        pfnrx = fdr_helper(beta_vec0, m1) / (1 - alpha_vec0[-1])
        alpha_vec1 = pfdr * alpha_vec0 / pfdrx
        beta_vec1 = pfnr * beta_vec0 / pfnrx
        logging.debug("Rej alpha log range: {0}\nAcc beta log range: {1}".format(
                      log10(max(abs(alpha_vec1 - alpha_vec0))), 
                      log10(max(abs(beta_vec1 - beta_vec0)))))
    return alpha_vec1, beta_vec1


def finite_sim_func(p0, p1, drr, n_periods, n_reps, job_id, hyp_type, 
                    imp_sample=True, imp_sample_prop=.2, imp_sample_hedge=.9):
    """Generate finite sample path maxs (and weights)
    args:
        p0: null param
        p1: alt param
        n_periods: horizon
        n_reps: number of MC reps
        job_id: Used for parralelization and reporting
        hyp_type: drug, binom, etc
        imp_sample: boolean, use importance sampling
        imp_sample_prop: p=p1 * q + p0 * (1-q)  as simulation dist param
        imp_sample_hedge: proportion of samples to draw from importance 
            sampling, vs true null
    """

    
    out_rec = []
    
        
    if imp_sample:
        if hyp_type=="drug" or hyp_type=="binom":
            sim_param = sigmoid(imp_sample_prop * log_odds(p1) + (1.0 - imp_sample_prop) * log_odds(p0))
        elif hyp_type=="pois":
            sim_param = exp(imp_sample_prop * log(p1) + (1.0 - imp_sample_prop) * log(p0))
        elif hyp_type=="gaussian":
            sim_param = imp_sample_prop * p1 + (1.0 - imp_sample_prop) * p0
        else:
            raise Exception("Unrecognized hyp type ", hyp_type)
            
        weight_out = []
        max_imp_samples = int(n_reps * imp_sample_hedge)
        do_imp = True
    else:
        sim_param = p0
        do_imp = False
        
    
    logger = logging.getLogger()
    if job_id==0:
        logger.setLevel(logging.DEBUG)
        range_vec = tqdm(range(n_reps), desc="MC cutoff simulations, Job 0")
        logger.debug("Simulating {0} with param {1}".format(hyp_type, sim_param))
    else:
        logger.setLevel(logging.INFO)
        range_vec = range(n_reps)
        
    for it, i in enumerate(range_vec):
        if it % int(len(range_vec) / 5.0)==1:
            print("job {0} is on {1}".format(job_id, it))
        
        if do_imp and (i >= max_imp_samples):
            logger.debug("Switch to H0 at {0} of {1}.".format(i, n_reps))
            do_imp = False
            sim_param = p0
                            
            
        if (hyp_type is None) or (hyp_type=="drug"):
            DRR_FACTOR = .8
            amnesia, nonamnesia = simulate_reactions(sim_param * drr, 
                                                     (1.0 - sim_param) * drr, 
                                                     n_periods)
            llr = data_funcs.assemble_drug_llr((amnesia, nonamnesia), p0, p1)
            if imp_sample:
                weight_out.append(data_funcs.imp_sample_drug_weight((amnesia, nonamnesia), p0, sim_param))
#                if i==0 and job_id==0:
#                    print("llr {0}".format(llr))
        elif hyp_type == "binom":
            amnesia = data_funcs.simulate_binom(pandas.Series(sim_param * ones(len(drr)), index=drr.index), n_periods)
            llr = data_funcs.assemble_binom_llr(amnesia, p0, p1)
            if imp_sample:
                weight_out.append(data_funcs.imp_sample_binom_weight(amnesia, p0, sim_param))
        elif hyp_type == "pois":
            amnesia = data_funcs.simulate_pois(pandas.Series(sim_param * ones(len(drr)), index=drr.index), n_periods)
            llr = data_funcs.assemble_pois_llr(amnesia, p0, p1)
            if imp_sample:
                weight_out.append(data_funcs.imp_sample_pois_weight(amnesia, p0, sim_param))
        elif hyp_type == "gaussian":
            amnesia = data_funcs.simulate_gaussian_noncum(pandas.Series(sim_param * ones(len(drr)), index=drr.index), drr, n_periods)
            llr = data_funcs.assemble_gaussian_llr(amnesia, p0, p1)
            if imp_sample:
                weight_out.append(data_funcs.imp_sample_gaussian_weight(amnesia, p0, sim_param, drr))
        else:
            raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
            
        if isnan(llr).any().any():
            raise ValueError("NaN in llr at iter {0}".format(i))
        if imp_sample and isnan(array(weight_out)).any().any():
            raise ValueError("NaN in weights at iter {0}".format(i))
#            # Record the max value the each llr path reached
        out_rec.append(llr.max(0))

    if imp_sample:
        return array(out_rec), array(weight_out)
        
    else:
        return out_rec
    
import traceback
def finite_sim_func_wrapper(kwargs):
    numpy.random.seed(kwargs['job_id'])
    try:
        return finite_sim_func(**kwargs)
    except:
        print(traceback.format_exc())
        return np.array([]), np.array([])

        
ALPHA_SHIFT = 0.0
def empirical_quant_presim_wrapper(kwargs):
    stream_specific_cutoff_levels = []
    record = kwargs["record"]
    weights = kwargs["weights"]
    alpha_levels = kwargs["alpha_levels"]
    if "no_left_extrapolate" in kwargs:
        left = np.NaN
    else:
        left = None

    range_iter = range(record.shape[1])
    
    if ("job_id" not in kwargs) or (kwargs["job_id"]==0):
        range_iter = tqdm(range_iter, desc="Per stream quantile estimation")
    
    for stream_num in range_iter:
        stream_record_raw = record[:, stream_num]
        stream_weight_raw = weights[:, stream_num]
        stream_idx = argsort(stream_record_raw)
        stream_record = stream_record_raw[stream_idx]
        stream_weight = stream_weight_raw[stream_idx] / stream_weight_raw.sum()
        stream_cdf = stream_weight.cumsum() - ALPHA_SHIFT * stream_weight[0]
        
        stream_cutoffs  = interp(1.0 - alpha_levels, stream_cdf, stream_record,
                                 left=left, right=np.NaN)
        if isnan(stream_cutoffs).any():
            print("RAW ", stream_weight[0])
            print("Stream min {0} stream max {1}".format(stream_cdf.min(), stream_cdf.max()))
            print("alpha min {0} alpha max {1}".format((1.0 - alpha_levels).min(), 
                  (1.0 - alpha_levels).max()))
            raise ValueError("NaN found in stream cutoffs.")
        
        stream_specific_cutoff_levels.append(stream_cutoffs)
    stream_specific_cutoff_levels = array(stream_specific_cutoff_levels).T
    return stream_specific_cutoff_levels

def finite_horizon_rejective_cutoffs(rate_data, p0, p1, alpha_levels, 
                                     n_periods, k_reps, dbg=False, do_parallel=False,
                                     hyp_type="drug", sleep_time=5, 
                                     normal_approx=False, imp_sample=True,
                                     imp_sample_prop=.5, imp_sample_hedge=.9,
                                     divide_cores=None):
    """Calculate finite horizon rejective cutoffs from alpha levels using MC sim.
    
    Number of simulations should exceed the inverse of the min diff of the 
    alpha vector, otherwise you might end up with the same cutoff values for 
    multiple levels.
    args:
        rate_data: vector of emission rates
        p0: null p
        p1: alt p
        alpha_levels: type 1 levels for which cutoffs should be calculated
        n_periods: finite horizon
        k_reps: number of MC sims for
    return:
        array of cutoff values which should be > 0
        
    """
    drr = Series(rate_data)
    record = []
    print("Num hyps", len(drr))
    print("n per", n_periods)
    
    ten_pct = int(k_reps / 10.0)
    # Run simulation k_reps times
    if do_parallel:
        num_cpus = multiprocessing.cpu_count()
        num_jobs = num_cpus - 1
        if divide_cores is not None:
            num_jobs = int(num_jobs / divide_cores)
            if num_jobs < 1:
                num_jobs = 1 
            print("num jobs {0}".format(num_jobs))
        pool = multiprocessing.Pool(num_cpus-1)
        n_rep_list = chunk_mc(k_reps, num_jobs)
        rs = pool.map_async(finite_sim_func_wrapper, [{"p0":p0, "p1":p1, 
           "drr":drr, "n_periods":n_periods, "n_reps":n_rep, "job_id":job_id,
           "hyp_type":hyp_type, "imp_sample":imp_sample, 
           "imp_sample_prop":imp_sample_prop, "imp_sample_hedge":imp_sample_hedge} 
           for job_id, n_rep in enumerate(n_rep_list)])
#        pool.close()
        remaining = rs._number_left
        logging.info("Rej Cutoffs MC waiting for  {0} tasks to complete. ({1})\n".format(
                    remaining, time.ctime(time.time())))
        while (True):
            if (rs.ready()) or rs._number_left<2: 
                break
            remaining = rs._number_left
            logging.info("Rej Cutoffs MC waiting for {0} tasks to complete. ({1})".format(
                remaining, time.ctime(time.time())))
            time.sleep(sleep_time)
        remaining = rs._number_left
        logging.info("Rej Cutoffs MC waiting for {0} tasks to complete. ({1})\n".format(
                    remaining, time.ctime(time.time())))
        record_raw = rs.get()
        pool.close()
        
        
        
        if imp_sample:
            zipped_recs = zip(*record_raw)
            record, weights = [vstack(rec_item) for rec_item in zipped_recs]
                
        else:
            record = list(itertools.chain.from_iterable(record_raw))
    else:
        record_raw = finite_sim_func_wrapper(
                    {"p0":p0, "p1":p1, "drr":drr, "n_periods":n_periods, 
                     "n_reps":k_reps, "job_id":0, "hyp_type":hyp_type, 
                     "imp_sample":imp_sample, 
                     "imp_sample_prop":imp_sample_prop, 
                     "imp_sample_hedge":imp_sample_hedge})
        if imp_sample:
            record, weights = record_raw
        else:
            record = record_raw
    # Combine all path maximums into one array.
    # Shape is (# of reps, # of hyps)
    record = array(record)
    
    # Get the cutoffs for each individual stream, either exact or using a tdist
    if normal_approx:
        warnings.warn("Using normal approximation for quantile estimation.")
        
#        fit_vals = [tdist.fit(record[:,ii]) for ii in tqdm(range(len(rate_data)), desc="t dist fits")]
#        stream_specific_cutoff_levels = array([tdist(*t_dist_vals).ppf(1-alpha_levels) for t_dist_vals in tqdm(fit_vals, desc="t dist quantiles")])
        
        fit_vals = [gaussian.fit(record[:,ii]) for ii in tqdm(range(len(rate_data)), desc="t dist fits")]
        stream_specific_cutoff_levels = array([gaussian(*dist_vals).ppf(1-alpha_levels) for dist_vals in tqdm(fit_vals, desc="t dist quantiles")])
    if imp_sample:
        stream_specific_cutoff_levels = empirical_quant_presim_wrapper({"record":record, "weights":weights,
                                        "alpha_levels":alpha_levels,
                                        "job_id":0})
        
    else:
        stream_specific_cutoff_levels = empirical_quant_presim_wrapper({"record":record, "weights":ones(record.shape),
                                        "alpha_levels":alpha_levels,
                                        "job_id":0})
#        stream_specific_cutoff_levels = percentile(record, 100 * (1 - alpha_levels), axis=0)

    # Take the max across streams for each cutoff.
    # Take A_j = max_i A_j^i, then 
    # P(Lambda^i > A_j) < P(Lambda^i > A_j^i) < alpha_j    
    cutoff_levels = stream_specific_cutoff_levels.max(1)
    
#    from IPython.display import display
#    display(cutoff_levels)
    
    if (cutoff_levels < 0).any():
        num_neg =  (cutoff_levels < 0).sum()
        
        warnings.warn("{0} Cutoff levels are negative: from {1} to {2}".format(
                num_neg, cutoff_levels.min(), cutoff_levels.max()))
    
    if dbg:
        return cutoff_levels, record
    else:
        return cutoff_levels
        
    
def infinite_horizon_MC_cutoffs(rate_data, p0, p1, alpha_levels, beta_levels, 
                                n_periods, k_reps, pair_iters=10, 
                                hyp_type=None, dbg=False):
    """Calculate finite horizon rejective cutoffs from alpha levels using MC sim.
    
    Number of simulations should exceed the inverse of the min diff of the 
    alpha vector, otherwise you might end up with the same cutoff values for 
    multiple levels.
    args:
        rate_data: vector of emission rates
        p0: null p
        p1: alt p
        alpha_levels: type 1 levels for which cutoffs should be calculated
        n_periods: finite horizon
        k_reps: number of MC sims for
    return:
        array of cutoff values which should be > 0
        
    """
    drr = Series(rate_data)
    
    
    rej_max = 0.0
    acc_min = 0.0
    # Run simulation k_reps times
    for pair_num in trange(pair_iters, desc="Pairs of rej/acc"):
        rej_record = []        
        for mc_it_number in trange(k_reps, desc="Rej MC cutoff simulations",
                                 leave=False):
            # Simulate n_periods worth of data under the null with the provided 
            # rates.
            if (hyp_type is None) or (hyp_type=="drug"):
                sim_amnesia_reactions, sim_nonamnesia_reactions = simulate_reactions(
                    p0 * drr, (1.0 - p0) * drr, n_periods)
            
                # Compute llr paths for data
                llr = data_funcs.assemble_drug_llr((sim_amnesia_reactions, 
                                                sim_nonamnesia_reactions), p0, p1)
            elif hyp_type == "binom":
                amnesia = data_funcs.simulate_binom(pandas.Series(p0 * ones(len(drr)), index=drr.index), n_periods)
                llr = data_funcs.assemble_binom_llr(amnesia, p0, p1)
            elif hyp_type == "pois":
                amnesia = data_funcs.simulate_pois(pandas.Series(p0 * ones(len(drr)), index=drr.index), n_periods)
                llr = data_funcs.assemble_pois_llr(amnesia, p0, p1)
            elif hyp_type == "gaussian":
                amnesia = data_funcs.simulate_gaussian_noncum(pandas.Series(p0 * ones(len(drr)), index=drr.index), 
                                                              drr, n_periods)
                llr = data_funcs.assemble_gaussian_llr(amnesia, p0, p1)
            else:
                raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
                
                
                
            llr[llr < acc_min] = 0.0
            # Record the max value the each llr path reached
            rej_record.append(llr.max(0))
        
        # Get the cutoffs for each individual stream.
        rej_stream_specific_cutoff_levels = percentile(array(rej_record), 100 * (1 - alpha_levels), axis=0)
        # Take the max across streams for each cutoff.
        # Take A_j = max_i A_j^i, then 
        # P(Lambda^i > A_j) < P(Lambda^i > A_j^i) < alpha_j    
        rej_cutoff_levels = rej_stream_specific_cutoff_levels.max(1)
        rej_max = rej_cutoff_levels.max()
        
        acc_record = []        
        for mc_it_number in tqdm(range(k_reps), desc="Acc MC cutoff simulations",
                                 leave=False):
            # Simulate n_periods worth of data under the null with the provided 
            # rates.
            if (hyp_type is None) or (hyp_type=="drug"):
                sim_amnesia_reactions, sim_nonamnesia_reactions = simulate_reactions(
                    p1 * drr, (1.0 - p1) * drr, n_periods)
            
                # Compute llr paths for data
                llr = data_funcs.assemble_drug_llr((sim_amnesia_reactions, 
                                                sim_nonamnesia_reactions), p0, p1)
            elif hyp_type == "binom":
                amnesia = data_funcs.simulate_binom(pandas.Series(p1 * ones(len(drr)), index=drr.index), n_periods)
                llr = data_funcs.assemble_binom_llr(amnesia, p0, p1)
            elif hyp_type == "pois":
                amnesia = data_funcs.simulate_pois(pandas.Series(p1 * ones(len(drr)), index=drr.index), n_periods)
                llr = data_funcs.assemble_pois_llr(amnesia, p0, p1)
            elif hyp_type == "gaussian":
                amnesia = data_funcs.simulate_gaussian_noncum(pandas.Series(p1 * ones(len(drr)), index=drr.index), 
                                                              drr, n_periods)
                llr = data_funcs.assemble_gaussian_llr(amnesia, p0, p1)
            else:
                raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
                
            llr[llr > rej_max] = 0.0
            # Record the max value the each llr path reached
            acc_record.append(llr.min(0))
        
        # Get the cutoffs for each individual stream.
        acc_stream_specific_cutoff_levels = percentile(array(acc_record), 100 * beta_levels, axis=0)
        # Take the max across streams for each cutoff.
        # Take A_j = max_i A_j^i, then 
        # P(Lambda^i > A_j) < P(Lambda^i > A_j^i) < alpha_j    
        acc_cutoff_levels = acc_stream_specific_cutoff_levels.min(1)
        acc_max = acc_cutoff_levels.min()
    
    if dbg:
        return cutoff_levels, record
    else:
        return rej_cutoff_levels, acc_cutoff_levels

        
def llr_term_moments(drr, p0, p1):
    term_mean = drr * (p0 * log(p1/p0) + (1-p0)*log((1-p1)/(1-p0)))
    term_var = drr * (p0 * log(p1/p0)**2.0 + (1-p0)*log((1-p1)/(1-p0))**2.0)
    return pandas.DataFrame({"term_mean":term_mean, "term_var":term_var})


def llr_binom_term_moments(p0, p1):
    const_a = log((1-p1)/(1-p0))
    const_b = log(p1/p0) - log((1-p1)/(1-p0))
    eX = p0
    varX = p0 * (1 -p0)
    term_mean = const_a + const_b * eX
    term_var = (const_b ** 2.0) * varX
    return pandas.Series({"term_mean":term_mean, "term_var":term_var})
    

def llr_pois_term_moments(lam0, lam1):
    const_a = -(lam1 - lam0)
    const_b = log(lam1/ lam0)
    eX = lam0
    varX = lam0
    term_mean = const_a + const_b * eX
    term_var = (const_b ** 2.0) * varX
    return pandas.Series({"term_mean":term_mean, "term_var":term_var})

    
#Var(aX+bY) = a**2 VarX + b**2 VarY + 2ab CovXY
#def est_sample_size(alpha, beta, drr, p0, p1, BH=True):
#    N_drugs = len(drr)  
#    # First come up with p-value cutoffs
#    # BH
#    if BH:
#        alpha_vec_raw = alpha * arange(1, 1+N_drugs) / float(N_drugs)
#    # Holm
#    else:        
#        alpha_vec_raw = alpha / (float(N_drugs) - arange(N_drugs))
#    
#    alpha_vec = create_fdr_controlled_alpha(alpha, alpha_vec_raw)
#    beta_vec = create_fdr_controlled_alpha(beta, alpha_vec_raw)
#    A_vec, B_vec = calculate_mult_sprt_cutoffs(alpha_vec, beta_vec)
#    return (max(A_vec * alpha_vec[::-1]) + max(-B_vec * beta_vec[::-1])) / abs(llr_term_moments(drr, p0, p1)["term_mean"]).max()

def est_sample_size(A_vec, B_vec, drr, p0, p1, hyp_type="drug"):
    if (hyp_type is None) or (hyp_type=="drug"):
        mu_0 = (llr_term_moments(drr, p0, p1)["term_mean"]).min()
        mu_1 = (-llr_term_moments(drr, p1, p0)["term_mean"]).min()
    elif hyp_type == "pois":
        mu_0 = llr_pois_term_moments(p0, p1)["term_mean"]
        mu_1 = llr_pois_term_moments(p1, p0)["term_mean"]
    elif hyp_type == "binom":
        mu_0 = llr_binom_term_moments(p0, p1)["term_mean"]
        mu_1 = llr_binom_term_moments(p1, p0)["term_mean"]
    else:
        raise ValueError("Unknown type {0}".format(hyp_type))
        
    def e1n(A, B, mu_1):
        return (B * exp(B) * (exp(A) - 1) + A * exp(A) * (1 - exp(B))) / ((exp(A) - exp(B)) * mu_1)
    
    def e0n(A, B, mu_0):
        return (B * (exp(A) - 1) + A * (1 - exp(B))) / ((exp(A) - exp(B)) * mu_0)

    return max((e1n(A_vec[0], B_vec[0], mu_1), e0n(A_vec[0], B_vec[0], mu_0)))