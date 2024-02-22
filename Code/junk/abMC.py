# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 18:14:23 2016

@author: mike
"""
from utils import cutoff_funcs, data_funcs
from pylab import *
import string
import pandas
import argparse

parser = argparse.ArgumentParser(description='MC infinite horizon cutoffs.')
parser.add_argument('--p0', type=float, default=.91)
parser.add_argument('--p1', type=float, default=.37 )
parser.add_argument('--pairs', type=int, default=3)
parser.add_argument('--reps', type=int, default=5000)


args = parser.parse_args()

from utils.data_funcs import assemble_fake_drugs

import traceback
import warnings
import sys

def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    traceback.print_stack()
    log = file if hasattr(file,'write') else sys.stderr
    log.write(warnings.formatwarning(message, category, filename, lineno, line))
    #raise Exception("Blah!")

warnings.showwarning = warn_with_traceback


try:
    get_ipython()
    dontcare = p0 +p1 + pair_iters
    print("Params set in IPython")
except:
    
    #p0 = .1
    #p1 = .378
    p0 = args.p0 #.91
    p1 = args.p1 # .37
    pair_iters = args.pairs # 3 # 12
    k_reps = args.reps
    print("Params set to p0 {0} p1 {1} pair_iters {2}".format(p0, p1, pair_iters))
max_rate = 1.0
m_null = 3
dar, dnar, ground_truth = assemble_fake_drugs(max_rate, m_null, False, p0, p1)
alpha_levels = arange(1,7, dtype=float) * .15 /(2 * m_null)
beta_levels = 0.25  / arange((2 * m_null), 0, -1, dtype=float)

drr = dar+dnar


A_B_wald = cutoff_funcs.calculate_mult_sprt_cutoffs(alpha_levels, beta_levels)
#A_B_mc = cutoff_funcs.infinite_horizon_MC_cutoffs(drr, p0, p1, alpha_levels, beta_levels, inf_horizon_steps, k_reps)
est_steps = cutoff_funcs.est_sample_size(A_B_wald[0], A_B_wald[1], drr, p0, p1)
#print("Est num steps: {0}".format(est_steps))


from tqdm import trange
rej_max = np.inf
acc_min = -np.inf

n_periods = max((500, 5.0 * est_steps))
print("n_periods ", n_periods)
# Run simulation k_reps times
rej_meta_rec = []
acc_meta_rec = []

import traceback
def MC_job_func_wrapper(kwargs):
    try:    
        return MC_job_func(**kwargs)
    except:
        traceback.print_exc()
        
#raise Exception()
    
def MC_job_func(p0, p1, drr, n_periods, k_reps, alt_bound, label, job_id):
    if job_id==0:
        rej_record = []        
        for mc_it_number in trange(k_reps, desc="{0} MC cutoff simulations {1}".format(label, pair_num)):
            # Simulate n_periods worth of data under the null with the provided 
            # rates.
            sim_amnesia_reactions, sim_nonamnesia_reactions = data_funcs.simulate_reactions(
                p0 * drr, (1.0 - p0) * drr, n_periods)
        
            # Compute llr paths for data
            rej_static_binom_llr = data_funcs.assemble_llr(sim_amnesia_reactions, 
                                            sim_nonamnesia_reactions, p0, p1)
            active_mask = (rej_static_binom_llr > alt_bound).astype(float).cumprod()
            screened_llr = rej_static_binom_llr * active_mask
            if alt_bound > -np.inf:
                 screened_llr = screened_llr + alt_bound * (1.0 - active_mask)
            # Record the max value the each llr path reached
            rej_record.append(screened_llr.max(0))
    else:
        rej_record = []        
        for mc_it_number in range(k_reps):
            # Simulate n_periods worth of data under the null with the provided 
            # rates.
            sim_amnesia_reactions, sim_nonamnesia_reactions = data_funcs.simulate_reactions(
                p0 * drr, (1.0 - p0) * drr, n_periods)
        
            # Compute llr paths for data
            rej_static_binom_llr = data_funcs.assemble_llr(sim_amnesia_reactions, 
                                            sim_nonamnesia_reactions, p0, p1)
#            rej_static_binom_llr[rej_static_binom_llr < alt_bound] = 0.0
#            # Record the max value the each llr path reached
#            rej_record.append(rej_static_binom_llr.max(0))
            active_mask = (rej_static_binom_llr > alt_bound).astype(float).cumprod()
            screened_llr = rej_static_binom_llr * active_mask
            if alt_bound > -np.inf:
                 screened_llr = screened_llr + alt_bound * (1.0 - active_mask)
            # Record the max value the each llr path reached
            rej_record.append(screened_llr.max(0))
    return rej_record
    
import itertools, multiprocessing, time, logging
from scipy.stats import percentileofscore
def one_side_MC(k_reps, pair_num, p0, p1, drr, n_periods, alt_bound, p_levels, label):

    num_jobs = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_jobs)
    chunksize = int(k_reps / num_jobs)
    n_rep_list = [chunksize, ] * (num_jobs - 1) + [k_reps - chunksize * num_jobs,]            
    rs = pool.map_async(MC_job_func_wrapper, [{"drr":drr,
                            "n_periods":n_periods, "p0":p0, "p1":p1, 
                            "label":label, "k_reps":n_rep, "job_id":job_id,
                            "alt_bound":alt_bound} 
                            for job_id, n_rep in enumerate(n_rep_list)])
    pool.close()
    while (True):
        if (rs.ready() and (rs._number_left == 0)): 
            break
        remaining = rs._number_left
        report_string =  "Waiting for {0} tasks to complete. ({1})".format(
            remaining, time.ctime(time.time()))
        #logging.info(report_string)
        #print(report_string +"\n")
        time.sleep(.1)
    rej_record = list(itertools.chain.from_iterable(rs.get()))
    
    # TESTING CODE
    # Get the cutoffs for each individual stream.
    rej_stream_specific_cutoff_levels = percentile(array(rej_record), 100 * (1 - p_levels), axis=0)    
    # Take the max across streams for each cutoff.
    # Take A_j = max_i A_j^i, then 
    # P(Lambda^i > A_j) < P(Lambda^i > A_j^i) < alpha_j    
    rej_cutoff_levels = rej_stream_specific_cutoff_levels.max(1)
    ### percentileofscore vectorized
#    if label=="Rej":
#        print("Prop exceeding: ", (array(rej_record).flatten() > A_B_wald[0][:, newaxis]).mean(1).round(3))
#        print("Expected prop: ", p_levels.round(3))
#    else:
#        print("Prop exceeding: ", (array(rej_record).flatten() > -A_B_wald[1][:, newaxis]).mean(1).round(3))
#        print("Expected prop: ", p_levels.round(3))
#    rej_cutoff_levels = percentile(array(rej_record).flatten(), 100 * (1 - p_levels))
    return rej_cutoff_levels

for pair_num in range(pair_iters):
    rej_cutoff_levels = one_side_MC(k_reps, pair_num, p0, p1, drr, n_periods, acc_min, alpha_levels, label="Rej")
    rej_max = rej_cutoff_levels.max()
    acc_cutoff_levels =  -one_side_MC(k_reps, pair_num, p1, p0, drr, n_periods, -rej_max, beta_levels, label="Acc")
    acc_min = acc_cutoff_levels.min()
    rej_meta_rec.append(rej_cutoff_levels)
    acc_meta_rec.append(acc_cutoff_levels)
    

rej_meta_rec = array(rej_meta_rec)
acc_meta_rec = array(acc_meta_rec)

figure(1, figsize=(16, 12))
plot(rej_meta_rec, linestyle="--")
plot(acc_meta_rec)
plot(ones(2*m_null) * (pair_iters - 1), A_B_wald[0],  "ro")
plot(ones(2*m_null) * (pair_iters - 1), A_B_wald[1],  "bo")
