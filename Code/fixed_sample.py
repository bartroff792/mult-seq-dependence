# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Dec 14 19:41:15 2016

# @author: mhankin
# """
# from pylab import *
# from scipy.stats import binom, poisson
# from scipy.special import factorial

# import multseq
# from multseq import step_down_elimination
# from utils.data_funcs import assemble_fake_drugs, simulate_correlated_reactions, assemble_llr
# from utils.cutoff_funcs import create_fdr_controlled_alpha, llr_term_moments
# from scipy.stats import norm as normal_var
# import pandas
# import numpy

# p0 = .05
# p1 = .045
# alpha = .25
# rho = -.6
# n_periods = 6
# m_null= 5
# dar, dnar, ground_truth = assemble_fake_drugs(max_magnitude=4.0, m_null=m_null, interleaved=False, p0=p0, p1=p1)
# drr = dar + dnar
# # Calculate alpha cutoffs
# N_drugs = len(drr)

# alpha_vec_raw = alpha * arange(1, 1+N_drugs) / float(N_drugs)


# #%%


# def simfunc(N_reps, n_periods, dar, dnar, rho, p0, p1, m_null, ground_truth, alpha_vec, job_id):
#     numpy.random.seed(job_id)
#     fdp_rec = zeros(N_reps)
#     fnp_rec = zeros(N_reps)
#     nrm_approx = llr_term_moments(drr, p0, p1) * n_periods
#     for i in range(N_reps):
#         amnesia_ts, nonamnesia_ts = simulate_correlated_reactions(n_periods * dar, n_periods * dnar, 2, rho)
#         llr_ts = assemble_llr(amnesia_ts, nonamnesia_ts, p0, p1)
#         amnesia = amnesia_ts.iloc[0]
#         tot_reacts = nonamnesia_ts.iloc[0] + amnesia
#         llr = llr_ts.iloc[-1]
        
#         Z_scores = (llr - nrm_approx["term_mean"]) / sqrt(nrm_approx["term_var"])
#         p_val = pandas.Series(1 - normal_var.cdf(Z_scores), index=drr.index)
#         #print(p_val.round(3))
#         rej_idx, num_rej = step_down_elimination(p_val, alpha_vec, 0, 'low')
#         #if num_rej>0:
#         #    print(i)
#         #rej_rec.loc[i, rej_idx] = 1.0
#         fdp_rec[i] = ground_truth[rej_idx].sum() / max((1, num_rej))
#         fnp_rec[i] = ((~ground_truth[-ground_truth.index.isin(rej_idx)]).sum()) / max((1, 2 * m_null - num_rej))
#     return pandas.DataFrame({"fdp":fdp_rec, "fnp":fnp_rec})

# def simfunc_wrapper(args):
#     return simfunc(*args)
    
# import multiprocessing, shelve

# pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

# nper_N = 10
# n_period_vec = (25 * (2.0 ** linspace(4.0, 10, nper_N))).astype(int)
# N_reps = 1000
# rej_rec = pandas.DataFrame(zeros((N_reps, 2 *m_null)).astype(float), columns=drr.index)
# out_dict = shelve.open("{STORAGE_DIR}/Data/fixed_sample.shelve")
# #out_dict = {}
# for alpha in [.15, .20, .25]:
#     scaled_alpha_vec = create_fdr_controlled_alpha(alpha, alpha_vec_raw)
#     fdr_vec = zeros(nper_N)
#     fnr_vec = zeros(nper_N)
#     for j in range(nper_N):
#         print(j)
#         n_periods = n_period_vec[j]
#         out_thing_raw = pool.map(simfunc_wrapper, 
#          [(N_reps, n_periods, dar, dnar, rho, p0, p1, m_null, ground_truth, scaled_alpha_vec, i) for i in range(15)])
#         out_thing = pandas.concat(out_thing_raw).reset_index(drop=True)
#     #    n_periods = n_period_vec[j]
#     #    fdp_rec = zeros(N_reps)
#     #    fnp_rec = zeros(N_reps)
#     #    for i in range(N_reps):
#     #        amnesia_ts, nonamnesia_ts = simulate_correlated_reactions(n_periods * dar, n_periods * dnar, 2, rho)
#     #        llr_ts = assemble_llr(amnesia_ts, nonamnesia_ts, p0, p1)
#     #        amnesia = amnesia_ts.iloc[0]
#     #        tot_reacts = nonamnesia_ts.iloc[0] + amnesia
#     #        llr = llr_ts.iloc[-1]
#     #        nrm_approx = llr_term_moments(drr, p0, p1) * n_periods
#     #        Z_scores = (llr - nrm_approx["term_mean"]) / sqrt(nrm_approx["term_var"])
#     #        p_val = pandas.Series(1 - normal_var.cdf(Z_scores), index=drr.index)
#     #        #print(p_val.round(3))
#     #        rej_idx, num_rej = step_down_elimination(p_val, scaled_alpha_vec, 0, 'low')
#     #        #if num_rej>0:
#     #        #    print(i)
#     #        rej_rec.loc[i, rej_idx] = 1.0
#     #        fdp_rec[i] = ground_truth[rej_idx].sum() / max((1, num_rej))
#     #        fnp_rec[i] = ((~ground_truth[-ground_truth.index.isin(rej_idx)]).sum()) / max((1, 2 * m_null - num_rej))
#         #out_thing = simfunc(N_reps, n_periods, dar, dnar, rho, p0, p1, m_null, ground_truth, 0)
        
#         fdr_vec[j] = out_thing["fdp"].mean()
#         fnr_vec[j] = out_thing["fnp"].mean()
#     out_dict["{0:.2f}".format(alpha)] = pandas.DataFrame({"fdr":fdr_vec, "fnr":fnr_vec})

# #print(out_dict)