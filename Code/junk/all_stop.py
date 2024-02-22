#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  7 17:31:51 2017

@author: mhankin
"""

from pylab import *
from utils import data_funcs
import pandas
from tqdm import tqdm
import warnings

def zero_check(llrvec, kpos, poscut, kneg, negcut):
    posvals = llrvec[llrvec>0].values
    posvals.sort()
    negvals = -(llrvec[llrvec<0].values)
    negvals.sort()
    possum = posvals[:kpos].sum()
    negsum = negvals[:kneg].sum()
    print(possum, negsum, kpos, kneg)
    return (possum > poscut) & (negsum > negcut)

def pos_check(posvals, negvals, lnum, poscut, negcut, posgam, neggam):
    
    npos = len(posvals)
    nneg = len(negvals)
    k1 = int(ceil(posgam * (npos + lnum)))
    k2 = int(ceil(neggam * (nneg - lnum)))
    if (k1 <= lnum):
        return False
     if (nneg <= lnum):
         warnings.warn("nneg <= l", RuntimeWarning)
    
    possum = posvals[:(k1 - lnum)].sum()
    negsum = negvals[lnum:(k2 + lnum)].sum()
    
    term_cond = (possum > poscut) & (negsum > negcut)
#     print(possum, negsum, term_cond)
    return term_cond

def main_check(llrvec, poscut, negcut, posgam, neggam, pad_infinite=True):
    npos = (llrvec>0).sum()
    nneg = (llrvec<0).sum()
    posvals = llrvec[llrvec>0].values
    posvals.sort()
    npos = len(posvals)
    negvals = -(llrvec[llrvec<0].values)
    negvals.sort()
    nneg = len(negvals)
    if pad_infinite:
        posvals_temp = posvals
        negvals_temp = negvals
        posvals = np.inf * ones(npos + nneg)
        negvals = np.inf * ones(npos + nneg)
        posvals[:npos] = posvals_temp
        negvals[:nneg] = negvals_temp
        
    
    sorted_index = llrvec.sort_values(inplace=False, ascending=False).index
    for lnum in range(nneg):
        if pos_check(posvals, negvals, lnum, poscut, negcut, posgam, neggam):
            return True, sort(sorted_index[:(npos + lnum)]) #.sort(inplace=False)
    
    for lnum in range(1, npos):
        if pos_check(negvals, posvals, lnum, negcut, poscut, neggam, posgam):
            return True, sort(sorted_index[:(npos - lnum)])
    
    return False, None





def run_sim(poscut = 3.6, negcut = 7.8,
            num_null = 25, num_alt = 20,
            lam0 = 1.5, lam1 = 1.75,
            n_periods = 1000, N_reps = 1000,
            gamma_alpha = .2, gamma_beta = .4, use_tqdm=True,
            pad_infinite=True):
    
    lamvals, gt = data_funcs.assemble_fake_pois(num_null, False, lam0, lam1, num_alt)
    
    
    fpT_rec = zeros((N_reps, 3))
    
    if use_tqdm:
        rep_iter = tqdm(range(N_reps))
    else:
        rep_iter = range(N_reps)
    for jj in rep_iter:
        reacts = data_funcs.simulate_pois(lamvals, n_periods)
        llr = data_funcs.assemble_pois_llr(lam0=lam0, lam1=lam1,pois_count=reacts)
    
        ii = 0
        for idx, rowser in llr.iterrows():
            ii = ii + 1
            dostop, decision = main_check(rowser, poscut, negcut, 
                                          gamma_alpha, gamma_beta,
                                          pad_infinite=pad_infinite)
            if dostop or (ii>950):
    #             print(ii)
    #             print(decision)
                break
    
        full_set = set(llr.columns.values)
        true_signal_set = set(llr.columns[num_null:].values)
        true_noise_set = full_set.difference(true_signal_set)
    
        dec_signal_set = set(decision)
        dec_noise_set = full_set.difference(dec_signal_set)
    
        false_rejections = dec_signal_set.difference(true_signal_set)
        false_accepts = dec_noise_set.difference(true_noise_set)
    
        fdp = len(false_rejections) / len(dec_signal_set)
        fnp = len(false_accepts) / len(dec_noise_set)
    #     print(fdp, fnp)
        fpT_rec[jj, :] = array([fdp, fnp, ii])
    FDRFNR = (fpT_rec[:, :2] > array([gamma_alpha, gamma_beta])).mean(0)
    avestop = fpT_rec[:, 2].mean()
    outrec = zeros(3)
    outrec[:2] = FDRFNR
    outrec[2] = avestop
    return outrec
    
    
import traceback
def run_sim_wrapper(meta_pncuts):
    metad = meta_pncuts[0]
    if metad["jobid"] == 0:
        pncuts = tqdm(meta_pncuts[1])
    else:
        pncuts = meta_pncuts[1]
    try:
        return [pandas.Series([pncut["poscut"], pncut["negcut"]] + run_sim(**pncut).tolist(),
                               index=["poscut", "negcut", "gfdp", "gfnp", "avetime"]) for pncut in pncuts]
    except:
        print(traceback.format_exc())
    
from multiprocessing import Pool
import itertools
def domc(gama_vec, gamb_vec, workers=5,
         num_null = 25, num_alt = 20,
         lam0 = 1.5, lam1 = 1.75,
         gamma_alpha = .2, gamma_beta = .4, pad_infinite=True):
    """Perform parralelized sims over cutooff vals
    """
    p = Pool(workers)
    # grid of pairs of cutoffs
    UU, VV = meshgrid(gama_vec, gamb_vec)
    cutpair = [el for el in zip(UU.flat, VV.flat)]
    random.shuffle(cutpair) # Hopefully evenly distributes hard tasks
    
    total_tasks = len(gama_vec) * len(gamb_vec)
    tasks_per_worker = int(ceil(total_tasks / workers))
    # build list of all full sim args
    list_of_configs = [{"poscut":el[0], "negcut":el[1], "use_tqdm":False,
                        "num_null":num_null, "num_alt":num_alt,
                        "lam0":lam0, "lam1":lam1,
                        "gamma_alpha":gamma_alpha, "gamma_beta":gamma_beta
                        } for el in cutpair]
    # block groups of sims per worker
    mapargs = [({"jobid":ii}, 
                list_of_configs[(ii * tasks_per_worker):((ii + 1) * tasks_per_worker)])
            for ii in range(workers)]
    map_output = p.map(run_sim_wrapper, mapargs)
    outframe = pandas.DataFrame([el for el in itertools.chain.from_iterable(map_output)])
    outframe["lgfdp"] = outframe["gfdp"]
    outframe["lgfnp"] = outframe["gfnp"]
    return outframe
    
    
#other_param_idx = ["num_null", "num_alt", "gamma_alpha", "gamma_beta"]
#alter_params = [dict(zip(other_param_idx, el)) for el in 
#                itertools.product([25, 35], [20, 40], [.2, .3], [.3, .4]) ]
#aps = pandas.Series({"num_null":25, "num_alt":20, "gamma_alpha":.2, "gamma_beta":.4})
    
#d1 = lam1 * log(lam1 / lam0) - (lam1 - lam0)
#d0 = lam0 * log(lam1 / lam0) - (lam1 - lam0)
#pandas.Series({"poscut":d1, "negcut":abs(d0)})
#zmz = all_stop.reganal(xxx, "avetime", noconst=True)
#1.0 / (pandas.Series({"poscut":d1, "negcut":abs(d0)}) * 
#       zmz.params * 
#       pandas.Series({"poscut":gamma_alpha, "negcut":gamma_beta}))

import statsmodels.api as sm
def reganal(xxx, endo="lgfdp", exog=["poscut", "negcut"], noconst=False):
    yyy = xxx.copy()
    yyy["lgfdp"] = log(yyy["gfdp"])
    yyy["lgfnp"] = log(yyy["gfnp"])
    if isinf(yyy[endo]).sum()>0:
        yyy.loc[isinf(yyy[endo]), endo] = np.nan
    for exo in exog:
        yyy.loc[isinf(yyy[exo]), exo] = np.nan
    yyy.dropna(inplace=True)

    if noconst:
        yyy.exog = yyy[exog]
    else:
        yyy.exog = sm.add_constant(yyy[exog], prepend=False)
    yyy.endog = yyy[endo]
    try:
        mod = sm.OLS(yyy.endog, yyy.exog)
        res = mod.fit()
    except:
        print("fail")
        return yyy
    # print(x.exog)
    # print(x.endog)
    
    
    
    print(res.summary())
    return res