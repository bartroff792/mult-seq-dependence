"""Simulation functions for MultSeq.

This module contains functions for simulating data for sequential testing 
of multiple hypotheses, and executing those procedures on it.

Functions:
    simfunc: Simulates a single path of a MultSPRT procedure.
"""
from numpy import arange, diff, zeros, mod, ones, log
import numpy
import pandas
import seaborn as sns
import multseq
import visualizations
import string
from tqdm import tqdm
from . import common_funcs
from .cutoff_funcs import (finite_horizon_rejective_cutoffs, 
                                create_fdr_controlled_alpha, fdr_helper,
                                infinite_horizon_MC_cutoffs,
                                create_fdr_controlled_bl_alpha_indpt)
from .data_funcs import (simulate_reactions, assemble_drug_llr, 
                         assemble_fake_drugs, assemble_fake_binom,
                         assemble_fake_pois, assemble_fake_pois_grad,
                         assemble_fake_gaussian, generate_llr)
from . import data_funcs, cutoff_funcs
import time
import logging, traceback
import multiprocessing 
import traceback
import warnings

# TODO: fix whatever nonsense this is.
# fh = logging.FileHandler(os.path.expanduser('~/Dropbox/Research/MultSeq/MainLog.txt'))
# fh.setLevel(logging.DEBUG)

def simfunc(positive_event_rate, negative_event_rate, n_periods, p0, p1, A_B, n_reps:int, job_id: int, **kwargs):  
    out_rec = []
    if job_id==0:
        rep_iter = tqdm(range(n_reps), desc="Job 0: MC full path simulations")
    else:
        rep_iter = range(n_reps)

    for _ in rep_iter:
        positive_events, negative_events = simulate_reactions(
            positive_event_rate, 
            negative_event_rate, 
            n_periods,
            )
        llr = assemble_drug_llr((positive_events, negative_events), p0, p1)
        del positive_events
        del negative_events
        tout = multseq.modular_sprt_test(
            llr, 
            A_B[0], 
            A_B[1], 
            record_interval=100, 
            stepup=False, 
            verbose=False, 
            rejective=A_B[1] is None)
        del llr
        out_rec.append(tout[0]['drugTerminationData']["ar0"])
    return out_rec
    
def simfunc_wrapper(kwargs):
    try:
        numpy.random.seed(kwargs['job_id'])
        return simfunc(**kwargs)
    except Exception as ex:
        logger = logging.getLogger()
        logger.error(traceback.format_exc())
        return [ex]
              
#def real_data_sim(alpha, beta, undershoot_prob=.1, sim_reps = None, min_am=1, 
#              min_tot=20, do_parallel=False, n_periods=None, fin_par=True, 
#              whole_data_n_se=3.0, am_prop_pctl=(.5, .9), use_am_prop=True, sleep_time=45):
#    """Runs MultSPRT on Yellowcard data
#    
#    args:
#        alpha: (float)
#        beta: (float)
#        undershoot_prob: (float) probability of undershoot:
#                For finite horizon, effects the number of MC cutoff sims
#                For inifinte horizon, effects the artificial horizon
#        sim_reps: (int, optional) None or number of simulations to run.
#        min_am: (int) Prescreens drugs for mininum amnesia reactions.
#        min_tot: (int) Prescreens drugs for mininum total reactions.
#        do_parallel: (bool) Run simulations in parallel (Very fragile)
#        n_periods: (int)  Number of periods for finite horizon rejective.
#
#    return:
#    
#    """
#    #alpha = .1
#    #beta = .2
#    dar, dnar, meta_data = data_funcs.read_drug_data(data_funcs.gen_skew_prescreen(min_am, min_tot))
#    if use_am_prop:
#        p0, p1 = data_funcs.am_prop_percentile_p0_p1(dar, dnar, *am_prop_pctl)
#    else:
#        p0 = data_funcs.whole_data_p0(*meta_data)
#        p1 = data_funcs.whole_data_p1(*meta_data, p0=p0, n_se=n_se)
#    #dar = dar #[drug_mask]
#    #dnar = dnar #[drug_mask]  
#    drr = dar + dnar
#    N_drugs = len(drr)  
#    
#    if beta is None:
#        # Rejective
#        # Next calculate llr cutoffs
#        alpha_vec_raw = alpha * arange(1, 1+N_drugs) / float(N_drugs)
#        alpha_vec = cutoff_funcs.create_fdr_controlled_alpha(alpha, alpha_vec_raw)
#        min_alpha_diff = min(diff(alpha_vec))
#        k_reps = int(1.0 / (min_alpha_diff * undershoot_prob))
#        logging.info("Finite Horizon MC cutoff reps: {0}".format(k_reps))
#        A_B = (finite_horizon_rejective_cutoffs(drr, p0, p1, alpha_vec, 
#                                                n_periods, k_reps, do_parallel=fin_par), None)
#    else:
#        # General
#        alpha_beta, A_B = cutoff_funcs.calc_bh_alpha_and_cuts(alpha, beta, N_drugs)
#        if n_periods is None:
#            n_periods = int(cutoff_funcs.est_sample_size(A_B[0], A_B[1], (dar + dnar), p0, p1) / undershoot_prob)
#        else:
#            logging.warn("General undershoot probability ignored. Using explicit n_periods.")
#        # Calculate cutoffs
#        logging.info("{0} drugs, {1} periods".format(N_drugs, n_periods))
#        
#
#    if sim_reps:
#        
#        if do_parallel:
#            logging.info("Cutoffs calcd, parallelized sims initiating")
#            num_cpus = multiprocessing.cpu_count()
#            num_jobs = num_cpus - 1
#            pool = multiprocessing.Pool(num_cpus)
#            n_rep_list = chunk_mc(sim_reps, num_jobs)       
#            rs = pool.map_async(simfunc_wrapper, [{"dar":dar, "dnar":dnar, 
#                                    "n_periods":n_periods, "p0":p0, "p1":p1, 
#                                    "A_B":A_B, "n_reps":n_rep, "job_id":job_id} 
#                                    for job_id, n_rep in enumerate(n_rep_list)])
#            pool.close()
#            while (True):
#                if (rs.ready() and (rs._number_left == 0)): 
#                    break
#                remaining = rs._number_left
#                report_string =  "Waiting for {0} tasks to complete. ({1})".format(
#                    remaining, time.ctime(time.time()))
#                logging.info(report_string)
#                time.sleep(max((sleep_time, sim_reps / (15 * num_jobs))))
#            par_out = list(itertools.chain.from_iterable(rs.get()))
#            logging.info("par_out   {0}".format(par_out))
#            out_rec = pandas.DataFrame(par_out).reset_index(drop=True)
#            return out_rec
#        else:
#            out_rec = pandas.DataFrame(0, index=dar.index, 
#                                   columns=["ar0_" + str(u) for u in arange(sim_reps)],
#                                    dtype="object")
#
#            for i in tqdm(range(sim_reps), desc="MC full path simulations"):
#                # Generate Data
#                amnesia, nonamnesia = simulate_reactions(dar, dnar, n_periods)
#                llr = assemble_llr(amnesia, nonamnesia, p0, p1)
#                # Run test
#                tout = multseq.modular_sprt_test(llr, A_B[0], A_B[1], record_interval=100, 
#                                                 stepup=False, verbose=False, rejective=A_B[1] is None)
#                out_rec["ar0_" + str(i)] = tout[0]['drugTerminationData']["ar0"]
#        return out_rec.T
#            
#            
#    else:
#        amnesia, nonamnesia = simulate_reactions(dar, 
#                                                 dnar, 
#                                                 n_periods)
#        llr = assemble_llr(amnesia, nonamnesia, p0, p1)
#        tout = multseq.modular_sprt_test(llr, A_B[0], A_B[1], record_interval=100, 
#                                         stepup=False, verbose=False)
#        return tout
#        
        


def synth_simfunc(dar, dnar, n_periods, p0, p1, A_B, n_reps, job_id, rho, 
                  rej_hist, ground_truth, hyp_type=None, stepup=False, 
                  m1=None, rho1=None, rand_order=False, cummax=False, **kwargs): 
    rejective=A_B[1] is None
    if job_id==0:
        main_iter = tqdm(range(n_reps), desc="MC full path simulations")
        
    else:
        main_iter = range(n_reps)
        logger = logging.getLogger()
        logger.setLevel(logging.ERROR)
        
    # Return details about termination timesteps
    if rej_hist:
        rej_rec = []
        step_rec = []
        for i in main_iter:
#            llr = generate_llr(dar, dnar, n_periods, rho, hyp_type, p0, p1, 
#                               m1, rho1, rand_order=rand_order, cummax=cummax)
            if rejective:
                llr_data = generate_llr(dar, dnar, n_periods, rho, hyp_type, p0, p1, 
                               m1, rho1, rand_order=rand_order, cummax=cummax)
                dgp = data_funcs.df_dgp_wrapper(llr_data)
            else:
                dgp = data_funcs.infinite_dgp_wrapper(dict(
                        dar=dar, dnar=dnar, n_periods=n_periods, rho=rho, 
                        hyp_type=hyp_type, p0=p0, p1=p1, m1=m1, rho1=rho1, 
                        rand_order=rand_order, cummax=cummax))
            llr = data_funcs.online_data(dar.index, dgp)
            
            tout = multseq.modular_sprt_test(llr, A_B[0], A_B[1], record_interval=100, 
                                             stepup=stepup, rejective=rejective, verbose=False)
            del llr
            rej_rec.append(tout[0]['drugTerminationData']["ar0"])
            step_rec.append(tout[0]['drugTerminationData']["step"])
            
        return (pandas.DataFrame(rej_rec).reset_index(drop=True), pandas.DataFrame(step_rec).reset_index(drop=True))
    else:
        fdp_rec = pandas.DataFrame(zeros((n_reps, 4)), columns=["fdp", "fnp", "tot_rej", "tot_acc"])
        for i in main_iter:
#            llr = generate_llr(dar, dnar, n_periods, rho, hyp_type, p0, p1, 
#                               m1, rho1, rand_order=rand_order, cummax=cummax)
            if rejective:
                llr_data = generate_llr(dar, dnar, n_periods, rho, hyp_type, p0, p1, 
                               m1, rho1, rand_order=rand_order, cummax=cummax)
                dgp = data_funcs.df_dgp_wrapper(llr_data)
            else:
                dgp = data_funcs.infinite_dgp_wrapper(dict(
                        dar=dar, dnar=dnar, n_periods=n_periods, rho=rho, 
                        hyp_type=hyp_type, p0=p0, p1=p1, m1=m1, rho1=rho1, 
                        rand_order=rand_order, cummax=cummax))
            llr = data_funcs.online_data(dar.index, dgp)
            
            tout = multseq.modular_sprt_test(llr, A_B[0], A_B[1], record_interval=100, 
                                             stepup=stepup, rejective=rejective, verbose=False)
            del llr
            dtd = tout[0]["drugTerminationData"]
            fdp_rec.ix[i] = compute_fdp(dtd, ground_truth)
            if (mod(i, 100)==1) and (i>1) and (job_id==0):
                tqdm.write("Running average: \n{0}".format(fdp_rec.mean()))
        return fdp_rec       

    
def synth_simfunc_wrapper(kwargs):
    try:
        numpy.random.seed(kwargs['job_id'])
        return synth_simfunc(**kwargs)
    except Exception as ex:
        logger = logging.getLogger()
        logger.error(traceback.format_exc())
        return [ex]

        
        
def calc_sim_cutoffs(drr, alpha, beta=None, scale_fdr=True, cut_type="BL", 
                     p0=None, p1=None, stepup=False, m0_known=False,
                     m_total=None, n_periods=None, undershoot_prob=.1,
                     do_iterative_cutoff_MC_calc=False, hyp_type=None,
                     fin_par=False, fh_sleep_time=6, 
                     fh_cutoff_normal_approx=False, 
                     fh_cutoff_imp_sample=False, fh_cutoff_imp_sample_prop=1.0,
                     fh_cutoff_imp_sample_hedge=.9,
                     dbg=False, divide_cores=None, cummax=False):
    m_hyps = len(drr)
    if (cut_type=="BY") or (cut_type=="BL"):
        alpha_vec_raw = create_fdr_controlled_bl_alpha_indpt(alpha, m_hyps)
    elif cut_type=="BH":
        alpha_vec_raw = alpha * arange(1, 1+m_hyps) / float(m_hyps)
        
    # Holm
    elif cut_type=="HOLM":        
        alpha_vec_raw = alpha / (float(m_hyps) - arange(m_hyps))
    else:
        raise Exception("Not implemented yet")
        
    if scale_fdr:
        if stepup:
            scaled_alpha_vec = alpha_vec_raw / log(m_hyps)
        else:
            if m0_known:
                scaled_alpha_vec = alpha * alpha_vec_raw / cutoff_funcs.fdr_helper(alpha_vec_raw, m_total)
            else:
                scaled_alpha_vec = cutoff_funcs.create_fdr_controlled_alpha(alpha, alpha_vec_raw)
    else: 
        scaled_alpha_vec = alpha_vec_raw
        
    if beta is not None: # Infinite horizon
        if cummax:
            warnings.warn("Cummulative Max stats not implemented for infinite horzion")
        beta_vec_raw = beta * alpha_vec_raw / alpha
        if scale_fdr:
            if stepup:
                scaled_beta_vec = beta_vec_raw / log(m_hyps)
            else:
                if m0_known:
                    scaled_beta_vec = beta * beta_vec_raw / cutoff_funcs.fdr_helper(beta_vec_raw, m_total)
                else:
                    scaled_beta_vec = cutoff_funcs.create_fdr_controlled_alpha(beta, beta_vec_raw)
        else: 
            scaled_beta_vec = beta_vec_raw
            
        A_vec, B_vec = cutoff_funcs.calculate_mult_sprt_cutoffs(scaled_alpha_vec, scaled_beta_vec)
        
        if n_periods is None:
            n_periods = int(cutoff_funcs.est_sample_size(A_vec, B_vec, drr, p0, p1,
                                                         hyp_type=hyp_type) / undershoot_prob)
            
        if do_iterative_cutoff_MC_calc:
            min_alpha_diff = min(diff(scaled_alpha_vec))
            min_beta_diff = min(diff(scaled_beta_vec))
            k_reps = int(1.0/float(undershoot_prob * min((min_alpha_diff, min_beta_diff))))
            infinite_horizon_MC_cutoffs(drr, p0, p1, scaled_alpha_vec, scaled_beta_vec, 
                                n_periods, k_reps, pair_iters=3, 
                                hyp_type=hyp_type)
        
        logging.info("n_periods: {0}".format(n_periods))
    else: # Rejective
        if n_periods is None:
            n_periods = 1000
            
        # Next calculate llr cutoffs
        min_alpha_diff = min(diff(scaled_alpha_vec))
        k_reps = int(1.0/float(undershoot_prob * min_alpha_diff))
        
        
#        raise ValueError("Alpha min {0} max {1}".format(scaled_alpha_vec.min(), scaled_alpha_vec.max()))
        A_vec = finite_horizon_rejective_cutoffs(drr, p0, p1, scaled_alpha_vec, 
                                                      n_periods, k_reps, do_parallel=fin_par,
                                                      hyp_type=hyp_type, sleep_time=fh_sleep_time,
                                                      normal_approx=fh_cutoff_normal_approx,
                                                      imp_sample=fh_cutoff_imp_sample,
                                                      imp_sample_prop=fh_cutoff_imp_sample_prop,
                                                      imp_sample_hedge=fh_cutoff_imp_sample_hedge,
                                                      divide_cores=divide_cores)
#        if stepup:
#            print("-"*10)
#            print("cummax", cummax)
#            print("scaled_alpha", scaled_alpha_vec)
#            print("A_Vec", A_vec)
#            print("-"*10)
        B_vec = None
          
    if dbg:
        return A_vec, B_vec, n_periods, scaled_alpha_vec, scaled_beta_vec
    else:
        return A_vec, B_vec, n_periods
    
    
def real_data_wrapper(alpha, beta, n_periods=None, cut_type="BL", sim_reps=100, 
                      scale_fdr=True, rho=-.5, undershoot_prob=.2,
                      min_am=1, min_tot=20, am_prop_pctl=(.5, .9), record_interval=100, 
                      do_parallel=False, fin_par=True, fh_sleep_time=30, 
                      sleep_time=25, do_iterative_cutoff_MC_calc=False, 
                      stepup=False, fh_cutoff_normal_approx=False, 
                      fh_cutoff_imp_sample=True, fh_cutoff_imp_sample_prop=.5, 
                      fh_cutoff_imp_sample_hedge=.9, divide_cores=None,
                      cummax=False):
    """Runs MultSPRT on Yellowcard data
    
    args:
        alpha: (float)
        beta: (float)
        undershoot_prob: (float) probability of undershoot:
                For finite horizon, effects the number of MC cutoff sims
                For inifinte horizon, effects the artificial horizon
        sim_reps: (int, optional) None or number of simulations to run.
        min_am: (int) Prescreens drugs for mininum amnesia reactions.
        min_tot: (int) Prescreens drugs for mininum total reactions.
        do_parallel: (bool) Run simulations in parallel (Very fragile)
        n_periods: (int)  Number of periods for finite horizon rejective.

    return:
    
    """
    dar, dnar, meta_data = data_funcs.read_drug_data(data_funcs.gen_skew_prescreen(min_am, min_tot))
    p0, p1 = data_funcs.am_prop_percentile_p0_p1(dar, dnar, *am_prop_pctl)
    
    
    drr = dar + dnar
    override_data = {"dar":dar, "dnar":dnar, "drr":drr, "ground_truth":None}
    
    return synth_data_sim(alpha=alpha, beta=beta, cut_type=cut_type, 
                          record_interval=record_interval, p0=p0, p1=p1,
             n_periods=n_periods, load_data=override_data,
             sim_reps=sim_reps,  scale_fdr=scale_fdr, rho=rho,
              undershoot_prob=undershoot_prob, 
             do_parallel=do_parallel, fin_par=fin_par, 
             fh_sleep_time=fh_sleep_time, sleep_time=sleep_time, 
             do_iterative_cutoff_MC_calc=do_iterative_cutoff_MC_calc,  
             stepup=stepup,
             fh_cutoff_normal_approx=fh_cutoff_normal_approx, 
             fh_cutoff_imp_sample=fh_cutoff_imp_sample,
             fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop, 
             fh_cutoff_imp_sample_hedge=fh_cutoff_imp_sample_hedge,
             m_alt=0,
             do_viz=False, hyp_type="drug", rej_hist=True, m0_known=False, 
             m_null = 0, max_magnitude = None, interleaved=False, 
             divide_cores=divide_cores, cummax=cummax)

def synth_data_sim(alpha=.1, beta=None, cut_type="BL", record_interval=100, 
                   p0 = .05, p1 = .045,
             n_periods = None, m_null = 3, max_magnitude = 10.0,
             sim_reps=100, m0_known=False, scale_fdr=True, rho=-.5,
             interleaved=False, undershoot_prob=.2, rej_hist=False,
             do_parallel=False, fin_par=True, do_viz=False, hyp_type="drug", 
             fh_sleep_time=60, sleep_time=25, 
             do_iterative_cutoff_MC_calc=False, m_alt=None, stepup=False,
             fh_cutoff_normal_approx=False, fh_cutoff_imp_sample=True,
             fh_cutoff_imp_sample_prop=.5, fh_cutoff_imp_sample_hedge=.9,
             load_data=None, divide_cores=None, split_corr=False, rho1=None,
             rand_order=False, cummax=False):
    """Perform sequential stepdown procedure on synthetic drug data.
        
    args:
        alpha: (float)
        beta: (float, optional) if set, indicates infinite horizon general 
            procedure. If None, use finite horizon rejective.
        BH: (bool)
        record_interval: (int)
        p0: (float)
        p1: (float)
        n_periods: (int)
        m_null: (int)
        max_magnitude: (float)
        sim_reps: (int) number of times to regenerate the data path for 
            establishing average FDP.
        m0_known: (bool) if fdr-controlling scaling of the alpha cutoff vector 
            is to be performed, indicates whether to assume number of true 
            nulls is known.
        scale_fdr: (bool) indicates whether or not to scale the alpha cutoffs
            to control fdr under arbitrary joint distributions.
        rho: (float) correlation coefficient for correlated statistics
        interleaved: (bool) whether or not to interleave the true and false 
            null hypotheses
        undershoot_prob: (float) probability of undershoot:
                For finite horizon, effects the number of MC cutoff sims
                For inifinte horizon, effects the artificial horizon
    return:
    """
    if m_alt is None:
        m_alt = m_null
    m_total = m_null + m_alt
    if load_data is None:
        if (hyp_type is None) or (hyp_type=="drug"):
            dar, dnar, ground_truth = assemble_fake_drugs(max_magnitude, m_null, interleaved, p0, p1)
            drr = dar + dnar
        elif hyp_type == "binom":
            dar, ground_truth = assemble_fake_binom(m_null, interleaved, p0, p1, m_alt=m_alt)
            drr = pandas.Series(ones(len(dar)), index=dar.index)
            dnar = None
        elif hyp_type == "pois":
            dar, ground_truth = assemble_fake_pois(m_null, interleaved, p0, p1, m_alt=m_alt)
            drr = pandas.Series(ones(len(dar)), index=dar.index)
            dnar = None
        elif hyp_type == "gaussian":
            dar, dnar, ground_truth = assemble_fake_gaussian(max_magnitude, m_null, p0, p1, m_alt=m_alt)
            drr = dnar
        elif hyp_type == "pois_grad":
            dar = assemble_fake_pois_grad(m_null, p0, p1, m_alt=m_alt)
            drr = pandas.Series(ones(len(dar)), index=dar.index)
            dnar = None
            ground_truth = drr.astype(bool)
            hyp_type = "pois"
        else:
            raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
    else:
        dar = load_data["dar"]
        dnar = load_data["dnar"]
        drr = load_data["drr"]
        ground_truth = load_data["ground_truth"]
        
    # Calculate alpha cutoffs
#    print("Cut type", cut_type)
    A_vec, B_vec, n_periods = calc_sim_cutoffs(
        drr, alpha, beta=beta, scale_fdr=scale_fdr, cut_type=cut_type, 
         p0=p0, p1=p1, stepup=stepup, m0_known=m0_known,
         m_total=m_total, n_periods=n_periods, 
         undershoot_prob=undershoot_prob, hyp_type=hyp_type,
         do_iterative_cutoff_MC_calc=do_iterative_cutoff_MC_calc, 
         fin_par=fin_par, fh_sleep_time=fh_sleep_time, 
         fh_cutoff_normal_approx=fh_cutoff_normal_approx,
         fh_cutoff_imp_sample=fh_cutoff_imp_sample,
         fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop,
         fh_cutoff_imp_sample_hedge=fh_cutoff_imp_sample_hedge,
         divide_cores=divide_cores)
    # TODO: add options for scaling style
           
    rejective = B_vec is None                                      
    # Generate data
    if split_corr:
        m1 = m_alt
        #rho1 = rho1
    else:
        m1 = None
        rho1 = None
    print("rho1", rho1)
    #confirm viability
    llr = generate_llr(dar, dnar, n_periods, rho, hyp_type, p0, p1, 
                       m1, rho1, rand_order=rand_order, cummax=cummax)


    
    # Perform testing procedure
    if sim_reps:
        print("Beginning simulation for ", hyp_type)
        if do_parallel:
            logging.info("Cutoffs calcd, parallelized sims initiating")
            num_cpus = multiprocessing.cpu_count()
            num_jobs = num_cpus - 1
            if divide_cores is not None:
                num_jobs = int(num_jobs / divide_cores)
                if num_jobs < 1:
                    num_jobs = 1 
                    
            pool = multiprocessing.Pool(num_cpus)
            n_rep_list = common_funcs.chunk_mc(sim_reps, num_jobs)         
            rs = pool.map_async(synth_simfunc_wrapper, [{"dar":dar, "dnar":dnar, 
                                    "n_periods":n_periods, "p0":p0, "p1":p1, 
                                    "A_B":[A_vec, B_vec], "n_reps":n_rep, 
                                    "job_id":job_id, "rho":rho, "rej_hist":rej_hist,
                                    "ground_truth":ground_truth, "hyp_type":hyp_type,
                                    "stepup":stepup, "m1":m1, "rho1":rho1, 
                                    "rand_order":rand_order, "cummax":cummax} 
                                    for job_id, n_rep in enumerate(n_rep_list)])
            pool.close()
            while (True):
                if (rs.ready() and (rs._number_left == 0)): 
                    break
                remaining = rs._number_left
                report_string =  "Waiting for {0} tasks to complete. ({1})".format(
                    remaining, time.ctime(time.time()))
                logging.info(report_string)
                time.sleep(max((sleep_time, sim_reps / (15 * num_jobs))))
            
            if rej_hist:
                uu, vv = zip(*rs.get())
                return (pandas.concat(uu).reset_index(drop=True), 
                        pandas.concat(vv).reset_index(drop=True),
                        ground_truth)
            else:
                return pandas.concat(rs.get()).reset_index(drop=True)
                
        else:
            arg_dict = {"dar":dar, "dnar":dnar, "n_periods":n_periods, "p0":p0, 
                        "p1":p1, "A_B":[A_vec, B_vec], "n_reps":sim_reps, 
                        "job_id":0, "rho":rho, "rej_hist":rej_hist,
                        "ground_truth":ground_truth, "hyp_type":hyp_type,
                        "stepup":stepup, "m1":m1, "rho1":rho1, 
                        "rand_order":rand_order, "cummax":cummax}
            outcome_arrays = synth_simfunc_wrapper(arg_dict)
            if rej_hist:
                return (outcome_arrays[0], outcome_arrays[1], ground_truth)
            else:
                return outcome_arrays

#        if rej_hist:
#            rej_rec = []
#            step_rec = []
#            for i in tqdm(range(sim_reps), desc="MC full path simulations"):
#                amnesia, nonamnesia = simulate_correlated_reactions(dar, dnar, n_periods, rho)
#                llr = assemble_llr(amnesia, nonamnesia, p0, p1)
#                tout = multseq.modular_sprt_test(llr, A_vec, B_vec, record_interval=100, stepup=False, rejective=rejective, verbose=False)
#                rej_rec.append(tout[0]['drugTerminationData']["ar0"])
#                step_rec.append(tout[0]['drugTerminationData']["step"])
#                
#            return (pandas.DataFrame(rej_rec).reset_index(drop=True), pandas.DataFrame(step_rec).reset_index(drop=True))
#        else:
#            fdp_rec = pandas.DataFrame(zeros((sim_reps, 4)), columns=["fdp", "fnp", "tot_rej", "tot_acc"])
#            for i in tqdm(range(sim_reps), desc="MC full path simulations"):
#                amnesia, nonamnesia = simulate_correlated_reactions(dar, dnar, n_periods, rho)
#                llr = assemble_llr(amnesia, nonamnesia, p0, p1)
#                tout = multseq.modular_sprt_test(llr, A_vec, B_vec, record_interval=100, stepup=False, rejective=rejective, verbose=False)
#                dtd = tout[0]["drugTerminationData"]
#                fdp_rec.ix[i] = multseq.compute_fdp(dtd, ground_truth)
#                if (mod(i, 100)==1) and (i>1):
#                    tqdm.write("Running average: \n{0}".format(fdp_rec.mean()))
#            return fdp_rec
            
    else:
        
        if do_viz:
            viz_stuff = visualizations.plot_multseq_llr(llr.copy(), A_vec, B_vec, ground_truth, 
                                                    verbose=False if sim_reps else True, 
                                                    stepup=stepup, jitter_mag=.01)    
            return (multseq.modular_sprt_test(llr, A_vec, B_vec, record_interval=100, stepup=stepup, rejective=True), 
                    llr, 
                    viz_stuff)
        else:
            return multseq.modular_sprt_test(llr, A_vec, B_vec, record_interval=100, stepup=stepup, rejective=True), llr
        
def compute_fdp(dtd, ground_truth):
    """Computes FDP and FNP of testing procedure output.
    
    args:
        dtd: pandas dataframe with ar0 column. Entries should be "acc", "rej",
            or NaN
        ground_truth: pandas series with drug names as index and boolean values.
            True nulls should be True. False nulls should be False.
    return:
        4-tuple: fdp (float), fnp (float), num rejected (int), num acc (int)
    """
    num_accepted = (dtd['ar0'] == "acc").sum()
    num_rejected = (dtd['ar0'] == "rej").sum()
    num_false_accepts = ((dtd['ar0'] == "acc")[~ground_truth]).sum()
    num_false_rejects = ((dtd['ar0'] == "rej")[ground_truth]).sum()
    if num_rejected>0:
        fdp_level = float(num_false_rejects) / float(num_rejected)
    else:
        fdp_level = 0
    if num_accepted>0:
        fnp_level = float(num_false_accepts) / float(num_accepted)
    else:
        fnp_level = 0
    return fdp_level, fnp_level, num_rejected, num_accepted