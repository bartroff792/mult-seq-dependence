"""Simulation functions for MultSeq.

This module contains functions for higher level simulation for sequential testing 
of multiple hypotheses (beyond just generating the observations), and 
executing the SPRT procedures on it.

Functions:
    simfunc: Simulates and runs MultSPRT procedures on drug rate data using prespecified llr cutoffs.
    simfunc_wrapper: Wrapper for simfunc to allow parallelization; takes a 
        single dictionary as an argument
    synth_simfunc: DIFF?
    synth_simfunc_wrapper: DIFF?
    calc_sim_cutoffs:
    real_data_wrapper:
    synth_data_sim:
    compute_fdp: Computes FDP and FNP of testing procedure output for a run 
        where ground truth is known.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Literal
from numpy import arange, diff, zeros, mod, ones, log
import numpy
import numpy as np
import pandas as pd
import seaborn as sns
from . import multseq
# import visualizations
# import string
from tqdm import tqdm
from . import common_funcs
from .cutoff_funcs import create_fdr_controlled_bl_alpha_indpt
from . import data_funcs
from .data_funcs import (
    generate_llr,
    check_params,
)
from . import data_funcs, cutoff_funcs
import time
import logging, traceback
import multiprocessing
import traceback
import warnings
from dataclasses import dataclass

AnalysisFuncType = Callable[[multseq.MSPRTOut, pd.Series], pd.Series]
# import xarray as xr

# TODO: fix whatever nonsense this is.
# fh = logging.FileHandler(os.path.expanduser('~/Dropbox/Research/MultSeq/MainLog.txt'))
# fh.setLevel(logging.DEBUG)


# def simfunc(
#     positive_event_rate: pd.Series,
#     negative_event_rate: pd.Series,
#     n_periods: int,
#     p0: float,
#     p1: float,
#     A_B: Tuple[np.ndarray, np.ndarray],
#     n_reps: int,
#     job_id: int,
#     **kwargs,
# ) -> List[pd.Series]:
#     """Simulates and runs MultSPRT procedures on drug rate data using prespecified cutoffs.

#     Args:
#         positive_event_rate: (pd.Series) Relevant event rate for each drug.
#         negative_event_rate: (pd.Series) Non-relevant event rate for each drug.
#         n_periods: (int) Number of periods to simulate.
#         p0: (float) Prior probability of the null hypothesis.
#         p1: (float) Prior probability of the alternative hypothesis.
#         A_B: (Tuple[np.ndarray, np.ndarray]) Cutoffs for accept/reject.
#         n_reps: (int) Number of replications to run.
#         job_id: (int) Job ID for parallelization.
#         **kwargs: (dict) Additional arguments to pass to multseq.modular_sprt_test.
#     Returns:
#         accept_reject_step_ser_list: (List[pd.Series]) List of accept/reject steps for each drug.
#     """
#     # Allocate list to collect series of accept/reject steps for each drug at each stage.
#     accept_reject_step_ser_list = []
#     # If job_id is 0, use tqdm to show progress bar, otherwise just iterate.
#     if job_id == 0:
#         rep_iter = tqdm(range(n_reps), desc="Job 0: MC full path simulations")
#     else:
#         rep_iter = range(n_reps)

#     for _ in rep_iter:
#         # {positive, negative}_events are DataFrames with drug names as columns
#         # and periods as rows.
#         positive_events, negative_events = data_funcs.simulate_reactions(
#             positive_event_rate,
#             negative_event_rate,
#             n_periods,
#         )
#         llr = data_funcs.assemble_drug_llr((positive_events, negative_events), p0, p1)
#         del positive_events
#         del negative_events
#         tout = multseq.msprt(
#             llr,
#             A_B[0],
#             A_B[1],
#             record_interval=100,
#             stepup=False,
#             verbose=False,
#             rejective=A_B[1] is None,
#         )
#         del llr
#         detailed_output = tout[0]
#         drug_termination_df = detailed_output["hypTerminationOutput"]
#         accept_reject_step_ser_list.append(drug_termination_df["ar0"])
#     return accept_reject_step_ser_list


# def simfunc_wrapper(kwargs: Dict[str, Any]) -> List[pd.Series]:
#     try:
#         numpy.random.seed(kwargs["job_id"])
#         return simfunc(**kwargs)
#     except Exception as ex:
#         logger = logging.getLogger()
#         logger.error(traceback.format_exc())
#         return [ex]


# def real_data_sim(alpha, beta, undershoot_prob=.1, sim_reps = None, min_am=1,
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
#            out_rec = pd.DataFrame(par_out).reset_index(drop=True)
#            return out_rec
#        else:
#            out_rec = pd.DataFrame(0, index=dar.index,
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

def run_mc_synth_sim_tests(
    params: Dict[str, pd.Series],
    n_periods: int,
    params0: Dict[str, pd.Series],
    params1: Dict[str, pd.Series],
    cutoff_df: pd.DataFrame,
    n_reps: int,
    rho: float,
    hyp_type=None,
    stepup=False,
    rand_order=False,
) -> List[multseq.MSPRTOut]:
    """Simulates and runs MultSPRT procedures on synthetic data using prespecified cutoffs.

    Args:
        params: (Dict[str, pd.Series]) Dictionary of parameters for the simulation. specific to hyp_type.
            For "drug" type, contains positive and negative event rates for each drug. as "dar" and "dnar".
            For "binom" type, contains the number of trials and the number of successes for each 
                drug, as "n_events" and "bin_prop".
            For "pois" type, contains the number of events for each drug, as "pois_rate".
        n_periods: (int) Number of periods to simulate.
        p0: (float) Rate of the null hypothesis.
        p1: (float) Rate of the alternative hypothesis.
        cutoff_df: (pd.DataFrame) Cutoffs for accept/reject.
        n_reps: (int) Number of replications to run.
        rho: (float) Correlation coefficient for correlated statistics.
        hyp_type: (str) Type of hypothesis to simulate.
        stepup: (bool) Whether to use the step-up procedure.
        m1: (int) Number of alternative hypotheses.
        rho1: (float) Correlation coefficient for alternative hypotheses.
        rand_order: (bool) Whether to randomize the order of the hypotheses.
        cummax: (bool) Whether to use the cumulative maximum for the alternative hypotheses.

    Returns:
        out_list: (List[multseq.MSPRTOut]) List of MultSPRT outputs.
    """
    check_params(hyp_type=hyp_type,
                 params=params,)
    rejective = "B" not in cutoff_df.columns
    main_iter = tqdm(range(n_reps), desc="MC full path simulations")
    out_list = []
    
    for i in main_iter:
        if rejective:
            llr_data, obs_data = generate_llr(
                params=params,
                n_periods=n_periods,
                rho=rho,
                hyp_type=hyp_type,
                params0=params0,
                params1=params1,
                rand_order=rand_order,
            )
            dgp = data_funcs.df_dgp_wrapper(llr_data)
        else:
            dgp = data_funcs.infinite_dgp_wrapper(
                dict(
                    params=params,
                    n_periods=n_periods,
                    rho=rho,
                    hyp_type=hyp_type,
                    params0=params0,
                    params1=params1,
                    rand_order=rand_order,
                )
            )
        hypothesis_idx = list(params.values())[0].index
        out_list.append( multseq.msprt(
            statistics=data_funcs.online_data(hypothesis_idx, dgp),
            cutoffs=cutoff_df,
            record_interval=100,
            stepup=stepup,
            rejective=rejective,
            verbose=False,
        ))
    return out_list

# def synth_simfunc(
#     dar: pd.Series,
#     dnar: pd.Series,
#     n_periods: int,
#     p0: float,
#     p1: float,
#     cutoff_df: pd.DataFrame,
#     n_reps: int,
#     job_id: int,
#     rho: float,
#     rej_hist: bool,
#     ground_truth,
#     hyp_type=None,
#     stepup=False,
#     m1=None,
#     rho1=None,
#     rand_order=False,
#     cummax=False,
#     **kwargs,
# ):
    
#     rejective = "B" not in cutoff_df.columns
#     # Handle progress logging for multi-process run
#     if job_id == 0:
#         main_iter = tqdm(range(n_reps), desc="MC full path simulations")

#     else:
#         main_iter = range(n_reps)
#         logger = logging.getLogger()
#         logger.setLevel(logging.ERROR)

#     # Return details about termination timesteps
#     if rej_hist:
#         rej_rec = []
#         step_rec = []
#         for i in main_iter:
#             if rejective:
#                 llr_data = generate_llr(
#                     dar,
#                     dnar,
#                     n_periods,
#                     rho,
#                     hyp_type,
#                     p0,
#                     p1,
#                     m1,
#                     rho1,
#                     rand_order=rand_order,
#                     cummax=cummax,
#                 )
#                 dgp = data_funcs.df_dgp_wrapper(llr_data)
#             else:
#                 dgp = data_funcs.infinite_dgp_wrapper(
#                     dict(
#                         dar=dar,
#                         dnar=dnar,
#                         n_periods=n_periods,
#                         rho=rho,
#                         hyp_type=hyp_type,
#                         p0=p0,
#                         p1=p1,
#                         m1=m1,
#                         rho1=rho1,
#                         rand_order=rand_order,
#                         cummax=cummax,
#                     )
#                 )
#             llr = data_funcs.online_data(dar.index, dgp)

#             tout = multseq.msprt(
#                 statistics=llr,
#                 cutoffs=cutoff_df,
#                 record_interval=100,
#                 stepup=stepup,
#                 rejective=rejective,
#                 verbose=False,
#             )
#             del llr
#             rej_rec.append(tout.fine_grained.hypTerminationData["ar0"])
#             step_rec.append(tout.fine_grained.hypTerminationData["step"])

#         return (
#             pd.DataFrame(rej_rec).reset_index(drop=True),
#             pd.DataFrame(step_rec).reset_index(drop=True),
#         )
#     else:
#         fdp_rec = pd.DataFrame(
#             zeros((n_reps, 4)), columns=["fdp", "fnp", "tot_rej", "tot_acc"]
#         )
#         for i in main_iter:
#             if rejective:
#                 llr_data = generate_llr(
#                     dar,
#                     dnar,
#                     n_periods,
#                     rho,
#                     hyp_type,
#                     p0,
#                     p1,
#                     m1,
#                     rho1,
#                     rand_order=rand_order,
#                     cummax=cummax,
#                 )
#                 dgp = data_funcs.df_dgp_wrapper(llr_data)
#             else:
#                 dgp = data_funcs.infinite_dgp_wrapper(
#                     dict(
#                         dar=dar,
#                         dnar=dnar,
#                         n_periods=n_periods,
#                         rho=rho,
#                         hyp_type=hyp_type,
#                         p0=p0,
#                         p1=p1,
#                         m1=m1,
#                         rho1=rho1,
#                         rand_order=rand_order,
#                         cummax=cummax,
#                     )
#                 )
#             llr = data_funcs.online_data(dar.index, dgp)

#             tout = multseq.msprt(
#                 statistics=llr,
#                 cutoffs=cutoff_df,
#                 record_interval=100,
#                 stepup=stepup,
#                 rejective=rejective,
#                 verbose=False,
#             )
#             del llr
#             fdp_rec.loc[i, :] = compute_fdp(tout.fine_grained.hypTerminationData, ground_truth)
#             if (mod(i, 100) == 1) and (i > 1) and (job_id == 0):
#                 tqdm.write("Running average: \n{0}".format(fdp_rec.mean()))
#         return fdp_rec


# def synth_simfunc_wrapper(kwargs):
#     try:
#         numpy.random.seed(kwargs["job_id"])
#         return synth_simfunc(**kwargs)
#     except Exception as ex:
#         logger = logging.getLogger()
#         logger.error(traceback.format_exc())
#         return [ex]


# def calc_sim_cutoffs(
#     theta0:float,
#     theta1:float,
#     extra_params: Dict[str, Any],
#     hyp_type: Literal["pois", "binom", "pois_grad"],
#     m_total:int,
#     alpha: float,
#     beta: Optional[float]=None,
#     error_control: Optional[Literal['fdr', 'pfdr']]='fdr',
#     cut_type: Literal["BH", "BY", "BL", "HOLM"]="BL",
#     stepup:bool=False,
#     # m0_known:bool=False,
#     m0: Optional[int]=None,
#     n_periods=None,
#     undershoot_prob=0.1,
#     do_iterative_cutoff_MC_calc=False, # what is this?
#     fin_par:bool=False,
#     fh_sleep_time=6,
#     fh_cutoff_imp_sample=False,
#     fh_cutoff_imp_sample_prop=1.0,
#     fh_cutoff_imp_sample_hedge=0.9,
#     divide_cores=None,
#     ) -> Tuple[pd.DataFrame, int]:
#     """Get the LLR cutoffs for a set of generating paramters.
    
#     Args:

#     Raises:
#         Exception: _description_
#         ValueError: _description_

#     Returns:
#         Tuple with 2 elements:
#         - A pandas DataFrame with the cutoffs columns A and B (B only appears if infinite horizon)
#             - might also have other stuff???
#         - The number of periods to run the simulation for. will just be an echo of the input if that
#             was provided. Otherwise 1000 for FH and data dependent for Infinite horizon.
#     """ 
#     if (cut_type == "BY") or (cut_type == "BL"):
#         alpha_vec_raw = create_fdr_controlled_bl_alpha_indpt(alpha, m_total)
#     elif cut_type == "BH":
#         alpha_vec_raw = alpha * arange(1, 1 + m_total) / float(m_total)

#     # Holm
#     elif cut_type == "HOLM":
#         alpha_vec_raw = alpha / (float(m_total) - arange(m_total))
#     else:
#         raise Exception("Not implemented yet")

#     if error_control=="fdr":
#         if stepup:
#             raise NotImplementedError("Stepup not implemented for FDR control")
#             scaled_alpha_vec = alpha_vec_raw / log(m_total)
#         else:
#             scaled_alpha_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(
#                     alpha, alpha_vec_raw, m0=m0,
#                 )
#     elif error_control=="pfdr":
#         if stepup:
#             raise NotImplementedError("Stepup not implemented for FDR control")
#             scaled_alpha_vec = alpha_vec_raw / log(m_total)
#         else:
#             raise NotImplementedError("pfdr not implemented yet")
#             scaled_alpha_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(
#                     alpha, alpha_vec_raw, m0=m0,
#                 )
#     elif error_control is None:
#         print("Warning! Not scaling alpha for FDR control")
#         scaled_alpha_vec = alpha_vec_raw
#     else:
#         raise ValueError(f"error_control must be 'fdr', 'pfdr', or None. Got {error_control}")

#     params0, _ = data_funcs.construct_dgp(
#             m_null=len(scaled_alpha_vec),
#             m_alt=0,
#             theta0=theta0,
#             theta1=theta1,
#             hyp_type=hyp_type,
#             extra_params=extra_params,
#             interleaved=False,
#         )
    
#     params1, _ = data_funcs.construct_dgp(
#             m_null=0,
#             m_alt=len(scaled_alpha_vec),
#             theta0=theta0,
#             theta1=theta1,
#             hyp_type=hyp_type,
#             extra_params=extra_params,
#             interleaved=False,
#         )
#     if beta is not None:  # Infinite horizon
#         beta_vec_raw = beta * alpha_vec_raw / alpha
#         if error_control=="fdr":
#             if stepup:
#                 raise NotImplementedError("Stepup not implemented for FDR control")
#                 scaled_beta_vec = beta_vec_raw / log(m_total)
#             else:
#                 scaled_beta_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(
#                         beta, beta_vec_raw, m0=m0,
#                     )
#         elif error_control=="pfdr":
#             if stepup:
#                 raise NotImplementedError("Stepup not implemented for FDR control")
#                 scaled_beta_vec = beta_vec_raw / log(m_total)
#             else:
#                 raise NotImplementedError("pfdr not implemented yet")
#                 scaled_beta_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(
#                         beta, beta_vec_raw, m0=m0,
#                     )
#         elif error_control is None:
#             print("Warning! Not scaling beta for FNR control")
#             scaled_beta_vec = beta_vec_raw
#         else:
#             raise ValueError(f"error_control must be 'fdr', 'pfdr', or None. Got {error_control}")

#         # Use Wald approximations to get from alpha and beta to A and B
#         cutoff_df = cutoff_funcs.calculate_mult_sprt_cutoffs(
#             scaled_alpha_vec, scaled_beta_vec
#         )
#         A_vec = cutoff_df["A"].values
#         B_vec = cutoff_df["B"].values

#         # In the infinite horizon case, estimate the number of periods
#         # necessary to accept or reject all candidates with some probability.
#         # First estimates expected number of periods for all hyps to terminate
#         # then uses markovs inequality to get the number of periods to get the
#         # desired bound.
#         if n_periods is None:
#             # Define \tilde{\tau}=\max_i \tau_i
#             # By Markov we have P(\tilde{\tau} > n) <= E[\tilde{\tau}]/n
#             # If we wish the LHS to be \leq \delta, we need
#             # E[\tilde{\tau}] / n <= \delta
#             # E[\tilde{\tau}] / \delta <= n
#             n_periods = int(
#                 cutoff_funcs.est_sample_size(
#                     A_vec, B_vec, params0, params1, hyp_type=hyp_type
#                 )
#                 / undershoot_prob
#             )

#         if do_iterative_cutoff_MC_calc:
#             raise NotImplementedError("Iterative cutoff MC calc not implemented. Too many bugs in this code")
#             # min_alpha_diff = min(diff(scaled_alpha_vec))
#             # min_beta_diff = min(diff(scaled_beta_vec))
#             # k_reps = int(
#             #     1.0 / float(undershoot_prob * min((min_alpha_diff, min_beta_diff)))
#             # )
#             # infinite_horizon_MC_cutoffs(
#             #     drr,
#             #     p0,
#             #     p1,
#             #     scaled_alpha_vec,
#             #     scaled_beta_vec,
#             #     n_periods,
#             #     k_reps,
#             #     pair_iters=3,
#             #     hyp_type=hyp_type,
#             # )

#         logging.info("n_periods: {0}".format(n_periods))
#     else:  # Rejective
#         if n_periods is None:
#             n_periods = 1000

#         # Next calculate llr cutoffs
#         min_alpha_diff = min(diff(scaled_alpha_vec))
#         k_reps = int(1.0 / float(undershoot_prob * min_alpha_diff))

#         #        raise ValueError("Alpha min {0} max {1}".format(scaled_alpha_vec.min(), scaled_alpha_vec.max()))
#         A_vec = cutoff_funcs.estimate_finite_horizon_rejective_llr_cutoffs(
#             params0=params0,
#             params1=params1,
#             hyp_type=hyp_type,
#             n_periods=n_periods,
#             alpha_levels=scaled_alpha_vec,
#             k_reps=k_reps,
#             imp_sample=fh_cutoff_imp_sample,
#             imp_sample_prop=fh_cutoff_imp_sample_prop,
#         )
#         # TODO: this is a hack to avoid negative values in the cutoffs. Its hacky and it could cause problems if multiple values are <0. 
#         # Instead estiamte the variance of the llr streams and use that to estimate the number of samples needed to ensure 
#         # we don't get negatie values.
#         EPS = 1e-4
#         A_vec[A_vec<=0] = EPS

#         # B_vec = None
#         cutoff_df = pd.DataFrame({"A": A_vec,})

#     return cutoff_df, n_periods


# TODO: move calc_llr_cutoffs to cutoff_funcs
def calc_llr_cutoffs(
    theta0:float,
    theta1:float,
    extra_params: Dict[str, Any],
    hyp_type: Literal["pois", "binom", "pois_grad"],
    alpha: np.ndarray,
    beta: Optional[np.ndarray]=None,
    n_periods=None,
    undershoot_prob=0.1,
    do_iterative_cutoff_MC_calc=False, # what is this?
    fh_cutoff_imp_sample=False,
    fh_cutoff_imp_sample_prop=1.0,
    fh_cutoff_imp_sample_hedge: Optional[float]=None,
    ) -> Tuple[pd.DataFrame, int]:
    """Get the LLR cutoffs for a set of generating paramters.
    
    Args:

    Raises:
        Exception: _description_
        ValueError: _description_

    Returns:
        Tuple with 2 elements:
        - A pandas DataFrame with the cutoffs columns A and B (B only appears if infinite horizon)
            - might also have other stuff???
        - The number of periods to run the simulation for. will just be an echo of the input if that
            was provided. Otherwise 1000 for FH and data dependent for Infinite horizon.
    """ 
    params0, _ = data_funcs.construct_dgp(
            m_null=len(alpha),
            m_alt=0,
            theta0=theta0,
            theta1=theta1,
            hyp_type=hyp_type,
            extra_params=extra_params,
            interleaved=False,
        )
    
    params1, _ = data_funcs.construct_dgp(
            m_null=0,
            m_alt=len(alpha),
            theta0=theta0,
            theta1=theta1,
            hyp_type=hyp_type,
            extra_params=extra_params,
            interleaved=False,
        )
    if beta is not None:  # Infinite horizon
        
        # Use Wald approximations to get from alpha and beta to A and B
        cutoff_df = cutoff_funcs.calculate_mult_sprt_cutoffs(
            alpha, beta
        )
        A_vec = cutoff_df["A"].values
        B_vec = cutoff_df["B"].values

        # In the infinite horizon case, estimate the number of periods
        # necessary to accept or reject all candidates with some probability.
        # First estimates expected number of periods for all hyps to terminate
        # then uses markovs inequality to get the number of periods to get the
        # desired bound.
        if n_periods is None:
            # Define \tilde{\tau}=\max_i \tau_i
            # By Markov we have P(\tilde{\tau} > n) <= E[\tilde{\tau}]/n
            # If we wish the LHS to be \leq \delta, we need
            # E[\tilde{\tau}] / n <= \delta
            # E[\tilde{\tau}] / \delta <= n
            n_periods = int(
                cutoff_funcs.est_sample_size(
                    A_vec, B_vec, params0, params1, hyp_type=hyp_type
                )
                / undershoot_prob
            )

        if do_iterative_cutoff_MC_calc:
            raise NotImplementedError("Iterative cutoff MC calc not implemented. Too many bugs in this code")
           

        logging.info("n_periods: {0}".format(n_periods))
    else:  # Rejective
        if n_periods is None:
            n_periods = 1000

        # Next calculate llr cutoffs
        min_alpha_diff = min(diff(alpha))
        k_reps = int(1.0 / float(undershoot_prob * min_alpha_diff))

        #        raise ValueError("Alpha min {0} max {1}".format(scaled_alpha_vec.min(), scaled_alpha_vec.max()))
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
        # TODO: this is a hack to avoid negative values in the cutoffs. Its hacky and it could cause problems if multiple values are <0. 
        # Instead estiamte the variance of the llr streams and use that to estimate the number of samples needed to ensure 
        # we don't get negatie values.
        EPS = 1e-4
        A_vec[A_vec<=0] = EPS

        # B_vec = None
        cutoff_df = pd.DataFrame({"A": A_vec,})

    return cutoff_df, n_periods

def construct_base_pvalue_cutoffs(cut_type: Literal["BH", "BY", "BL", "HOLM"], 
                                  m_total:int, alpha: float) -> np.ndarray:
    if (cut_type == "BY") or (cut_type == "BL"):
        alpha_vec_raw = create_fdr_controlled_bl_alpha_indpt(alpha, m_total)
    elif cut_type == "BH":
        alpha_vec_raw = alpha * arange(1, 1 + m_total) / float(m_total)

    # Holm
    elif cut_type == "HOLM":
        alpha_vec_raw = alpha / (float(m_total) - arange(m_total))
    else:
        raise Exception("Not implemented yet")
    
    return alpha_vec_raw


def construct_sim_pvalue_cutoffs(
    m_total:int,
    alpha: float,
    beta: Optional[float]=None,
    error_control: Optional[Literal['fdr', 'pfdr']]='fdr',
    cut_type: Literal["BH", "BY", "BL", "HOLM"]="BL",
    stepup:bool=False,
    m0: Optional[int]=None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Construct the pvalue cutoff vectors for a set of control levels and shapes.
    
    """ 
    if stepup:
        raise NotImplementedError("Stepup not implemented for FDR control")
    if m0 is None:
        m1 = None
    else:
        m1 = m_total - m0
    alpha_vec = construct_base_pvalue_cutoffs(cut_type, m_total, alpha)
    if beta is not None:  # Infinite horizon
        beta_vec = construct_base_pvalue_cutoffs(cut_type, m_total, beta)
    else:
        beta_vec = None

    if error_control=="fdr":

        alpha_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(
                    alpha, alpha_vec, m0=m0,
                )
        if beta is not None:
            beta_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(
                    beta, beta_vec, m0=m1,
                )
    elif error_control=="pfdr" and beta is None:
        alpha_vec = cutoff_funcs.pfdr_finite_horizon_pvalue_cutoffs(
            alpha_vec,
            alpha,
            m0=m0,
        )
    elif error_control=="pfdr" and beta is not None:
        alpha_vec, beta_vec = cutoff_funcs.pfdr_pfnr_infinite_horizon_pvalue_cutoffs(
            alpha_vec,
            beta_vec,
            pfdr=alpha,
            pfnr=beta,
            m0=m0,
        )

    return alpha_vec, beta_vec


# def real_data_wrapper(
#     alpha,
#     beta,
#     n_periods=None,
#     cut_type="BL",
#     sim_reps=100,
#     scale_fdr=True,
#     rho=-0.5,
#     undershoot_prob=0.2,
#     min_am=1,
#     min_tot=20,
#     am_prop_pctl=(0.5, 0.9),
#     record_interval=100,
#     do_parallel=False,
#     fin_par=True,
#     fh_sleep_time=30,
#     sleep_time=25,
#     do_iterative_cutoff_MC_calc=False,
#     stepup=False,
#     fh_cutoff_imp_sample=True,
#     fh_cutoff_imp_sample_prop=0.5,
#     fh_cutoff_imp_sample_hedge=0.9,
#     divide_cores=None,
#     cummax=False,
# ):
#     """Runs MultSPRT on Yellowcard data

#     args:
#         alpha: (float)
#         beta: (float)
#         undershoot_prob: (float) probability of undershoot:
#                 For finite horizon, effects the number of MC cutoff sims
#                 For inifinte horizon, effects the artificial horizon
#         sim_reps: (int, optional) None or number of simulations to run.
#         min_am: (int) Prescreens drugs for mininum amnesia reactions.
#         min_tot: (int) Prescreens drugs for mininum total reactions.
#         do_parallel: (bool) Run simulations in parallel (Very fragile)
#         n_periods: (int)  Number of periods for finite horizon rejective.

#     return:

#     """
#     dar, dnar, meta_data = data_funcs.read_drug_data(
#         data_funcs.gen_skew_prescreen(min_am, min_tot)
#     )
#     p0, p1 = data_funcs.am_prop_percentile_p0_p1(dar, dnar, *am_prop_pctl)

#     drr = dar + dnar
#     override_data = {"dar": dar, "dnar": dnar, "drr": drr, "ground_truth": None}

#     return synth_data_sim(
#         alpha=alpha,
#         beta=beta,
#         cut_type=cut_type,
#         record_interval=record_interval,
#         p0=p0,
#         p1=p1,
#         n_periods=n_periods,
#         load_data=override_data,
#         sim_reps=sim_reps,
#         scale_fdr=scale_fdr,
#         rho=rho,
#         undershoot_prob=undershoot_prob,
#         do_parallel=do_parallel,
#         fin_par=fin_par,
#         fh_sleep_time=fh_sleep_time,
#         sleep_time=sleep_time,
#         do_iterative_cutoff_MC_calc=do_iterative_cutoff_MC_calc,
#         stepup=stepup,
#         fh_cutoff_imp_sample=fh_cutoff_imp_sample,
#         fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop,
#         fh_cutoff_imp_sample_hedge=fh_cutoff_imp_sample_hedge,
#         m_alt=0,
#         do_viz=False,
#         hyp_type="drug",
#         rej_hist=True,
#         m0_known=False,
#         m_null=0,
#         max_magnitude=None,
#         interleaved=False,
#         divide_cores=divide_cores,
#         cummax=cummax,
#     )


# def synth_data_sim(
#     alpha=0.1,
#     beta=None,
#     cut_type="BL",
#     record_interval=100,
#     p0=0.05,
#     p1=0.045,
#     n_periods=None,
#     m_null=3,
#     m_alt=None,
#     max_magnitude=10.0,
#     sim_reps=100,
#     m0_known=False,
#     scale_fdr=True,
#     rho=-0.5,
#     interleaved=False,
#     undershoot_prob=0.2,
#     rej_hist=False,
#     do_parallel=False,
#     fin_par=True,
#     do_viz=False,
#     hyp_type="drug",
#     fh_sleep_time=60,
#     sleep_time=25,
#     do_iterative_cutoff_MC_calc=False,
#     stepup=False,
#     fh_cutoff_imp_sample=True,
#     fh_cutoff_imp_sample_prop=0.5,
#     fh_cutoff_imp_sample_hedge=0.9,
#     load_data=None,
#     divide_cores=None,
#     split_corr=False,
#     rho1=None,
#     rand_order=False,
#     cummax=False,
# ):
#     """Perform sequential stepdown procedure on synthetic drug data.

#     args:
#         alpha: (float)
#         beta: (float, optional) if set, indicates infinite horizon general
#             procedure. If None, use finite horizon rejective.
#         BH: (bool)
#         record_interval: (int)
#         p0: (float)
#         p1: (float)
#         n_periods: (int)
#         m_null: (int)
#         max_magnitude: (float)
#         sim_reps: (int) number of times to regenerate the data path for
#             establishing average FDP.
#         m0_known: (bool) if fdr-controlling scaling of the alpha cutoff vector
#             is to be performed, indicates whether to assume number of true
#             nulls is known.
#         scale_fdr: (bool) indicates whether or not to scale the alpha cutoffs
#             to control fdr under arbitrary joint distributions.
#         rho: (float) correlation coefficient for correlated statistics
#         interleaved: (bool) whether or not to interleave the true and false
#             null hypotheses
#         undershoot_prob: (float) probability of undershoot:
#                 For finite horizon, effects the number of MC cutoff sims
#                 For inifinte horizon, effects the artificial horizon
#     return:
#     """
#     if m_alt is None:
#         m_alt = m_null
#     m_total = m_null + m_alt
#     # Populate the dar, dnar, drr, and ground_truth data necessary for generating
#     # the synthetic observations.
#     if load_data is None:
#         if (hyp_type is None) or (hyp_type == "drug"):
#             dar, dnar, ground_truth = assemble_fake_drugs(
#                 max_magnitude, m_null, interleaved, p0, p1
#             )
#             drr = dar + dnar
#         elif hyp_type == "binom":
#             dar, ground_truth = assemble_fake_binom(
#                 m_null, interleaved, p0, p1, m_alt=m_alt
#             )
#             drr = pd.Series(ones(len(dar)), index=dar.index)
#             dnar = None
#         elif hyp_type == "pois":
#             dar, ground_truth = assemble_fake_pois(
#                 m_null, interleaved, p0, p1, m_alt=m_alt
#             )
#             drr = pd.Series(ones(len(dar)), index=dar.index)
#             dnar = None
#         elif hyp_type == "gaussian":
#             dar, dnar, ground_truth = assemble_fake_gaussian(
#                 max_magnitude, m_null, p0, p1, m_alt=m_alt
#             )
#             drr = dnar
#         elif hyp_type == "pois_grad":
#             dar = assemble_fake_pois_grad(m_null, p0, p1, m_alt=m_alt)
#             drr = pd.Series(ones(len(dar)), index=dar.index)
#             dnar = None
#             ground_truth = drr.astype(bool)
#             hyp_type = "pois"
#         else:
#             raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
#     else:
#         dar = load_data["dar"]
#         dnar = load_data["dnar"]
#         drr = load_data["drr"]
#         ground_truth = load_data["ground_truth"]

#     # Calculate the LLR cutoffs and the number of simulation steps to run.
#     # If n
#     cutoff_df, n_periods = calc_sim_cutoffs(
#         drr,
#         alpha,
#         beta=beta,
#         scale_fdr=scale_fdr,
#         cut_type=cut_type,
#         p0=p0,
#         p1=p1,
#         stepup=stepup,
#         m0=m_null if m0_known else None,
#         m_total=m_total,
#         n_periods=n_periods,
#         undershoot_prob=undershoot_prob,
#         hyp_type=hyp_type,
#         do_iterative_cutoff_MC_calc=do_iterative_cutoff_MC_calc, # what is this?
#         fin_par=fin_par,
#         fh_sleep_time=fh_sleep_time,
#         fh_cutoff_imp_sample=fh_cutoff_imp_sample,
#         fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop,
#         fh_cutoff_imp_sample_hedge=fh_cutoff_imp_sample_hedge,
#         divide_cores=divide_cores,
#     )
#     # TODO: add options for scaling style

#     rejective = "B" in cutoff_df.columns
#     # Generate data
#     if split_corr:
#         m1 = m_alt
#         # rho1 = rho1
#     else:
#         m1 = None
#         rho1 = None
#     print("rho1", rho1)
#     # confirm viability
#     llr = generate_llr(
#         dar,
#         dnar,
#         n_periods,
#         rho,
#         hyp_type,
#         p0,
#         p1,
#         m1,
#         rho1,
#         rand_order=rand_order,
#         cummax=cummax,
#     )

#     # Perform testing procedure
#     if sim_reps:
#         print("Beginning simulation for ", hyp_type)
#         if do_parallel:
#             logging.info("Cutoffs calcd, parallelized sims initiating")
#             num_cpus = multiprocessing.cpu_count()
#             num_jobs = num_cpus - 1
#             if num_jobs > 8:
#                 num_jobs = 8
#             if divide_cores is not None:
#                 num_jobs = int(num_jobs / divide_cores)
#                 if num_jobs < 1:
#                     num_jobs = 1

#             pool = multiprocessing.Pool(num_cpus)
#             n_rep_list = common_funcs.chunk_mc(sim_reps, num_jobs)
#             rs = pool.map_async(
#                 synth_simfunc_wrapper,
#                 [
#                     {
#                         "dar": dar,
#                         "dnar": dnar,
#                         "n_periods": n_periods,
#                         "p0": p0,
#                         "p1": p1,
#                         "cutoff_df": cutoff_df,
#                         "n_reps": n_rep,
#                         "job_id": job_id,
#                         "rho": rho,
#                         "rej_hist": rej_hist,
#                         "ground_truth": ground_truth,
#                         "hyp_type": hyp_type,
#                         "stepup": stepup,
#                         "m1": m1,
#                         "rho1": rho1,
#                         "rand_order": rand_order,
#                         "cummax": cummax,
#                     }
#                     for job_id, n_rep in enumerate(n_rep_list)
#                 ],
#             )
#             pool.close()
#             while True:
#                 if rs.ready() and (rs._number_left == 0):
#                     break
#                 remaining = rs._number_left
#                 report_string = "Waiting for {0} tasks to complete. ({1})".format(
#                     remaining, time.ctime(time.time())
#                 )
#                 logging.info(report_string)
#                 time.sleep(max((sleep_time, sim_reps / (15 * num_jobs))))

#             if rej_hist:
#                 uu, vv = zip(*rs.get())
#                 return (
#                     pd.concat(uu).reset_index(drop=True),
#                     pd.concat(vv).reset_index(drop=True),
#                     ground_truth,
#                 )
#             else:
#                 return pd.concat(rs.get()).reset_index(drop=True)

#         else:
#             arg_dict = {
#                 "dar": dar,
#                 "dnar": dnar,
#                 "n_periods": n_periods,
#                 "p0": p0,
#                 "p1": p1,
#                 "cutoff_df": cutoff_df,
#                 "n_reps": sim_reps,
#                 "job_id": 0,
#                 "rho": rho,
#                 "rej_hist": rej_hist,
#                 "ground_truth": ground_truth,
#                 "hyp_type": hyp_type,
#                 "stepup": stepup,
#                 "m1": m1,
#                 "rho1": rho1,
#                 "rand_order": rand_order,
#                 "cummax": cummax,
#             }
#             outcome_arrays = synth_simfunc_wrapper(arg_dict)
#             if rej_hist:
#                 return (outcome_arrays[0], outcome_arrays[1], ground_truth)
#             else:
#                 return outcome_arrays

#     #        if rej_hist:
#     #            rej_rec = []
#     #            step_rec = []
#     #            for i in tqdm(range(sim_reps), desc="MC full path simulations"):
#     #                amnesia, nonamnesia = simulate_correlated_reactions(dar, dnar, n_periods, rho)
#     #                llr = assemble_llr(amnesia, nonamnesia, p0, p1)
#     #                tout = multseq.modular_sprt_test(llr, A_vec, B_vec, record_interval=100, stepup=False, rejective=rejective, verbose=False)
#     #                rej_rec.append(tout[0]['drugTerminationData']["ar0"])
#     #                step_rec.append(tout[0]['drugTerminationData']["step"])
#     #
#     #            return (pd.DataFrame(rej_rec).reset_index(drop=True), pd.DataFrame(step_rec).reset_index(drop=True))
#     #        else:
#     #            fdp_rec = pd.DataFrame(zeros((sim_reps, 4)), columns=["fdp", "fnp", "tot_rej", "tot_acc"])
#     #            for i in tqdm(range(sim_reps), desc="MC full path simulations"):
#     #                amnesia, nonamnesia = simulate_correlated_reactions(dar, dnar, n_periods, rho)
#     #                llr = assemble_llr(amnesia, nonamnesia, p0, p1)
#     #                tout = multseq.modular_sprt_test(llr, A_vec, B_vec, record_interval=100, stepup=False, rejective=rejective, verbose=False)
#     #                dtd = tout[0]["drugTerminationData"]
#     #                fdp_rec.ix[i] = multseq.compute_fdp(dtd, ground_truth)
#     #                if (mod(i, 100)==1) and (i>1):
#     #                    tqdm.write("Running average: \n{0}".format(fdp_rec.mean()))
#     #            return fdp_rec

#     else:
#         if do_viz:
#             A_vec = cutoff_df["A"].values
#             if rejective:
#                 B_vec = None
#             else:
#                 B_vec = cutoff_df["B"].values
#             raise NotImplementedError("Viz not implemented")
#             viz_stuff = visualizations.plot_multseq_llr(
#                 llr.copy(),
#                 A_vec,
#                 B_vec,
#                 ground_truth,
#                 verbose=False if sim_reps else True,
#                 stepup=stepup,
#                 jitter_mag=0.01,
#             )
#             return (
#                 multseq.msprt(
#                     llr,
#                     A_vec,
#                     B_vec,
#                     record_interval=100,
#                     stepup=stepup,
#                     rejective=True,
#                 ),
#                 llr,
#                 viz_stuff,
#             )
#         else:
#             return (
#                 multseq.msprt(
#                     statistics=llr,
#                     cutoffs=cutoff_df,
#                     record_interval=100,
#                     stepup=stepup,
#                     rejective=True,
#                 ),
#                 llr,
#             )


def mc_sim_and_analyze_synth_data(
    alpha=0.1,
    beta=None,
    cut_type="BL",
    theta0:float=0.05,
    theta1:float=0.045,
    extra_params:Optional[Dict[str, Union[float, int]]]=None,
    n_periods=None,
    m_null=3,
    m_alt=None,
    sim_reps=100,
    m0_known=False,
    error_control: Optional[Literal['pfdr', 'fdr']]='fdr',
    rho=-0.5,
    interleaved=False,
    undershoot_prob=0.2,
    fin_par=True,
    hyp_type="drug",
    fh_sleep_time=60,
    do_iterative_cutoff_MC_calc=False,
    stepup=False,
    analysis_func: AnalysisFuncType=None,
    fh_cutoff_imp_sample=True,
    fh_cutoff_imp_sample_prop:float=0.5,
    fh_cutoff_imp_sample_hedge:float=0.9,
    load_data=None,
    divide_cores=None,
    split_corr=False,
    rho1=None,
    rand_order=False,
    cummax=False,
) -> pd.DataFrame:
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
        error_control (str): 'pfdr' or 'fdr' or None. If None, no error control 
            adjustment beyond the standard BL alpha values is performed.
        rho: (float) correlation coefficient for correlated statistics
        interleaved: (bool) whether or not to interleave the true and false
            null hypotheses
        undershoot_prob: (float) probability of undershoot:
                For finite horizon, effects the number of MC cutoff sims
                For inifinte horizon, effects the artificial horizon
    return:
    """
    assert analysis_func is not None, "analysis_func must be provided"
    assert sim_reps > 0, "sim_reps must be greater than 0"
    # Confirm that null and alternative aren't identical
    rel_base = np.max([1e-5, np.abs(theta0), np.abs(theta1)])
    rel_diff = np.abs(theta0 - theta1) / rel_base
    assert rel_diff > 1e-5, "p0 and p1 must be different"
    # If the number of alternative hypotheses is not specified, assume it is equal to the number of null hypotheses.
    if m_alt is None:
        logging.info("m_alt not specified, assuming m_alt = m_null")
        m_alt = m_null
    m_total = m_null + m_alt
    
    # Construct the DGP from the inputs
    if load_data is None:
        params, ground_truth = data_funcs.construct_dgp(
            m_null=m_null,
            m_alt=m_alt,
            theta0=theta0,
            theta1=theta1,
            hyp_type=hyp_type,
            extra_params=extra_params,
            interleaved=interleaved,
        )
        params0, _ = data_funcs.construct_dgp(
            m_null=m_null+m_alt,
            m_alt=0,
            theta0=theta0,
            theta1=theta1,
            hyp_type=hyp_type,
            extra_params=extra_params,
            interleaved=False,
        )
        params1, _ = data_funcs.construct_dgp(
            m_null=0,
            m_alt=m_null+m_alt,
            theta0=theta0,
            theta1=theta1,
            hyp_type=hyp_type,
            extra_params=extra_params,
            interleaved=False,
        )

    else:
        raise NotImplementedError("Loading data not implemented yet")
        ground_truth = load_data.pop("ground_truth")
        params = load_data

    # Calculate the LLR cutoffs and the number of simulation steps to run.
    # If n
    scaled_alpha, scaled_beta = construct_sim_pvalue_cutoffs(
        m_total=m_total,
        alpha=alpha,
        beta=beta,
        error_control=error_control,
        cut_type=cut_type,
        stepup=stepup,
        m0=m_null,
    )
    cutoff_df, n_periods = calc_llr_cutoffs(
        theta0=theta0,
        theta1=theta1,
        extra_params=extra_params,
        hyp_type=hyp_type,
        alpha=scaled_alpha,
        beta=scaled_beta,
        n_periods=n_periods,
        undershoot_prob=undershoot_prob,
        do_iterative_cutoff_MC_calc=do_iterative_cutoff_MC_calc, # what is this?
        fh_cutoff_imp_sample=fh_cutoff_imp_sample,
        fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop,
        fh_cutoff_imp_sample_hedge=fh_cutoff_imp_sample_hedge,
    )
    # TODO: add options for scaling style

    rejective = "B" in cutoff_df.columns
    # Confirm that it doesn't crash when generating the LLR
    llr, obs = generate_llr(
        params=params,
        n_periods=n_periods,
        rho=rho,
        hyp_type=hyp_type,
        params0=params0,
        params1=params1,
        rand_order=rand_order,
    )

    # Perform testing procedure
    print("Beginning simulation for ", hyp_type)
    list_of_msprtout_objs = run_mc_synth_sim_tests(
        params=params,
        n_periods=n_periods,
        params0=params0,
        params1=params1,
        cutoff_df=cutoff_df,
        n_reps=sim_reps,
        rho=rho,
        hyp_type=hyp_type,
        stepup=stepup,
        rand_order=rand_order,
        # extra_params=extra_params,
    )
    return pd.DataFrame([analysis_func(msprtout, ground_truth) for msprtout in list_of_msprtout_objs])
    
# def single_sim_synth_data(
#     alpha=0.1,
#     beta=None,
#     cut_type="BL",
#     p0=0.05,
#     p1=0.045,
#     n_periods=None,
#     m_null=3,
#     m_alt=None,
#     max_magnitude=10.0,
#     m0_known=False,
#     scale_fdr=True,
#     rho=-0.5,
#     interleaved=False,
#     undershoot_prob=0.2,
#     fin_par=True,
#     hyp_type="drug",
#     fh_sleep_time=60,
#     do_iterative_cutoff_MC_calc=False,
#     stepup=False,
#     fh_cutoff_imp_sample=True,
#     fh_cutoff_imp_sample_prop:float=0.5,
#     fh_cutoff_imp_sample_hedge:float=0.9,
#     load_data=None,
#     divide_cores=None,
#     split_corr=False,
#     rho1=None,
#     rand_order=False,
#     cummax=False,
# ) -> multseq.MSPRTOut:
#     """Simulates and runs a single multseq test on synthetic data.

#     args:
#         alpha: (float)
#         beta: (float, optional) if set, indicates infinite horizon general
#             procedure. If None, use finite horizon rejective.
#         BH: (bool)
#         record_interval: (int)
#         p0: (float)
#         p1: (float)
#         n_periods: (int)
#         m_null: (int)
#         max_magnitude: (float)
#         sim_reps: (int) number of times to regenerate the data path for
#             establishing average FDP.
#         m0_known: (bool) if fdr-controlling scaling of the alpha cutoff vector
#             is to be performed, indicates whether to assume number of true
#             nulls is known.
#         scale_fdr: (bool) indicates whether or not to scale the alpha cutoffs
#             to control fdr under arbitrary joint distributions.
#         rho: (float) correlation coefficient for correlated statistics
#         interleaved: (bool) whether or not to interleave the true and false
#             null hypotheses
#         undershoot_prob: (float) probability of undershoot:
#                 For finite horizon, effects the number of MC cutoff sims
#                 For inifinte horizon, effects the artificial horizon
#     return:
#     """
#     # If the number of alternative hypotheses is not specified, assume it is equal to the number of null hypotheses.
#     if m_alt is None:
#         logging.info("m_alt not specified, assuming m_alt = m_null")
#         m_alt = m_null
#     m_total = m_null + m_alt
#     # Populate the dar, dnar, drr, and ground_truth data necessary for generating
#     # the synthetic observations.
#     if load_data is None:
#         if (hyp_type is None) or (hyp_type == "drug"):
#             dar, dnar, ground_truth = assemble_fake_drugs(
#                 max_magnitude, m_null, interleaved, p0, p1
#             )
#             drr = dar + dnar
#         elif hyp_type == "binom":
#             dar, ground_truth = assemble_fake_binom(
#                 m_null, interleaved, p0, p1, m_alt=m_alt
#             )
#             drr = pd.Series(ones(len(dar)), index=dar.index)
#             dnar = None
#         elif hyp_type == "pois":
#             dar, ground_truth = assemble_fake_pois(
#                 m_null, interleaved, p0, p1, m_alt=m_alt
#             )
#             drr = pd.Series(ones(len(dar)), index=dar.index)
#             dnar = None
#         elif hyp_type == "gaussian":
#             dar, dnar, ground_truth = assemble_fake_gaussian(
#                 max_magnitude, m_null, p0, p1, m_alt=m_alt
#             )
#             drr = dnar
#         elif hyp_type == "pois_grad":
#             dar = assemble_fake_pois_grad(m_null, p0, p1, m_alt=m_alt)
#             drr = pd.Series(ones(len(dar)), index=dar.index)
#             dnar = None
#             ground_truth = drr.astype(bool)
#             hyp_type = "pois"
#         else:
#             raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
#     else:
#         dar = load_data["dar"]
#         dnar = load_data["dnar"]
#         drr = load_data["drr"]
#         ground_truth = load_data["ground_truth"]

#     # Calculate the LLR cutoffs and the number of simulation steps to run.
#     # If n
#     cutoff_df, n_periods = calc_sim_cutoffs(
#         drr,
#         alpha,
#         beta=beta,
#         scale_fdr=scale_fdr,
#         cut_type=cut_type,
#         p0=p0,
#         p1=p1,
#         stepup=stepup,
#         m0=m_null if m0_known else None,
#         m_total=m_total,
#         n_periods=n_periods,
#         undershoot_prob=undershoot_prob,
#         hyp_type=hyp_type,
#         do_iterative_cutoff_MC_calc=do_iterative_cutoff_MC_calc, # what is this?
#         fin_par=fin_par,
#         fh_sleep_time=fh_sleep_time,
#         fh_cutoff_imp_sample=fh_cutoff_imp_sample,
#         fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop,
#         fh_cutoff_imp_sample_hedge=fh_cutoff_imp_sample_hedge,
#         divide_cores=divide_cores,
#     )
#     # TODO: add options for scaling style

#     rejective = "B" in cutoff_df.columns
#     # TODO: WTF is going on here???
#     # Generate data
#     if split_corr:
#         m1 = m_alt
#         # rho1 = rho1
#     else:
#         m1 = None
#         rho1 = None
#     print("rho1", rho1)
#     # Confirm that it doesn't crash when generating the LLR
#     llr = generate_llr(
#         dar,
#         dnar,
#         n_periods,
#         rho,
#         hyp_type,
#         p0,
#         p1,
#         m1,
#         rho1,
#         rand_order=rand_order,
#         cummax=cummax,
#     )
#     tout = multseq.msprt(
#         statistics=llr,
#         cutoffs=cutoff_df,
#         record_interval=100,
#         stepup=stepup,
#         rejective=True,
#     )
#     tout.full_llr = llr
#     return tout

def compute_fdp(
    tout: multseq.MSPRTOut, ground_truth: pd.Series
) -> pd.Series:
    """Computes FDP and FNP of testing procedure output.

    args:
        tout: multseq.MSPRTOut
        ground_truth: pandas series with drug names as index and boolean values.
            True nulls should be True. False nulls should be False.
    return:
        pandas series with fdp, fnp, num_rejected, num_accepted, num_not_terminated,
        and avg_sample_number
    """
    dtd = tout.fine_grained.hypTerminationData
    num_accepted = (dtd["ar0"] == "acc").sum()
    num_rejected = (dtd["ar0"] == "rej").sum()
    num_false_accepts = ((dtd["ar0"] == "acc")[~ground_truth]).sum()
    num_false_rejects = ((dtd["ar0"] == "rej")[ground_truth]).sum()
    if num_rejected > 0:
        fdp_level = float(num_false_rejects) / float(num_rejected)
    else:
        fdp_level = 0
    if num_accepted > 0:
        fnp_level = float(num_false_accepts) / float(num_accepted)
    else:
        fnp_level = 0
    num_not_terminated = dtd["step"].isna().sum()
    avg_sample_number = dtd["step"].mean()
    return pd.Series([fdp_level, fnp_level, num_rejected, num_accepted, num_not_terminated, avg_sample_number], 
                     index=["fdp", "fnp", "num_rejected", "num_accepted", "num_not_terminated", "avg_sample_number"])
