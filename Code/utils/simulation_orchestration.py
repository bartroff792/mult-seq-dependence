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
    check_params(
        hyp_type=hyp_type,
        params=params,
    )
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
        out_list.append(
            multseq.msprt(
                statistics=data_funcs.online_data(hypothesis_idx, dgp),
                cutoffs=cutoff_df,
                record_interval=100,
                stepup=stepup,
                rejective=rejective,
                verbose=False,
            )
        )
    return out_list


# TODO: move calc_llr_cutoffs to cutoff_funcs
def calc_llr_cutoffs(
    theta0: float,
    theta1: float,
    extra_params: Dict[str, Any],
    hyp_type: Literal["pois", "binom", "drug"],
    alpha: np.ndarray,
    beta: Optional[np.ndarray] = None,
    n_periods=None,
    undershoot_prob=0.1,
    do_iterative_cutoff_MC_calc=False,  # what is this?
    fh_cutoff_imp_sample=False,
    fh_cutoff_imp_sample_prop=1.0,
    fh_cutoff_imp_sample_hedge: Optional[float] = None,
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
        cutoff_df = cutoff_funcs.calculate_mult_sprt_cutoffs(alpha, beta)
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
            raise NotImplementedError(
                "Iterative cutoff MC calc not implemented. Too many bugs in this code"
            )

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
        A_vec[A_vec <= 0] = EPS

        # B_vec = None
        cutoff_df = pd.DataFrame(
            {
                "A": A_vec,
                "alpha": alpha,
            }
        )

    return cutoff_df, n_periods


def construct_base_pvalue_cutoffs(
    cut_type: Literal["BH", "BY", "BL", "HOLM"], m_total: int, alpha: float
) -> np.ndarray:
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
    m_total: int,
    alpha: float,
    beta: Optional[float] = None,
    error_control: Optional[Literal["fdr", "pfdr"]] = "fdr",
    cut_type: Literal["BH", "BY", "BL", "HOLM"] = "BL",
    stepup: bool = False,
    m0: Optional[int] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Construct the pvalue cutoff vectors for a set of control levels and shapes."""
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

    if error_control == "fdr":

        alpha_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(
            alpha,
            alpha_vec,
            m0=m0,
        )
        if beta is not None:
            beta_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(
                beta,
                beta_vec,
                m0=m1,
            )
    elif error_control == "pfdr" and beta is None:
        alpha_vec = cutoff_funcs.pfdr_finite_horizon_pvalue_cutoffs(
            alpha_vec,
            alpha,
            m0=m0,
        )
    elif error_control == "pfdr" and beta is not None:
        alpha_vec, beta_vec = cutoff_funcs.pfdr_pfnr_infinite_horizon_pvalue_cutoffs(
            alpha_vec,
            beta_vec,
            pfdr=alpha,
            pfnr=beta,
            m0=m0,
        )

    return alpha_vec, beta_vec


def mc_sim_and_analyze_synth_data(
    alpha=0.1,
    beta=None,
    cut_type="BL",
    theta0: float = 0.05,
    theta1: float = 0.045,
    extra_params: Optional[Dict[str, Union[float, int]]] = None,
    n_periods=None,
    m_null=3,
    m_alt=None,
    sim_reps=100,
    m0_known=False,
    error_control: Optional[Literal["pfdr", "fdr"]] = "fdr",
    rho=-0.5,
    interleaved=False,
    undershoot_prob=0.2,
    fin_par=True,
    hyp_type="drug",
    fh_sleep_time=60,
    do_iterative_cutoff_MC_calc=False,
    stepup=False,
    analysis_func: AnalysisFuncType = None,
    fh_cutoff_imp_sample=True,
    fh_cutoff_imp_sample_prop: float = 0.5,
    fh_cutoff_imp_sample_hedge: float = 0.9,
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
            m_null=m_null + m_alt,
            m_alt=0,
            theta0=theta0,
            theta1=theta1,
            hyp_type=hyp_type,
            extra_params=extra_params,
            interleaved=False,
        )
        params1, _ = data_funcs.construct_dgp(
            m_null=0,
            m_alt=m_null + m_alt,
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
        do_iterative_cutoff_MC_calc=do_iterative_cutoff_MC_calc,  # what is this?
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
    return pd.DataFrame(
        [analysis_func(msprtout, ground_truth) for msprtout in list_of_msprtout_objs]
    )


def compute_fdp(tout: multseq.MSPRTOut, ground_truth: pd.Series) -> pd.Series:
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
    return pd.Series(
        [
            fdp_level,
            fnp_level,
            num_rejected,
            num_accepted,
            num_not_terminated,
            avg_sample_number,
        ],
        index=[
            "fdp",
            "fnp",
            "num_rejected",
            "num_accepted",
            "num_not_terminated",
            "avg_sample_number",
        ],
    )


def single_sim(
    alpha=0.1,
    beta=None,
    cut_type="BL",
    theta0: float = 0.05,
    theta1: float = 0.045,
    extra_params: Optional[Dict[str, Union[float, int]]] = None,
    n_periods=None,
    m_null: Optional[int]=None,
    m_alt: Optional[int]=None,
    # m0_known=False,
    error_control: Optional[Literal["pfdr", "fdr"]] = "fdr",
    rho=-0.5,
    interleaved=False,
    undershoot_prob=0.2,
    # fin_par=True,
    hyp_type="drug",
    # fh_sleep_time=60,
    do_iterative_cutoff_MC_calc=False,
    stepup=False,
    fh_cutoff_imp_sample=True,
    fh_cutoff_imp_sample_prop: float = 0.5,
    fh_cutoff_imp_sample_hedge: float = 0.9,
    load_data=None,
    # divide_cores=None,
    # split_corr=False,
    # rho1=None,
    rand_order=False,
    # cummax=False,
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
    # Confirm that null and alternative aren't identical
    rel_base = np.max([1e-5, np.abs(theta0), np.abs(theta1)])
    rel_diff = np.abs(theta0 - theta1) / rel_base
    assert rel_diff > 1e-5, "p0 and p1 must be different"
    

    # Construct the DGP from the inputs
    if load_data is None:
        # If the number of alternative hypotheses is not specified, assume it is equal to the number of null hypotheses.
        if m_alt is None:
            logging.info("m_alt not specified, assuming m_alt = m_null")
            m_alt = m_null
        m_total = m_null + m_alt
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
            m_null=m_null + m_alt,
            m_alt=0,
            theta0=theta0,
            theta1=theta1,
            hyp_type=hyp_type,
            extra_params=extra_params,
            interleaved=False,
        )
        params1, _ = data_funcs.construct_dgp(
            m_null=0,
            m_alt=m_null + m_alt,
            theta0=theta0,
            theta1=theta1,
            hyp_type=hyp_type,
            extra_params=extra_params,
            interleaved=False,
        )

    else:
        params = load_data["params"]
        ground_truth = load_data["ground_truth"]
        params0 = load_data["params0"]
        params1 = load_data["params1"]
        m_total = len(ground_truth)
        # raise NotImplementedError("Loading data not implemented yet")
        # ground_truth = load_data.pop("ground_truth")
        # params = load_data
    
    m_total = len(ground_truth)

    
    # print(f"{params=}")
    # print(f"{ground_truth=}")
    # print(f"{params0=}")
    # print(f"{params1=}")

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
        do_iterative_cutoff_MC_calc=do_iterative_cutoff_MC_calc,  # what is this?
        fh_cutoff_imp_sample=fh_cutoff_imp_sample,
        fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop,
        fh_cutoff_imp_sample_hedge=fh_cutoff_imp_sample_hedge,
    )
    # TODO: add options for scaling style


    check_params(
        hyp_type=hyp_type,
        params=params,
    )
    rejective = "B" not in cutoff_df.columns

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
            ),
            drop_old_data=False,
        )
    hypothesis_idx = list(params.values())[0].index
    test_output = multseq.msprt(
                statistics=data_funcs.online_data(hypothesis_idx, dgp),
                cutoffs=cutoff_df,
                record_interval=100,
                stepup=stepup,
                rejective=rejective,
                verbose=False,
            )
    if not rejective:
        llr_data = dgp.get_data_record()

    return llr_data, test_output