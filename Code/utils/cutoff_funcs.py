"""Functions for calculating cutoffs for significance tests.
TODO: explain the basic scheme of alpha vetors, raw alpha vectors, llr 
cutoffs, and their signs and orders

In general, you can assume that the llr cutoffs will take the form
A[0] > ... A[m-1] > 0 > B[m-1] > ... > B[0]
and that the alpha vectors will take the form
0 < alpha[0] < ... alpha[m-1] < 1
0 < beta[0] < ... beta[m-1] < 1


List of all functions in this module with one line descriptions:
* FDR values and FDR controlled pvalue adjustments
    * guo_rao_stepdown_fdr_level: Returns stepdown FDR bound for an alpha 
        vector under arbitrary joint.
    * guo_rao_scaling: Returns a constanst lambda such that using cutoffs 
        (alpha_i*lambda, ... ) control fdr at fdr_level
    * apply_fdr_controlled_alpha: Scales an alpha vector such that it 
        controls FDR for a stepdown procedure.
    * create_fdr_controlled_bh_alpha_indpt: Creates alpha vector that 
        controls FDR for stepup under independence, with BH shape.
    * create_fdr_controlled_bh_alpha: Build alpha vector that controls FDR
        for stepdown with arbitrary joint but has BH (stepup) shape.
    * create_fdr_controlled_bl_alpha_inpdt: Get FDR controlled alpha cutoffs for the 
        Benjamini-Liu stepdown procedure under independence.
    * create_fdr_controlled_bl_alpha: Get FDR controlled alpha cutoffs for the 
        Benjamini-Liu stepdown procedure
    * calc_bh_alpha_and_cuts: Calculates llr alpha and beta and cutoffs for
        sequential stepdown with BH shape alpha and beta vector.
    * calc_bl_alpha_and_cuts: Caclulates llr alpha and beta and cutoffs for
        sequential stepdown with BL shape alpha and beta vector.
    * cutoff_truncation: sketchy functions that maps negative cutoffs small 
        positive values... not sure if this is a good idea.
    * calculate_mult_sprt_cutoffs: Uses Wald approx to calculate llr cutoffs from
        type 1 and 2 error thresholds
    * get_pvalue_cutoffs: Inverts Wald approx to get type 1/2 error cutoffs from 
        llr cutoffs.
    * pfdr_pfnr_cutoffs: broken? write a unit test
* ????
    * finite_horizon_rejective_cutoffs: Calculate finite horizon rejective cutoffs 
        from alpha levels using MC for drug sim.
    * infinite_horizon_MC_cutoffs: Calculate finite horizon rejective cutoffs from 
        alpha levels using MC for drug sim.
    * finite_horizon_cutoff_simulation: Generate finite horizon sample path maxs (and weights) for 
        MC cutoff estimation. Works for drug, binom, and pois.
    * finite_horizon_cutoff_simulation_wrapper: Wraps finite_horizon_cutoff_simulation
        with all arguments wrapped into a single dictionary. Useful for parallelization.
    * empirical_quant_presim_wrapper:
* LLR moments for different types of hypotheses
    * llr_term_moments: Expectation of the llr terms for a drug sim  under the null
        hypothesis for each step.
    * llr_binom_term_moments: Calculates mean and variance of single-example 
        binomial llr terms.
    * llr_pois_term_moments: Calculates mean and variance of one step for a poisson
        llr.
    * single_hyp_sequential_expected_rejection_time: estimated termination time for
        a single hypothesis sprt when null is false.
    * single_hyp_sequential_expected_acceptance_time: estimated termination time for
        a single hypothesis sprt when null is true.
    * est_sample_size: Estimate the expected sample size needed to accept or reject 
        all hypotheses... uses worst case scenarios applied to the previous two funcs.
* Importance Sampling helper functions: calculate importance sampling weights for 
        finite horizon MC sims for cutoff calculation
    * imp_sample_drug_weight
    * imp_sample_binom_weight
    * imp_sample_pois_weight


Roadmap:
* Confirm pfdr_pfnr_cutoffs and get_pvalue_cutoffs are working right
* as with everywhere else, move any drug specific stuff into its own module,
    and leverage as much general non-drug code as possible.
* clean up imports
* pass parameters (and distribution types?) as dictionaries to avoid forcing 
    everything to use `p0` and `p1` etc.
* Fix multiprocessing to use ray instead
* imp_sample_drug_weight confirm and document poisson formula
* imp_sample_binom_weight extend for n>=1 stages
* switch the non-drug llr moments functions to handle/anticipate series inputs instead of floats.


"""

from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from scipy.special import logit, expit
import numpy as np
from numpy.typing import NDArray
import pandas as pd
import warnings
from tqdm import tqdm, trange
from . import data_funcs
from . import common_funcs
import multiprocessing
import time
import logging
import traceback
import itertools
import copy
from scipy import stats

from scipy.optimize import minimize as scipy_minimize


# FDR controlled pvalue cutoffs and related functions
FloatArray = NDArray[np.float32]
HypTypes = Literal["drug", "pois", "binom"]

# Define a type of dataframe that contains columns `alpha`, `beta`, `A`, and `B`.
# Some of those may be full of nulls, but all should be expeted to exist?
CutoffDF = pd.DataFrame


def build_cutoff_df(
    alpha: Optional[np.ndarray] = None,
    beta: Optional[np.ndarray] = None,
    A: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
):
    if alpha is None and beta is None and A is None and B is None:
        raise ValueError("must pass at least one of alpha, beta, A, and B.")
    if alpha is not None:
        assert np.all(alpha > 0.0) and np.all(
            1.0 >= alpha
        ), "Alpha must be between (0,1]"
        assert np.all(np.diff(alpha) >= 0.0), "Alpha vector is not monotone increasing."
    if beta is not None:
        assert np.all(beta > 0.0) and np.all(1.0 >= beta), "Beta must be between (0,1]"
        assert np.all(np.diff(beta) >= 0.0), "Beta vector is not monotone increasing."
    if A is not None:
        assert np.all(A > 0.0), "Rejection boundary As must be greater than 0."
        assert np.all(
            np.diff(A) <= 0.0
        ), "Rejection boundary As must be non-increasing."
    if B is not None:
        assert np.all(B < 0.0), "Acceptance boundary Bs must be less than 0."
        assert np.all(
            np.diff(B) >= 0.0
        ), "Acceptance boundary Bs must be non-decreasing."

    return CutoffDF({"alpha": alpha, "beta": beta, "A": A, "B": B})


def guo_rao_stepdown_fdr_level(
    alpha_vec: FloatArray,
    m0: Optional[int] = None,
    get_max_m0: bool = False,
) -> float:
    """Returns stepdown FDR bound for an alpha vector under arbitrary joint.

    Guo-Rao based FDR control level for a stepdown prodedure.
    Assumes alpha_vec[0] = alpha_1 <= alpha_vec[1] = alpha_2 <= alpha_vec[m-1] = alpha_m
    Args:
        alpha_vec: increasing array of p-value/significance cutoffs
        m0: number of true null hypotheses, if known, otherwise None.
        get_max_m0: If m0 is None and a search is to be performed, passing True
            to this argument will return both the max fdr and the m0 at which
            it was achieved.
    Return:
        Tight upper bound on FDR, when m0 is known.
    """
    # Verify alpha vec
    assert np.all(np.diff(alpha_vec) >= 0.0), "Alpha vector is not monotone increasing."
    assert np.all(
        0.0 < alpha_vec
    ), "Alpha vector contains values less than or equal to 0."
    if np.any(1.0 <= alpha_vec):
        warnings.warn("Alpha vector contains values greater than or equal to 1.")

    # When number of true nulls is unknown, search over all possible values.
    if m0 is None:
        fdr_vec = [
            guo_rao_stepdown_fdr_level(alpha_vec, m0, False)
            for m0 in range(1, len(alpha_vec) + 1)
        ]
        if get_max_m0:
            return np.max(fdr_vec), np.argmax(fdr_vec)
        else:
            return np.max(fdr_vec)

    # Otherwise proceed
    # Total number of hypotheses
    m = len(alpha_vec)
    # Number of false hypotheses
    m1 = m - m0
    index_vec = np.arange(1, m + 1)
    # Pad the alpha vector with 0 and take its first order difference so that
    # \alpha_{1} - \alpha_{0} = \alpha_{1}
    # Now diff_vec[i] = \alpha_{i+1} - \alpha_{i}, so to make reconciling this
    # with the formulas more efficient, we use index_vec.
    # diff_vec[i] = \alpha_{index_vec[i]} - ...
    diff_vec = np.diff(np.concatenate(([0], alpha_vec)))
    # \sum_{i=1}^{m1+1} (\alpha_i - \alpha_{i-1}) / i
    false_null_mask = index_vec <= m1 + 1
    term1 = (diff_vec / index_vec)[false_null_mask].sum()
    # \sum_{i=m1+2}^{m} m1 *(\alpha_i - \alpha_{i-1}) / (i * (i-1))
    term2 = (
        m1
        * (
            diff_vec[index_vec > m1 + 1]
            / (index_vec[index_vec > m1 + 1] * (index_vec[index_vec > m1 + 1] - 1))
        ).sum()
    )
    return m0 * (term1 + term2)


def get_guo_rao_scaling_factor(
    fdr_level: float,
    alpha_vec: np.ndarray,
    m0: Optional[int] = None,
) -> float:
    """Returns a constanst lambda such that using cutoffs (alpha_i*lambda, ... ) control fdr at fdr_level

    Args:
        fdr_level: level at which FDR must be controlled.
        alpha_vec: initial, unscaled vector of increasing alpha cutoffs.
        m0: number of true null hypotheses, if known, otherwise None.
    Returns:
        A scaling factor which, when multiplied by alpha_vec, will return a
        proportional cutoff vector that controls alpha at the correct level.
    """
    assert 0.0 < fdr_level <= 1.0, "FDR level must be between 0 and 1."
    assert np.all(
        0.0 < alpha_vec
    ), "Alpha vector contains values less than or equal to 0."
    if m0 is not None:
        assert m0 >= 0, "Number of true null hypotheses must be non-negative."
    return fdr_level / guo_rao_stepdown_fdr_level(alpha_vec, m0=m0)


def apply_fdr_control_to_alpha_vec(
    fdr_level: float,
    alpha_vec: FloatArray,
    m0: Optional[int] = None,
) -> FloatArray:
    """Scales an alpha vector such that it controls FDR for a stepdown procedure.

    Does not require independence.

    Args:
        fdr_level: level at which FDR must be controlled for stepdown procedures.
        alpha_vec: initial, unscaled vector of increasing alpha cutoffs.
        m0: number of true null hypotheses, if known, otherwise None.

    Returns:
        A vector of alpha cutoffs that control FDR at fdr_level for stepdown procedures.
    """
    assert 0.0 < fdr_level <= 1.0, "FDR level must be between 0 and 1."
    assert np.all(
        0.0 < alpha_vec
    ), "Alpha vector contains values less than or equal to 0."
    if m0 is not None:
        assert m0 >= 0, "Number of true null hypotheses must be non-negative."
    return get_guo_rao_scaling_factor(fdr_level, alpha_vec, m0=m0) * alpha_vec


def create_fdr_controlled_bh_alpha_indpt(fdr_level: float, m_hyps: int) -> FloatArray:
    """Creates alpha vector that controls FDR for stepup under independence.

    For a stepup procedure with independent hypotheses, the alpha cutoffs
    alpha_{i} = fdr_level * i / m_hyps
    will control FDR at fdr_level.

    This vector can, however, be scaled using Guo Rao to control FDR for a stepdown procedure.

    Args:
        fdr_level: level at which FDR must be controlled for a stepup procedure under independence.
            Not directly relevant otherwise.
        m_hyps: number of hypotheses.

    Returns:
        A vector of alpha cutoffs that control FDR at fdr_level.
    """
    assert 0.0 < fdr_level <= 1.0, "FDR level must be between 0 and 1."
    assert m_hyps >= 1, "Number of hypotheses must be positive."
    return fdr_level * np.arange(1, m_hyps + 1, dtype=float) / float(m_hyps)


def create_fdr_controlled_bh_alpha(fdr_level: float, m_hyps: int) -> FloatArray:
    """Build alpha vector that controls FDR for stepdown with arbitrary joint but has BH (stepup) shape.

    Ratio of alpha_{i} / alpha_{j} = i / j for all i, j.

    Args:
        fdr_level: level at which FDR must be controlled for a stepdown procedure.
        m_hyps: number of hypotheses.

    Returns:
        A vector of alpha cutoffs that control FDR at fdr_level for stepdown procedures.
    """
    alpha_vec_raw = create_fdr_controlled_bh_alpha_indpt(fdr_level, m_hyps)
    return apply_fdr_control_to_alpha_vec(fdr_level, alpha_vec_raw)


def create_fdr_controlled_bl_alpha_indpt(
    fdr_level: FloatArray, m_hyps: int, hedge: bool = True
) -> FloatArray:
    """Get FDR controlled alpha cutoffs for the Benjamini-Liu stepdown procedure w/independence.

    See Benjamini Liu (1999). Stepdown cutoffs for independent hypotheses.
    alpha_i = 1 - (1 - min(1, m * alpha / (m - i + 1))) ** (1 / (m - i + 1))

    Args:
        fdr_level: level at which FDR must be controlled for a stepdown procedure under independence.
        m_hyps: number of hypotheses.
        hedge: if True, shrinks the most significant cutoffs slightly to avoid
             sure rejects. This will induce weird biases...

    Returns:
        A vector of alpha cutoffs that control FDR at fdr_level for stepdown procedures under independence.
    """
    # Create a matrix for the cases
    tempmat = np.ones((2, m_hyps))
    tempmat[1, :] = m_hyps * fdr_level / (m_hyps - np.arange(m_hyps)).astype(float)
    casevec = tempmat.min(0)
    # Construct alpha vector
    alpha_vec = 1.0 - (1.0 - casevec) ** (
        1.0 / (m_hyps - np.arange(m_hyps)).astype(float)
    )

    # If there are low-significance cutoffs equal to 1.0 and the hedge option
    # is requested, shrink them slightly
    num_sure_rejects = (alpha_vec == 1.0).sum()
    if hedge and (num_sure_rejects > 0):
        # Choose a value to shrink the most significant cutoff thats currently
        # equal to 1.0 down to by either choosing the least significant cutoff
        # less than 1.0 or 1- .5 / m
        low1 = 1.0 - (min(1.0 - alpha_vec[-(num_sure_rejects + 1)], 1.0 / m_hyps) / 2.0)
        # Interpolate from there to 1.0, leaving off the 1.0
        oneadjvec = np.linspace(low1, 1.0, num_sure_rejects + 1)
        alpha_vec[-num_sure_rejects:] = oneadjvec[:-1]

    return alpha_vec


def create_fdr_controlled_bl_alpha(
    fdr_level: float, m_hyps: int, indpt_fdr_level: Optional[float] = None
) -> FloatArray:
    """FDR controlled alpha cutoffs for the Benjamini-Liu stepdown procedure

    Shape from Benjamini Liu (1999), scaled with Guo Rao for arbitrary joint.
    See create_fdr_controlled_bl_alpha_indpt.

    Args:
        fdr_level: level at which FDR must be controlled for a stepdown procedure.
        m_hyps: number of hypotheses.
        indpt_fdr_level: nominal level at which fdr would be controlled for a
            stepdown BL procedure under independence. Will only affect the
            shape of the alpha cutoffs, as the final scaling uses Guo+Rao.

    Returns:
        A vector of alpha cutoffs that control FDR at fdr_level for stepdown procedures.
    """
    if indpt_fdr_level is None:
        indpt_fdr_level = fdr_level
    alpha_vec_raw = create_fdr_controlled_bl_alpha_indpt(indpt_fdr_level, m_hyps)
    # print(delvec)
    return apply_fdr_control_to_alpha_vec(fdr_level, alpha_vec_raw)


def calc_bh_alpha_and_cuts(
    fdr_level: float,
    fnr_level: float,
    N_drugs: int,
) -> CutoffDF:
    """Caclulates llr alpha and beta and cutoffs for sequential stepdown with BH shape alpha and beta vector.

    Args:
        `fdr_level`: desired FDR level
        `fnr_level`: desired FNR level
        `N_drugs`: number of hypothesea to test

    Returns:
        A CutoffDF with alpha, beta, A and B columns
    """
    alpha_vec = create_fdr_controlled_bh_alpha(fdr_level, N_drugs)
    beta_vec = create_fdr_controlled_bh_alpha(fnr_level, N_drugs)
    return calculate_mult_sprt_cutoffs(alpha_vec, beta_vec)


def calc_bl_alpha_and_cuts(
    fdr_level: float,
    fnr_level: float,
    N_drugs: int,
) -> CutoffDF:
    """Caclulates llr alpha and beta and cutoffs for sequential stepdown with BL shape alpha and beta vector.

    Args:
        `fdr_level`: desired FDR level
        `fnr_level`: desired FNR level
        `N_drugs`: number of hypothesea to test

    Returns:
        2 2-tuples:
            1. (alpha_vec, beta_vec): alpha and beta cutoffs for each drug
            2. (alpha_cutoffs, beta_cutoffs): llr cutoffs for each drug
    """
    alpha_vec = create_fdr_controlled_bl_alpha(fdr_level, N_drugs)
    beta_vec = create_fdr_controlled_bl_alpha(fnr_level, N_drugs)
    return calculate_mult_sprt_cutoffs(alpha_vec, beta_vec)


def cutoff_truncation(cut_vec: FloatArray) -> FloatArray:
    """Prevents negative cutoffs."""
    cut_vec = cut_vec.copy()
    # Get least positive cutoff
    lowest_pos = cut_vec[cut_vec > 0].min()
    # count the number of negative cutoffs
    num_neg = (cut_vec <= 0).sum()
    if num_neg > 0:
        warnings.warn("Truncating cutoffs")
    # Set negative cutoffs to be equally spaced between 0 and lowest positive cutoff, in descending order.
    cut_vec[cut_vec <= 0] = np.linspace(lowest_pos, 0.0, num_neg + 2)[1:-1]
    return cut_vec


def calculate_mult_sprt_cutoffs(
    alpha_vec: FloatArray,
    beta_vec: FloatArray,
    rho: float = 0.583,
    do_trunc: bool = True,
) -> CutoffDF:
    """Uses Wald approx to calculate llr cutoffs from type 1 and 2 error thresholds.
    A > 0 > B
    Reject H0 when Lambda > A
    Accept H0 when Lambda < B

    Args:
        alpha_vec: increasing vector of type 1 error cutoffs.
        beta_vec: increasing vector of type 2 error cutoffs.
        rho: scalar adjustment factor for the Wald approximations.

    Returns:
        CutoffDF with columns alpha, beta, A, and B.
        A: positive, increasing rejection cutoffs **for log likelihood ratio**
        B: negative, decreasing acceptance cutoffs **for log likelihood ratio**
        alpha and beta: (0,1] increasing pvalue cutoffs
    """
    alpha1 = alpha_vec[0]
    beta1 = beta_vec[0]
    A_vec = (
        np.log((1 - alpha1 - beta1 * (1 - alpha_vec)) / (alpha_vec * (1 - alpha1)))
        - rho
    )
    B_vec = np.log(beta_vec * (1 - beta1) / (1 - beta1 - alpha1 * (1 - beta_vec))) + rho

    if (A_vec < 0).any():
        num_neg = (A_vec < 0).sum()

        warnings.warn(
            "{0} A_vec Cutoff levels are negative: from {1} to {2}".format(
                num_neg, A_vec.min(), A_vec.max()
            )
        )

        if do_trunc:
            A_vec = cutoff_truncation(A_vec)

    if (B_vec > 0).any():
        num_pos = (B_vec > 0).sum()

        warnings.warn(
            "{0} B_vec Cutoff levels are positive: from {1} to {2}".format(
                num_pos, B_vec.min(), B_vec.max()
            )
        )

        if do_trunc:
            B_vec = -cutoff_truncation(-B_vec)
    assert np.diff(A_vec).max() < 0, "A_vec is non-decreasing"
    assert np.diff(B_vec).min() > 0, "B_vec is non-increasing"
    return build_cutoff_df(alpha=alpha_vec, beta=beta_vec, A=A_vec, B=B_vec)


# TODO: fix this. calculating for reversed statistics
def get_pvalue_cutoffs(A_vec: np.ndarray, B_vec: np.ndarray, rho=0.583) -> CutoffDF:
    """Inverts Wald approx to get type 1/2 error cutoffs from llr cutoffs."""
    alpha_vec = np.zeros(A_vec.shape)
    beta_vec = np.zeros(B_vec.shape)
    highest_sig = np.linalg.solve(
        np.array([[np.exp(B_vec[0] + rho), 1], [np.exp(A_vec[0] - rho), 1]]),
        np.array([1, np.exp(A_vec[0] - rho)]),
    )
    alpha_vec[0] = highest_sig[0]
    beta_vec[0] = highest_sig[1]

    alpha_vec[1:] = (1 - highest_sig.sum()) / (
        (1 - highest_sig[0]) * np.exp(B_vec[1:] + rho) - highest_sig[1]
    )
    beta_vec[1:] = (1 - highest_sig.sum()) / (
        (1 - highest_sig[1]) * np.exp(-A_vec[1:] + rho) - highest_sig[0]
    )
    return build_cutoff_df(alpha=alpha_vec, beta=beta_vec, A=A_vec, B=B_vec)


def pfdr_pfnr_infinite_horizon_pvalue_cutoffs(
    alpha_raw_vec: FloatArray,
    beta_raw_vec: FloatArray,
    pfdr: float,
    pfnr: float,
    m0: Optional[int] = None,
    epsilon: float = 10.0**-8,
) -> Tuple[FloatArray, FloatArray]:
    """pFDR and pFNR controlled pvalue cutoffs for infinite horizon sequential stepdown.

    Uses an interative scheme to find pvalue cutoffs that satisfy the pfdr and
    pfnr levels that utilize the same vector structure as the raw inputs.
    See the section of the readme titled "THM: pFDR and pFNR Control for Infinite Horizon"

    Args:
        alpha_raw_vec (np.ndarray): raw vector of rejective p-value cutoffs.
            only the structure will be used, as a linear scaling will be applied.
        beta_raw_vec (np.ndarray): raw vector of acceptive p-value cutoffs.
            only the structure will be used, as a linear scaling will be applied.
        pfdr (float): desired pFDR control level
        pfnr (float): desired pFNR control level
        m0 (Optional[int]): number of true nulls, if known
        epsilon (float, optional): tolerance for convergence. Defaults to 10.0**-8.

    Returns:
        Tuple[np.ndarray, np.ndarray]: final alpha vector and beta vector.
    """
    # Total number of hypotheses
    m = len(alpha_raw_vec)
    # number of false hypotheses
    if m0 is None:
        m1 = None
    else:
        assert (m0 > 0) and (
            m0 < m
        ), "Number of true nulls must be in (0, m) for joint pFDR and pFNR control."
        m1 = m - m0
    # Raw alpha and beta vectors
    alpha_vec0 = alpha_raw_vec
    beta_vec0 = beta_raw_vec
    pvalue_cutoff_verifier(alpha_vec0, beta_vec0)
    # Get pFDR and pFNR control levels using current alpha and beta vectors
    # by first calculating FDR and FNR levels (via Guo+Rao) then scaling those
    # by the (approximate) probability of at least one rejection (or acceptance)
    pfdrx = guo_rao_stepdown_fdr_level(alpha_vec0, m0) / (1 - beta_vec0[-1])
    pfnrx = guo_rao_stepdown_fdr_level(beta_vec0, m1) / (1 - alpha_vec0[-1])
    # Now scale the vectors to ensure control
    alpha_vec1 = pfdr * alpha_vec0 / pfdrx
    beta_vec1 = pfnr * beta_vec0 / pfnrx
    # However this adjusts the denominators of the scaling factors pfdrx and
    # pfnrx, meaning the bound may not hold. Repeat this procedure until
    # convergence.
    while (
        max(abs(alpha_vec1 - alpha_vec0)) > epsilon
        or max(abs(beta_vec1 - beta_vec0)) > epsilon
    ):
        alpha_vec0 = copy.copy(alpha_vec1)
        beta_vec0 = copy.copy(beta_vec1)
        pfdrx = guo_rao_stepdown_fdr_level(alpha_vec0, m0) / (1 - beta_vec0[-1])
        pfnrx = guo_rao_stepdown_fdr_level(beta_vec0, m1) / (1 - alpha_vec0[-1])
        alpha_vec1 = pfdr * alpha_vec0 / pfdrx
        beta_vec1 = pfnr * beta_vec0 / pfnrx
        logging.debug(
            "Rej alpha log range: {0}\nAcc beta log range: {1}".format(
                np.log10(max(abs(alpha_vec1 - alpha_vec0))),
                np.log10(max(abs(beta_vec1 - beta_vec0))),
            )
        )
    # Confirm that nothing has gone wrong.
    pvalue_cutoff_verifier(alpha_vec1, beta_vec1)
    return alpha_vec1, beta_vec1

def get_lb_prob_at_least_one_rejection(alpha_vec: FloatArray, dontbelazy=True) -> float:
    """Get lower bound for the probability of at least one rejection.

    Args:
        alpha_vec (np.ndarray): increasing vector of p-value cutoffs.

    Returns:
        float: lower bound for the probability of at least one rejection.
    """
    if dontbelazy:
        import datetime
        if datetime.datetime.now(tz=datetime.timezone.utc) > datetime.datetime(
            2024, 3,4,10,10,10, tzinfo=datetime.timezone.utc):
            raise ValueError("Don't be lazy")
    return np.min(1.0 - alpha_vec)

def pfdr_finite_horizon_pvalue_cutoffs(
    alpha_raw_vec: FloatArray,
    pfdr: float,
    m0: Optional[int] = None,
    epsilon: float = 10.0**-8,
    max_iters: int = 25,
) -> FloatArray:
    """pFDR controlled pvalue cutoffs for finite horizon rejective sequential stepdown.

    Uses an interative scheme to find pvalue cutoffs that satisfy the pfdr
    levels that utilize the same vector structure as the raw inputs.
    

    Args:
        alpha_raw_vec (np.ndarray): raw vector of rejective p-value cutoffs.
            only the structure will be used, as a linear scaling will be applied.
        pfdr (float): desired pFDR control level
        m0 (Optional[int]): number of true nulls, if known
        epsilon (float, optional): tolerance for convergence. Defaults to 10.0**-8.

    Returns:
        np.ndarray: final alpha vector.
    """
    # Total number of hypotheses
    m = len(alpha_raw_vec)
    # number of false hypotheses
    if m0 is None:
        m1 = None
    else:
        assert (m0 > 0) and (
            m0 < m
        ), "Number of true nulls must be in (0, m) for joint pFDR and pFNR control."
        m1 = m - m0
    # Raw alpha and beta vectors
    alpha_vec0 = alpha_raw_vec
    pvalue_cutoff_verifier(alpha_vec0)
    lb_prob_at_least_one_rejection = get_lb_prob_at_least_one_rejection(alpha_vec0)
    # Get pFDR control levels using current alpha vector by first 
    # calculating FDR (via Guo+Rao) then scaling those by an approximate 
    # lower bound for the probability of at least one rejection)
    pfdrx = guo_rao_stepdown_fdr_level(alpha_vec0, m0) / lb_prob_at_least_one_rejection
    
    # Now scale the vectors to ensure control
    alpha_vec1 = pfdr * alpha_vec0 / pfdrx
    # However this adjusts the denominators of the scaling factors pfdrx and
    # pfnrx, meaning the bound may not hold. Repeat this procedure until
    # convergence.
    ii = 0
    while (
        max(abs(alpha_vec1 - alpha_vec0)) > epsilon
    ):
        alpha_vec0 = copy.copy(alpha_vec1)
        lb_prob_at_least_one_rejection = get_lb_prob_at_least_one_rejection(alpha_vec0)
        pfdrx = guo_rao_stepdown_fdr_level(alpha_vec0, m0) / lb_prob_at_least_one_rejection
        
        alpha_vec1 = pfdr * alpha_vec0 / pfdrx
        logging.debug(
            "Rej alpha log range: {0}".format(
                np.log10(max(abs(alpha_vec1 - alpha_vec0)))
            )
        )
        if ii > max_iters:
            raise ValueError("Failed to converge in {0} iterations.".format(max_iters))
        ii += 1
    # Confirm that nothing has gone wrong.
    pvalue_cutoff_verifier(alpha_vec1)
    return alpha_vec1

def importance_sample_interpolation_helper(
    param0: Dict[str, Any],
    param1: Dict[str, Any],
    imp_sample_prop: float,
    hyp_type: HypTypes,
) -> float:
    """Calculate the simulation parameter for importance sampling.

    Args:
        theta0: null param
        theta1: alt param
        imp_sample_prop: p=finv( f(p1) * q + f(p0) * (1-q))  as simulation dist param
            where f and finv depend on the hypothesis type.
        hyp_type: drug, binom, etc

    Returns:
        The simulation parameter for importance sampling.
    """
    if hyp_type == "drug" or hyp_type == "binom":
        theta0 = param0["p"]
        theta1 = param1["p"]
        sim_theta = expit(
            imp_sample_prop * logit(theta1) + (1.0 - imp_sample_prop) * logit(theta0)
        )
        sim_params = copy.deepcopy(param0)
        sim_params["p"] = sim_theta

    elif hyp_type == "pois":
        theta0 = param0["mu"]
        theta1 = param1["mu"]
        sim_theta = np.exp(
            imp_sample_prop * np.log(theta1) + (1.0 - imp_sample_prop) * np.log(theta0)
        )
        sim_params = copy.deepcopy(param0)
        sim_params["mu"] = sim_theta
    elif hyp_type == "gaussian":
        raise NotImplementedError("Gaussian not implemented")
        theta0 = param0["mu"]
        theta1 = param1["mu"]
        sim_theta = imp_sample_prop * theta1 + (1.0 - imp_sample_prop) * theta0
        sim_params = copy.deepcopy(param0)
        sim_params["mu"] = sim_theta
    else:
        raise Exception("Unrecognized hyp type ", hyp_type)
    return sim_params


def finite_horizon_cutoff_simulation(
    params0: Dict[str, Any],
    params1: Dict[str, Any],
    hyp_type: HypTypes,
    n_periods: int = 100,
    n_reps: int = 1000,
    imp_sample: bool = True,
    imp_sample_prop: float = 0.2,
    imp_sample_hedge: Optional[float] = None,
) -> Tuple[FloatArray, FloatArray]:
    """Generate finite horizon sample path maxs (and weights) for MC cutoff estimation.

    Allows use of importance sampling to reduce variance of simulation.

    Args:
        params0: null param
        params1: alt param
        hyp_type: drug, binom, etc
        n_periods: horizon
        n_reps: number of MC reps
        imp_sample: boolean, use importance sampling
        imp_sample_prop: p=p1 * q + p0 * (1-q)  as simulation dist param

    Returns:
        maxs: array of max llr for each MC rep
        weights: array of weights for each MC rep. For a non-importance sampled run, these should be even.
            When importance sampling is employed,

    """
    if imp_sample_hedge is not None:
        raise NotImplementedError("imp_sample_hedge is not yet implemented.")
    out_rec = []

    weight_out = []
    # If importance sampling is requested, calculate the simulation parameter
    # based on the model type, the hypotheses, and the interpolation parameter.
    sim_params = importance_sample_interpolation_helper(
        params0, params1, imp_sample_prop, hyp_type
    )
    # Calculate the number of importance samples to draw.
    # n_imp_samples = int(n_reps * imp_sample_hedge)
    # Set up iterator and logging for parallelization.
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # logger.debug("Simulating {0} with param {1}".format(hyp_type, sim_param))

    # Perform MC iterations.
    for ii in tqdm(range(n_reps), desc="MC cutoff simulations"):

        # Switch to true H0 after max_imp_samples
        # if do_imp and (i >= max_imp_samples):
        #     logger.debug("Switch to H0 at {0} of {1}.".format(i, n_reps))
        #     do_imp = False
        #     sim_param = p0

        llr_paths, obs_data = data_funcs.generate_llr(
            params=sim_params,
            n_periods=n_periods,
            rho=1e-8,
            hyp_type=hyp_type,
            params0=params0,
            params1=params1,
            rand_order=False,
            # extra_params=extra_params,
        )
        log_stepwise_imp_weight = data_funcs.compute_llr(
            observed_data=obs_data,
            hyp_type=hyp_type,
            params0=sim_params,
            params1=params0,
            cumulative=False,
        )
        log_imp_weight = log_stepwise_imp_weight.sum(0)
        imp_weight = np.exp(log_imp_weight)

        weight_out.append(imp_weight)

        if np.isnan(llr_paths).any().any():
            raise ValueError("NaN in llr at iter {0}".format(ii))
        if imp_sample and np.isnan(np.array(weight_out)).any().any():
            raise ValueError("NaN in weights at iter {0}".format(ii))
        #            # Record the max value the each llr path reached
        out_rec.append(llr_paths.max(0))
    out_rec = np.array(out_rec)
    weight_out = np.array(weight_out)

    return out_rec, weight_out


def gumbel_r_param_fit(samples: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    """Get the location and scale fit for a Gumbel R distribution using weighted samples.

    Args:
        samples: array of samples
        weights: array of weights for each sample.

    Returns:
        Tuple of location and scale parameters for the Gumbel R distribution."""

    # Define the weighted log-likelihood function for the Gumbel distribution
    def weighted_log_likelihood(params: Tuple[float, float]) -> Tuple[float, float]:
        mu, log_beta = params
        beta = np.exp(log_beta)
        # Compute the log-likelihood of each sample
        log_likelihoods = (
            -(samples - mu) / beta - np.exp(-(samples - mu) / beta) - np.log(beta)
        )
        # Apply weights
        weighted_log_likelihoods = weights * log_likelihoods
        # Return the negative sum of weighted log-likelihoods (since we're minimizing)
        return -np.sum(weighted_log_likelihoods)

    # Initial parameter guesses
    initial_params = [np.mean(samples), np.log(np.std(samples))]

    # Minimize the negative weighted log-likelihood
    result = scipy_minimize(weighted_log_likelihood, initial_params)

    # Extract the estimated parameters
    mu_est, log_beta_est = result.x
    beta_est = np.exp(log_beta_est)
    return mu_est, beta_est


def gumbel_r_fit_dist(samples: np.ndarray, weights: np.ndarray) -> stats.gumbel_r:
    """Fit a Gumbel R distribution to weighted samples.

    Args:
        samples: array of samples
        weights: array of weights for each sample.

    Returns:
        A Gumbel R distribution fitted to the weighted samples."""
    mu_est, beta_est = gumbel_r_param_fit(samples, weights)
    return stats.gumbel_r(loc=mu_est, scale=beta_est)

def get_tail_values(dists: List[stats.rv_continuous], quantiles: FloatArray) -> FloatArray:
    """Get the tail values  a list of distributions at specified quantiles.

    Args:
        dists: list of distributions
        quantiles: quantiles at which to evaluate the tails

    Returns:
        Square array with rows corresponding to data streams/hypotheses and columns to cutoffs."""
    return np.array([dist.ppf(1 - quantiles) for dist in dists])

def get_max_tail_values(dists: List[stats.rv_continuous], quantiles: FloatArray) -> FloatArray:
    """Get the max tail values for a list of distributions at specified quantiles.

    Args:
        dists: list of distributions
        quantiles: quantiles at which to evaluate the tails

    Returns:
        Array of max tail values for each quantile."""
    return get_tail_values(dists, quantiles).max(0)


def estimate_finite_horizon_rejective_llr_cutoffs(
    params0: Dict[str, Any],
    params1: Dict[str, Any],
    alpha_levels: FloatArray,
    n_periods: int,
    k_reps: int,
    hyp_type: HypTypes,
    # sleep_time=5,
    imp_sample: bool = True,
    imp_sample_prop: float = 0.5,
    imp_sample_hedge: Optional[float] = None,
    clip_at_zero: bool = True,
):
    """Calculate finite horizon rejective cutoffs from alpha levels using MC."""
    # Get the max LLR value accross each stream with a weight for each
    # sample via importance sampling by generating from parameters in between
    # the null and alternative.
    # k_reps is the number of samples to draw
    record, weights = finite_horizon_cutoff_simulation(
        params0,
        params1,
        hyp_type,
        n_periods,
        k_reps,
        imp_sample=imp_sample,
        imp_sample_prop=imp_sample_prop,
        # imp_sample_hedge=imp_sample_hedge,
    )
    assert (
        record.shape == weights.shape
    ), f"record shape {record.shape} does not match weights shape {weights.shape}."
    m_hyps = record.shape[-1]
    # Get the cutoffs for each individual stream, either exact or using a tdist
    fit_dists = [gumbel_r_fit_dist(record[:, ii], weights[:, ii]) for ii in range(m_hyps)]
    cutoff_levels = get_max_tail_values(fit_dists, alpha_levels)

    #    from IPython.display import display
    #    display(cutoff_levels)

    if (cutoff_levels < 0).any():
        num_neg = (cutoff_levels < 0).sum()

        warnings.warn(
            "{0} Cutoff levels are negative: from {1} to {2}".format(
                num_neg, cutoff_levels.min(), cutoff_levels.max()
            )
        )
    EPS = 1e-5
    # TODO: clipping cutoffs at EPS is serious kludge. Find a better approach here.
    if clip_at_zero:
        cutoff_levels[cutoff_levels<EPS] = EPS
    llr_cutoff_verifier(cutoff_levels)

    return cutoff_levels


# ALPHA_SHIFT = 0.0


# def empirical_quant_presim_wrapper(kwargs):
#     stream_specific_cutoff_levels = []
#     record = kwargs["record"]
#     weights = kwargs["weights"]
#     alpha_levels = kwargs["alpha_levels"]
#     if "no_left_extrapolate" in kwargs:
#         left = np.NaN
#     else:
#         left = None

#     range_iter = range(record.shape[1])

#     if ("job_id" not in kwargs) or (kwargs["job_id"] == 0):
#         range_iter = tqdm(range_iter, desc="Per stream quantile estimation")

#     for stream_num in range_iter:
#         stream_record_raw = record[:, stream_num]
#         stream_weight_raw = weights[:, stream_num]
#         stream_idx = np.argsort(stream_record_raw)
#         stream_record = stream_record_raw[stream_idx]
#         stream_weight = stream_weight_raw[stream_idx] / stream_weight_raw.sum()
#         stream_cdf = stream_weight.cumsum() - ALPHA_SHIFT * stream_weight[0]

#         stream_cutoffs = np.interp(
#             1.0 - alpha_levels, stream_cdf, stream_record, left=left, right=np.NaN
#         )
#         if np.isnan(stream_cutoffs).any():
#             print("RAW ", stream_weight[0])
#             print(
#                 "Stream min {0} stream max {1}".format(
#                     stream_cdf.min(), stream_cdf.max()
#                 )
#             )
#             print(
#                 "alpha min {0} alpha max {1}".format(
#                     (1.0 - alpha_levels).min(), (1.0 - alpha_levels).max()
#                 )
#             )
#             raise ValueError("NaN found in stream cutoffs.")

#         stream_specific_cutoff_levels.append(stream_cutoffs)
#     stream_specific_cutoff_levels = np.array(stream_specific_cutoff_levels).T
#     return stream_specific_cutoff_levels



# def llr_term_moments(drr: pd.Series, p0: float, p1: float) -> pd.DataFrame:
#     """Expectation of the llr terms for a drug sim  under the null hypothesis for each step.

#     Used for calculating finite horizon cutoffs via gaussian approximation of
#     the llr statistic path. Assumes temporal independence and thus a simple
#     hypothesis.

#     Args:
#         drr (pd.Series): drug use rate. Informs how many samples should be
#             expected at each step.
#         p0 (float): null hypothesis probability.
#         p1 (float): alternative hypothesis probability.

#     Returns:
#         pd.DataFrame: with two columns:
#             term_mean: the expected value of an individual term's (or stage's)
#             contribution to the llr statistic
#             term_var: variance of each term's contribution.
#     """
#     term_mean = drr * (p0 * np.log(p1 / p0) + (1 - p0) * np.log((1 - p1) / (1 - p0)))
#     term_var = drr * (
#         p0 * np.log(p1 / p0) ** 2.0 + (1 - p0) * np.log((1 - p1) / (1 - p0)) ** 2.0
#     )
#     return pd.DataFrame({"term_mean": term_mean, "term_var": term_var})


# def llr_binom_term_moments(p0: pd.Series, p1: pd.Series) -> pd.DataFrame:
#     """Calculates mean and variance of single-example binomial llr terms.

#     Args:
#         p0 (pd.Series): null hypothesis probability.
#         p1 (pd.Series): alternative hypothesis probability.

#     Returns:
#         pd.DataFrame: contains two columns:
#             team_mean: mean of the llr terms.
#             term_var: variance of the llr terms.
#     """
#     const_a = np.log((1 - p1) / (1 - p0))
#     const_b = np.log(p1 / p0) - np.log((1 - p1) / (1 - p0))
#     eX = p0
#     varX = p0 * (1 - p0)
#     term_mean = const_a + const_b * eX
#     term_var = (const_b**2.0) * varX
#     return pd.DataFrame({"term_mean": term_mean, "term_var": term_var})


# def llr_pois_term_moments(lam0: pd.Series, lam1: pd.Series) -> pd.DataFrame:
#     """Calculates mean and variance of one step for a poisson llr.

#     Args:
#         lam0 (pd.Series): null hypothesis rate.
#         lam1 (pd.Series): alternative hypothesis rate.

#     Returns:
#         pd.DataFrame: contains two columns:
#             team_mean: mean of the llr terms.
#             term_var: variance of the llr terms.
#     """
#     const_a = -(lam1 - lam0)
#     const_b = np.log(lam1 / lam0)
#     eX = lam0
#     varX = lam0
#     term_mean = const_a + const_b * eX
#     term_var = (const_b**2.0) * varX
#     return pd.DataFrame({"term_mean": term_mean, "term_var": term_var})


# Var(aX+bY) = a**2 VarX + b**2 VarY + 2ab CovXY
# def est_sample_size(alpha, beta, drr, p0, p1, BH=True):
#    N_drugs = len(drr)
#    # First come up with p-value cutoffs
#    # BH
#    if BH:
#        alpha_vec_raw = alpha * np.arange(1, 1+N_drugs) / float(N_drugs)
#    # Holm
#    else:
#        alpha_vec_raw = alpha / (float(N_drugs) - np.arange(N_drugs))
#
#    alpha_vec = create_fdr_controlled_alpha(alpha, alpha_vec_raw)
#    beta_vec = create_fdr_controlled_alpha(beta, alpha_vec_raw)
#    A_vec, B_vec = calculate_mult_sprt_cutoffs(alpha_vec, beta_vec)
#    return (max(A_vec * alpha_vec[::-1]) + max(-B_vec * beta_vec[::-1])) / abs(llr_term_moments(drr, p0, p1)["term_mean"]).max()


# From govindarajulu. Expected stopping times for a single hypothesis
def single_hyp_sequential_expected_acceptance_time(
    A: float, B: float, mu_1: float
) -> int:
    """Expected acceptance time for single hypothesis SPRT from govindarajulu.

    Args:
        A (float): Rejection threshold
        B (float): Acceptance threshold
        mu_1 (float): average llr increment per step under the alternative hypothesis.

    Returns:
        int: expected number of steps until termination.
    """
    return int(
        (B * np.exp(B) * (np.exp(A) - 1) + A * np.exp(A) * (1 - np.exp(B)))
        / ((np.exp(A) - np.exp(B)) * mu_1)
    )


def single_hyp_sequential_expected_rejection_time(
    A: float, B: float, mu_0: float
) -> int:
    """Expected rejection time for single hypothesis SPRT from govindarajulu.

    Args:
        A (float): Rejection threshold
        B (float): Acceptance threshold
        mu_0 (float): average llr increment per step under the null hypothesis.

    Returns:
        int: expected number of steps until termination.
    """
    return int(B * (np.exp(A) - 1) + A * (1 - np.exp(B))) / (
        (np.exp(A) - np.exp(B)) * mu_0
    )


def llr_term_mean_general(
    params0: Dict[str, Any], params1: Dict[str, Any], hyp_type: HypTypes
) -> pd.Series:
    """Calculate the expected value of the llr terms for a general hypothesis test under the null.

    E_{H_{0}}[log(L(theta_1) / L(theta_0))]
    Args:
        params0 (Dict[str, Any]): null hypothesis parameters
        params1 (Dict[str, Any]): alternative hypothesis parameters
        hyp_type (HypTypes): The type of hypothesis test to use.  One of "drug", "pois", or "binom"

    Returns:
        pd.Series: The expected value of the llr terms
    """
    if hyp_type == "drug":
        # Likelihood L(mu,p)=exp(-mu) * (mu ** (x+y)) * (p**x) * ((1-p)**y) / (x! * y!)
        eX = params0["mu"] * params0["p"]
        eY = params0["mu"] * (1.0 - params0["p"])
        const_a = np.log(params1["p"] / params0["p"])
        const_b = np.log((1 - params1["p"]) / (1 - params0["p"]))
        term_mean = const_a * eX + const_b * eY
        return term_mean
    elif hyp_type == "pois":

        const_a = -(params1["mu"] - params0["mu"])
        const_b = np.log(params1["mu"] / params0["mu"])
        eX = params0["mu"]
        term_mean = const_a + const_b * eX
        return term_mean
    elif hyp_type == "binom":
        eX = params0["p"] * params0["n"]
        const_a = np.log(params1["p"] / params0["p"])
        const_b = np.log((1 - params1["p"]) / (1 - params0["p"]))
        term_mean = const_a * eX + const_b * (params0["n"] - eX)
        return term_mean
    else:
        raise ValueError("Unknown type {0}".format(hyp_type))


def llr_cutoff_verifier(
    A_vec: Union[np.ndarray, pd.Series],
    B_vec: Union[np.ndarray, pd.Series, None] = None,
):
    """Verify that the cutoffs are in the correct order and are the same length."""
    if isinstance(A_vec, pd.Series):
        A_vec = A_vec.values
    assert (np.diff(A_vec) < 0).all(), "A_vec is not in descending order"
    assert A_vec[-1] > 0, "A_vec[-1] is not greater than 0"
    if B_vec is not None:
        if isinstance(B_vec, pd.Series):
            B_vec = B_vec.values
        assert (np.diff(B_vec) > 0).all(), "B_vec is not in ascending order"
        assert len(A_vec) == len(B_vec), "A_vec and B_vec are not the same length"
        assert B_vec[-1] < 0, "B_vec[-1] is not less than 0"


def pvalue_cutoff_verifier(
    alpha: Union[np.ndarray, pd.Series],
    beta: Union[np.ndarray, pd.Series, None] = None,
):
    """Verify pvalue cutoffs ordering and length."""
    assert (alpha > 0).all(), "alpha is not greater than 0"
    assert (np.diff(alpha) > 0).all(), "alpha is not in ascending order"
    if beta is not None:
        assert (beta > 0).all(), "beta is not greater than 0"
        assert (np.diff(beta) > 0).all(), "beta is not in ascending order"
        assert len(alpha) == len(beta), "alpha and beta are not the same length"


def est_sample_size(
    A_vec: FloatArray,
    B_vec: FloatArray,
    params0: Dict[str, Any],
    params1: Dict[str, Any],
    hyp_type: HypTypes,
) -> int:
    """Estimate the sample size needed to accept or reject all hypotheses.

    in general, very conservative. Calculates expected rejection and
    acceptance times for the worst case cutoffs and worst case
    hypotheses, then takes the worst of the two. The expectation could
    be misleading, but given the pairing of worst case hypothesis with
    worst case cutoff, unlikely to underestimate.


    Args:
        A_vec (np.array):   A_vec[i] is the rejective?? cutoff for the ith hypothesis
        B_vec (np.array):   B_vec[i] is the acceptive?? cutoff for the ith hypothesis
        drr (pd.Series):    drug use rate series for drug hyptotheses
        p0 (float):         The null hypothesis probability (or poisson rate if hyp_type is "pois")
        p1 (float):         The alternative hypothesis probability (or poisson rate if hyp_type is "pois")
        hyp_type (str):     The type of hypothesis test to use.  One of "drug", "pois", or "binom"

    Returns:
        int: The estimated sample size needed to accept or reject all hypotheses
    """
    llr_cutoff_verifier(A_vec, B_vec)
    negative_drift_under_null = llr_term_mean_general(params0, params1, hyp_type)
    positive_drift_under_alt = -llr_term_mean_general(params1, params0, hyp_type)

    # Get slowest drifting hypotheses, ie worst case.
    mu_0 = (negative_drift_under_null).max()
    mu_1 = (positive_drift_under_alt).min()
    # Get the most extreme cutoffs
    most_extreme_rej_cutoff = A_vec[0]
    most_extreme_acc_cutoff = B_vec[0]
    # Get expected stopping time for absolute worst case rej and acc
    max_expected_acceptance_time = single_hyp_sequential_expected_acceptance_time(
        most_extreme_rej_cutoff,
        most_extreme_acc_cutoff,
        mu_1,
    )
    max_expected_rejection_time = single_hyp_sequential_expected_rejection_time(
        most_extreme_rej_cutoff,
        most_extreme_acc_cutoff,
        mu_0,
    )
    # Take the worst of those.
    return max((max_expected_acceptance_time, max_expected_rejection_time))


# def est_sample_size(
#     A_vec: FloatArray,
#     B_vec: FloatArray,
#     theta0: pd.Series,
#     theta1: pd.Series,
#     hyp_type: Optional[HypTypes] = "drug",
#     extra_params: Optional[Dict[str, Any]] = None,
# ) -> int:
#     """Estimate the sample size needed to accept or reject all hypotheses.

#     in general, very conservative. Calculates expected rejection and
#     acceptance times for the worst case cutoffs and worst case
#     hypotheses, then takes the worst of the two. The expectation could
#     be misleading, but given the pairing of worst case hypothesis with
#     worst case cutoff, unlikely to underestimate.


#     Args:
#         A_vec (np.array):   A_vec[i] is the rejective?? cutoff for the ith hypothesis
#         B_vec (np.array):   B_vec[i] is the acceptive?? cutoff for the ith hypothesis
#         drr (pd.Series):    drug use rate series for drug hyptotheses
#         p0 (float):         The null hypothesis probability (or poisson rate if hyp_type is "pois")
#         p1 (float):         The alternative hypothesis probability (or poisson rate if hyp_type is "pois")
#         hyp_type (str):     The type of hypothesis test to use.  One of "drug", "pois", or "binom"

#     Returns:
#         int: The estimated sample size needed to accept or reject all hypotheses
#     """
#     if (hyp_type is None) or (hyp_type == "drug"):
#         negative_drift_under_null = llr_term_moments(drr, p0, p1)["term_mean"]
#         positive_drift_under_alt = -llr_term_moments(drr, p1, p0)["term_mean"]
#     elif hyp_type == "pois":
#         negative_drift_under_null = llr_pois_term_moments(np.array([p0]), np.array([p1]))["term_mean"]
#         positive_drift_under_alt = -llr_pois_term_moments(np.array([p1]), np.array([p0]))["term_mean"]
#     elif hyp_type == "binom":
#         negative_drift_under_null = llr_binom_term_moments(np.array([p0]), np.array([p1]))["term_mean"]
#         positive_drift_under_alt = -llr_binom_term_moments(np.array([p1]), np.array([p0]))["term_mean"]
#     else:
#         raise ValueError("Unknown type {0}".format(hyp_type))
#     # Get slowest drifting hypotheses, ie worst case.
#     mu_0 = (negative_drift_under_null).max()
#     mu_1 = (positive_drift_under_alt).min()
#     # Get the most extreme cutoffs
#     most_extreme_rej_cutoff = A_vec[0]
#     most_extreme_acc_cutoff = B_vec[0]
#     # Get expected stopping time for absolute worst case rej and acc
#     max_expected_acceptance_time = single_hyp_sequential_expected_acceptance_time(
#         most_extreme_rej_cutoff,
#         most_extreme_acc_cutoff,
#         mu_1,
#         )
#     max_expected_rejection_time = single_hyp_sequential_expected_rejection_time(
#         most_extreme_rej_cutoff,
#         most_extreme_acc_cutoff,
#         mu_0,
#         )
#     # Take the worst of those.
#     return max((max_expected_acceptance_time, max_expected_rejection_time))


# Importance sampling section


def imp_sample_drug_weight(
    counts: Tuple[pd.DataFrame, pd.DataFrame],
    p0: Union[float, pd.Series],
    p1: Union[float, pd.Series],
    drr0: Union[float, pd.Series, None] = None,
    drr1: Union[float, pd.Series, None] = None,
) -> pd.Series:
    """Calculates the ratio of the likelihood of hte null and importance distributions

    The model is, as elsewhere, that there is an overall sideeffect rate with a
    proportion of those side effects being amnesia reports

    When only counts, p0, and p1 are passed, this func assumes the overall side
    effect rate under the null and importance distributions are the same, and
    that they wash out in the weight, so it can be treated as a simple binomial.

    When drr0 and drr1 are passed in addition to the first three arguments, the
    full poisson distribution is considered.

    p0, p1, drr0, and drr1 can be floats if constant across MC reps, or Series
    if they differ across reps.

    Args:
        counts: tuple of amnesia and non-amensia count dataframes for all MC sims.
            columns are ... TODO
        p0: null probability of a side effect report being amnesia.
        p1: importance sampling distribution probabilty of a side effect being amnesia
        drr0: optional overall reaction rate for the drug under the null
        drr1: optional overall reaction rate for the drug under the importance
            sampling distribution.

    Retruns:
        pandas series of weights for each MC simulation.

    """
    # Get final outcomes from both amnesia and non-amnesia reactions
    # for each MC rep. These are both series.
    amcount = counts[0].iloc[-1, :]  # Final step is all thats important
    nonamcount = counts[1].iloc[-1, :]
    # Calculate the main parameter terms in the likelihood ratio.
    am_factor = p0 / p1
    nonam_factor = (1 - p0) / (1 - p1)
    # Calculate likelihood ratio for binomial.
    log_weight = amcount * np.log(am_factor) + nonamcount * np.log(nonam_factor)
    raw_weight = np.exp(log_weight)
    if np.isnan(raw_weight).any():
        logger = logging.getLogger()
        logger.debug("NaN weights: {0}".format(raw_weight[np.isnan(raw_weight)]))
        logger.debug("Log weights: {0}".format(log_weight[np.isnan(raw_weight)]))

    # TODO: check this formula and describe it in detail.
    if drr0 is not None and drr1 is not None:
        # Number of periods
        T = counts[0].shape[0]
        weight = (
            raw_weight
            * ((drr0 / drr1).astype("float128") ** (amcount + nonamcount))
            * np.exp(-T * (drr0 - drr1)).astype("float128")
        )
    if ((drr0 is None) and (drr1 is not None)) or (
        (drr0 is not None) and (drr1 is None)
    ):
        raise ValueError(
            "Either both drug response rates should be passed, or "
            "niether, for calculating importance sampling weights"
        )
    else:
        weight = raw_weight
    return weight


def imp_sample_binom_weight(counts: pd.DataFrame, p0: float, p1: float) -> pd.Series:
    """Calculates likelihood ratio importance sampling wieghts for finite horizon MC reps for binomial model.

    Assumes each step is a bernoulli. TODO: consider chaning that??

    Args:
        counts (Tuple[pd.DataFrame, pd.DataFrame]): _description_
        p0 (float): _description_
        p1 (float): _description_

    Returns:
        pd.Series: _description_
    """
    # Gets seires of
    n_flips = counts.shape[0]
    success_ser = counts.iloc[-1, :]  # Final step is all thats important
    fail_ser = n_flips - success_ser
    # Calculate parameter terms in likelihood ratio.
    event_factor = p0 / p1
    fail_factor = (1 - p0) / (1 - p1)
    # Compute likelihood ratio weights.
    log_weight = success_ser * np.log(event_factor) + fail_ser * np.log(fail_factor)
    return np.exp(log_weight)


def imp_sample_pois_weight(counts, p0, p1):
    events = counts.iloc[-1, :]  # Final step is all thats important
    n = counts.shape[0]
    factor = p0 / p1
    # irrelevant factor
    # stupid = np.exp(-counts.shape[0]*(po - p1))
    log_weight = events * np.log(factor) - n * (p0 - p1)
    return np.exp(log_weight)


def imp_sample_gaussian_weight(counts, p0, p1, drr):
    raise Exception("Not implemented yet. Composite or simple? SD or VAR?")
