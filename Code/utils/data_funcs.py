"""Funcs for reading drug data, generating fake data, generating hypotheses, and computing llr paths.

List of all functions in this module:
* Loading and screening drug data
    * gen_names: Generates a list of n_hyps unique names for hypotheses.
    * read_drug_data: Reads drug reaction counts, screens drugs, and returns rates df.
    * prescreen_abs: generates a drug screening function based on absolute cutoffs in reaction counts.
    * prescreen_rel: generates a drug screening function based on percentile cutoffs in reaction counts.
    * gen_skew_prescreen:
    * drug_data_metadata: dataclass... not really metadata and not used in this module.... maybe delete?
* Generate fake data rates (but not observations)
    * assemble_fake_drugs:
    * assemble_fake_gaussian:
    * assemble_fake_binom:
    * assemble_fake_pois:
    * assemble_fake_pois_grad:
* Generate fake observations
    * simulate_reactions: Given series of drug amnesia and non-amnesia rates, gerenates data.
    * simulate_binom:
    * simulate_pois:
* Assemble llr statistic paths based on hypotheses and observed data
    * assemble_drug_llr:
    * assemble_binom_llr:
    * assemble_pois_llr:
    * generate_llr: takes a distribution argument, generates observation path, and calls one of the specific llrs
* Generate null and alt hypotheses from drug reaction data
    * whole_data_p0:
    * whole_data_p1:
    * drug_data_p1:
    * am_prop_percentile_p0_p1:
* Possibly functional data generators for correlated data streams
    * toep_corr_matrix:
    * simulate_correlated_reactions:
    * simulate_correlated_binom:
    * simulate_correlated_pois:
* Non-functional data generator structures for future use in real streaming applications
    * online_data:
    * online_data_generator:
    * dg_dgp_wrapper:
    * df_generator:


"""
from typing import Callable, Dict, List, Optional, Tuple
import pandas as pd
import datetime, os, string, warnings
import numpy as np
import numpy.random
from scipy.linalg import toeplitz
from scipy.stats import norm, multivariate_normal, poisson, binom
import hashlib
import logging
from dataclasses import dataclass


def gen_names(n_hyps: int) -> List[str]:
    """Generates a list of n_hyps unique names for hypotheses.

    Names will all be of the same length and take the form
    {hypothesis number}-{hashed string}
    for instance, if there are 12 hypotheses we might have
    07-3f2

    Args:
        n_hyps (int): number of hypotheses to generate names for.

    Returns:
        List[str]: list of names for hypotheses.
    """
    STR_LEN = 3
    digits = int(numpy.ceil(numpy.log10(n_hyps)))
    for i in range(n_hyps):
        return [
            ("{0:0=" + str(digits) + "d}-{1}").format(
                i, hashlib.sha1(str(i).encode("utf-8")).hexdigest()[:STR_LEN]
            )
            for i in range(n_hyps)
        ]


# TODO: hardcoding this is EXTREMELY amateur
TOTAL_AMNESIA_CNAME = "Total All"
TOTAL_REACTS_CNAME = "total_reactions"
DATA_FILE_PATH = os.path.expanduser("../../Data/YellowcardData.csv")
ScreeningFuncType = Callable[[pd.DataFrame, str, str], pd.DataFrame]


def read_drug_data(
    path: str, prescreen: Optional[ScreeningFuncType] = None, aug: int = 1
) -> Tuple[pd.Series, pd.Series, Tuple[int, int]]:
    """Read drug reaction counts and metadata from file; screen drugs and return rates df.

    Args:
        path (str): path to csv file containing drug reaction data.
        prescreen (bool, optional): whether to prescreen drugs with few reports. Defaults to True.
        aug (int, optional): amount to augment amnesia and non-amnesia report counts by. Defaults to 1.

    Returns:
        Tuple[pd.Series, pd.Series, Tuple[int, int]]: Tuple containing:
            * amnesia rates for each drug
            * non-amnesia rates for each drug
            * tuple containing (total number of amnesia reports, total number of all side effect reports)
    """
    # sum total days over all drugs, divide total reports by that
    # emit reports from each drug and total drugs with given rates
    # can't be formulated as sprt.... look at bartroff's???

    reacts_df = pd.read_csv(
        path, index_col=0, parse_dates=["end_date", "start_date", "first_react"]
    )

    # Here either exclude drugs without first report date or fill in report start date
    first_react_unknown_mask = reacts_df.isnull()["first_react"]
    reacts_df.loc[first_react_unknown_mask, "first_react"] = reacts_df[
        first_react_unknown_mask
    ]["start_date"]

    # Optional prescreening
    if prescreen:
        reacts_df = prescreen(reacts_df, TOTAL_AMNESIA_CNAME, TOTAL_REACTS_CNAME)

    amnesia_reactions = reacts_df[TOTAL_AMNESIA_CNAME]
    total_reactions = reacts_df[TOTAL_REACTS_CNAME]
    secs_per_year = datetime.timedelta(365).total_seconds()
    date_range_col = (reacts_df["end_date"] - reacts_df["first_react"]).apply(
        lambda drange: drange / numpy.timedelta64(1, "s")
    ) / secs_per_year
    N_reports = reacts_df[TOTAL_REACTS_CNAME].sum()
    N_amnesia = reacts_df[TOTAL_AMNESIA_CNAME].sum()
    # Calculate rates
    drug_amnesia_rate = (aug + amnesia_reactions) / date_range_col
    drug_reacts_rate = (2 * aug + total_reactions) / date_range_col
    drug_nonamnesia_rate = drug_reacts_rate - drug_amnesia_rate
    return drug_amnesia_rate, drug_nonamnesia_rate, (N_amnesia, N_reports)


def prescreen_abs(min_am_reacts: int, min_total_reacts: int) -> ScreeningFuncType:
    """Creates a prescreening function based on absolute reaction counts.

    Clips drugs with too few reports based on absolute counts. Screens out drugs that have
    fewer than min_am_reacts amnesia reports *or* fewer than min_total_reacts total side
    effect reports.
    Screening function takes a DataFrame of reaction records, as well as the target reaction type
    column and the total reaction column. The function is intended to be passed to read_drug_data.


    Args:
        min_am_reacts (int): minimum number of amnesia reports for a drug to be included.
        min_total_reacts (int): minimum number of total side effect reports for a drug to be included.

    Returns:
        Callable[[pd.DataFrame, str, str], pd.DataFrame]: prescreening function that takes
            a dataframe of drug reaction rates and returns a dataframe of drug reaction rates.
    """

    def prescreen(
        reacts_df: pd.DataFrame,
        target_reaction_col_name: str,
        total_reaction_col_name: str,
    ) -> pd.DataFrame:
        """Screens a dataframe of drug reaction rates for drugs with too few reports."""
        mask = np.logical_and(
            reacts_df[target_reaction_col_name] >= min_am_reacts,
            reacts_df[total_reaction_col_name] >= min_total_reacts,
        )
        return reacts_df[mask]

    return prescreen


def prescreen_rel(
    min_am_reacts_percentile: float, min_total_reacts_percentile: float
) -> ScreeningFuncType:
    """Creates a prescreening function based on percentile reaction counts.

    Takes a percentile (numpy, (0,100)) of target reaction counts and total reaction counts
    over the whole set of drugs, and sets those as cutoffs, screens any drug with counts that
    fall below *either* of those cutoffs. Screening function takes a DataFrame of reaction records,
    as well as the target reaction type column and the total reaction column. The function is
    intended to be passed to read_drug_data.


    Args:
        min_am_reacts_percentile (float): minimum percentile of amnesia reports for a drug to be included. Between 1 and 99.
        min_total_reacts_percentile (float): minimum percentile of total side effect reports for a drug to be included. Between 1 and 99.

    Returns:
        Callable[[pd.DataFrame, str, str], pd.DataFrame]: prescreening function that takes a dataframe of drug reaction rates and
            returns a dataframe of drug reaction rates.
    """

    def prescreen(
        reacts_df: pd.DataFrame,
        target_reaction_col_name: str,
        total_reaction_col_name: str,
    ) -> pd.DataFrame:
        """Removes drugs with too few reports from a dataframe of drug reaction rates."""
        min_am_reacts = np.percentile(
            reacts_df[target_reaction_col_name], min_am_reacts_percentile
        )
        min_total_reacts = np.percentile(
            reacts_df[total_reaction_col_name], min_total_reacts_percentile
        )
        mask = np.logical_and(
            reacts_df[target_reaction_col_name] >= min_am_reacts,
            reacts_df[total_reaction_col_name] >= min_total_reacts,
        )
        return reacts_df[mask]

    return prescreen


def gen_skew_prescreen(
    min_am: int = 1, min_tot: int = 20, aug_am: int = 1, aug_non: int = 1
) -> ScreeningFuncType:
    """Generates a drug prescreening (and reaction count augmenting) function.

    Screens drugs that fall below *both* the minimum amnesia report count and the minimum total
    side effect report count. This allows more skewed drugs. Also augments amnesia report counts
    to smooth a bit.
    Screening function takes a DataFrame of reaction records, as well as the target reaction type
    column and the total reaction column. The function is intended to be passed to read_drug_data.

    Args:
        min_am (int, optional): minimum number of amnesia (or other target reaction) reports for
            a drug to be included. Defaults to 1.
        min_tot (int, optional): minimum number of total side effect reports for a drug to be
            included. Defaults to 20.
        aug_am (int, optional): amount to augment amnesia report counts by. Defaults to 1.
        aug_non (int, optional): amount to augment non-amnesia report counts by. Defaults to 1.

    Returns:
        Callable[[pd.DataFrame, str, str], pd.DataFrame]: prescreening function that takes
            a dataframe of drug reaction rates and returns a dataframe that contains augmented
            reaction counts for a subset of those initial drugs.
    """

    def skew_prescreen(
        reacts_df: pd.DataFrame, am_col: str, tot_col: str
    ) -> pd.DataFrame:
        # Select only drugs that have either min_am amnesia reports or min_tot total side effect reports.
        screened_df = reacts_df[
            (reacts_df[am_col] > min_am) | (reacts_df[tot_col] > min_tot)
        ].copy()
        # Calculate the amnesia rate for each drug (augmenting by aug_am)
        screened_df[TOTAL_AMNESIA_CNAME] = screened_df[TOTAL_AMNESIA_CNAME] + aug_am
        # Calculate the total side effects rate for each drug (augmenting by aug_non + aug_am)
        screened_df[TOTAL_REACTS_CNAME] = (
            screened_df[TOTAL_REACTS_CNAME] + aug_am + aug_non
        )
        return screened_df

    return skew_prescreen


@dataclass
class drug_data_metadata(object):
    N_reports: int
    N_amnesia: int
    p0: float
    p1: float
    n_periods: int


def build_interleaved_ground_truth(m_null: int, m_alt: int) -> list[bool]:
    both = np.min([m_null, m_alt])
    remaining_null = m_null - both
    remaining_alt = m_alt - both
    ground_truth = (
        [True, False] * both + [True] * remaining_null + [False] * remaining_alt
    )
    return ground_truth


def assemble_fake_drugs(
    max_magnitude: float,
    m_null: int,
    interleaved: bool,
    p0: float,
    p1: float,
    m_alt: Optional[int] = None,
) -> (pd.Series, pd.Series, pd.Series):
    """Assembles dar and dnar for fake drugs.

    Note: when m_alt is specified, the magnitudes will be messily interleaved.

    Args:
        max_magnitude (float): maximum magnitude of amnesia and non-amnesia rates.
        m_null (int): number of true nulls.
        interleaved (bool): whether to interleave null and alternative hypotheses.
        p0 (float): null amnesia rate.
        p1 (float): alternative amnesia rate.
        m_alt (Optional[int], optional): number of true alternatives. Defaults to None,

    Returns:
        (pd.Series, pd.Series, pd.Series): tuple of amnesia rate series, non-amnesia
            rate series, and ground truth series.
    """
    mag_vec = np.linspace(1, max_magnitude, m_null)

    drug_names = gen_names(2 * m_null)

    # Create null/alternative masks, and total magnitude series
    if interleaved:
        if m_alt is None:
            ground_truth = pd.Series(
                np.tile(np.array([True, False]), m_null), index=drug_names
            )
            drr = pd.Series(np.repeat(mag_vec, 2), index=drug_names)
        else:
            ground_truth = pd.Series(
                build_interleaved_ground_truth(m_null, m_alt), index=drug_names
            )
            drr = pd.Series(np.tile(mag_vec, 2), index=drug_names)
    else:
        if m_alt is None:
            ground_truth = pd.Series(
                np.repeat(np.array([True, False]), m_null), index=drug_names
            )
            drr = pd.Series(np.repeat(mag_vec, 2), index=drug_names)
        else:
            ground_truth = pd.Series(
                np.repeat(np.array([True, False]), [m_null, m_alt]), index=drug_names
            )
            drr = pd.Series(np.tile(mag_vec, 2), index=drug_names)

    # Create (non) amensia magnitude
    dar = (p0 * ground_truth + p1 * ~ground_truth) * drr
    dnar = drr - dar
    return dar, dnar, ground_truth


def assemble_fake_gaussian(max_magnitude, m_null, p0, p1, m_alt=None):
    """Assembles dar and dnar for fake gaussian."""
    var_vec_true = np.linspace(1, max_magnitude, m_null)
    var_vec_false = np.linspace(1, max_magnitude, m_alt)

    drug_names = gen_names(m_null + m_alt)

    # Create null/alternative masks, and total magnitude series
    if m_alt is None:
        ground_truth = pd.Series(
            np.repeat(np.array([True, False]), m_null), index=drug_names
        )
    else:
        ground_truth = pd.Series(
            np.repeat(np.array([True, False]), [m_null, m_alt]), index=drug_names
        )
    mean_vec = p0 * ground_truth + p1 * ~ground_truth
    sd_vec = pd.Series(
        np.sqrt(numpy.concatenate((var_vec_true, var_vec_false))), index=mean_vec.index
    )

    return mean_vec, sd_vec, ground_truth


def assemble_fake_binom(
    m_null: int, interleaved: bool, p0: float, p1: float, m_alt: Optional[int] = None
) -> (pd.Series, pd.Series):
    """Assembles dar and dnar for fake drugs."""
    if m_alt is None:
        drug_names = gen_names(2 * m_null)
    else:
        drug_names = gen_names(m_null + m_alt)
    # Create null/alternative masks, and total magnitude series
    if interleaved:
        if m_alt is None:
            ground_truth = pd.Series(
                np.tile(np.array([True, False]), m_null), index=drug_names
            )
        else:
            ground_truth = pd.Series(
                build_interleaved_ground_truth(m_null, m_alt), index=drug_names
            )

    else:
        if m_alt is None:
            ground_truth = pd.Series(
                np.repeat(np.array([True, False]), m_null), index=drug_names
            )
        else:
            ground_truth = pd.Series(
                np.repeat(np.array([True, False]), [m_null, m_alt]), index=drug_names
            )

    # Create (non) amensia magnitude
    dar = p0 * ground_truth + p1 * ~ground_truth
    return dar, ground_truth


def assemble_fake_pois(
    m_null: int,
    interleaved: bool,
    p0: float,
    p1: float,
    m_alt: Optional[int] = None,
) -> (pd.Series, pd.Series):
    """Assembles dar and dnar for fake drugs.

    Args:
        m_null (int): number of true nulls.
        interleaved (bool): whether to interleave null and alternative hypotheses.
        p0 (float): null poisson rate.
        p1 (float): alternative poisson rate.
        m_alt (Optional[int], optional): number of true alternatives. Defaults to None,
            which means m_alt = m_null.

    Returns:
        (pd.Series, pd.Series): tuple of poisson rate series and ground truth series.
    """
    if m_alt is None:
        drug_names = gen_names(2 * m_null)
    else:
        drug_names = gen_names(m_null + m_alt)

    # Create null/alternative masks, and total magnitude series
    if interleaved:
        if m_alt is None:
            ground_truth = pd.Series(
                np.tile(np.array([True, False]), m_null), index=drug_names
            )
        else:
            ground_truth = pd.Series(
                build_interleaved_ground_truth(m_null, m_alt), index=drug_names
            )
    else:
        if m_alt is None:
            ground_truth = pd.Series(
                np.repeat(np.array([True, False]), m_null), index=drug_names
            )
        else:
            ground_truth = pd.Series(
                np.repeat(np.array([True, False]), [m_null, m_alt]), index=drug_names
            )
    # Create (non) amensia magnitude
    dar = p0 * ground_truth + p1 * ~ground_truth
    return dar, ground_truth


def assemble_fake_pois_grad(m_null, p0, p1, m_alt=None):
    """Assembles dar and dnar for fake drugs."""
    if m_alt is None:
        m_alt = m_null
    lam0 = min((p0, p1))
    lam1 = max((p0, p1))
    lam_ratio = lam0 / lam1
    log_low_lam = np.log(lam0) + np.log(lam_ratio)
    log_high_lam = np.log(lam1) - np.log(lam_ratio)
    dar_vals = np.exp(np.linspace(log_low_lam, log_high_lam, m_null + m_alt))
    drug_names = gen_names(m_null + m_alt)
    #    drug_names = list(map(lambda u,v: u + v,
    #                     np.array(list(string.ascii_letters))[np.arange(0, 4 * m_null, 2)],
    #                     np.array(list(string.ascii_letters))[np.arange(1, 4 * m_null, 2)]))

    return pd.Series(dar_vals, index=drug_names)


def simulate_reactions(
    drug_amnesia_rate: pd.Series,
    drug_nonamnesia_rate: pd.Series,
    n_periods: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Given series of drug amnesia and non-amnesia rates, gerenates data.

    args:
        drug_amnesia_rate: Series
        drug_nonamnesia_rate: Series
        n_periods: int
    return:
        tuple of DataFrames of reactions. Columns are drugs, rows are periods.
    """
    sim_amnesia_reactions = pd.DataFrame(
        dict(
            [
                (drug_name, poisson.rvs(individual_drug_rate, size=n_periods))
                if individual_drug_rate > 0
                else (drug_name, np.zeros(n_periods))
                for drug_name, individual_drug_rate in drug_amnesia_rate.items()
            ]
        )
    ).cumsum()

    sim_nonamnesia_reactions = pd.DataFrame(
        dict(
            [
                (drug_name, poisson.rvs(individual_drug_rate, size=n_periods))
                if individual_drug_rate > 0
                else (drug_name, np.zeros(n_periods))
                for drug_name, individual_drug_rate in drug_nonamnesia_rate.items()
            ]
        )
    ).cumsum()
    return (
        sim_amnesia_reactions.reindex(columns=drug_amnesia_rate.index),
        sim_nonamnesia_reactions.reindex(columns=drug_nonamnesia_rate.index),
    )


# def simulate_gaussian_noncum_internal(gauss_moments_df, n_periods):
#     out_dict = {}
#     for hyp_name, hyp_data in gauss_moments_df.iterrows():
#         out_dict[hyp_name] = hyp_data["mean"] + hyp_data["sd"]*numpy.random.randn(n_periods)
#     return pd.DataFrame(out_dict).reindex(columns=gauss_moments_df.index)

# def simulate_gaussian_noncum(dar, dnar, n_periods):
#     return simulate_gaussian_noncum_internal(pd.DataFrame({"mean":dar, "sd":dnar}), n_periods)


def simulate_binom(bin_props: pd.Series, n_periods: int) -> pd.DataFrame:
    """Generates cumulative success counts for binomial processes."""
    out_dict = {}
    for hyp_name, hyp_prop in bin_props.items():
        out_dict[hyp_name] = binom.rvs(1, hyp_prop, size=n_periods)
    return pd.DataFrame(out_dict).cumsum().reindex(columns=bin_props.index)


def simulate_pois(pois_rates: pd.Series, n_periods:int)->pd.DataFrame:
    return (
        pd.DataFrame(
            dict(
                [
                    (hyp_name, poisson.rvs(hyp_rate, size=n_periods))
                    for hyp_name, hyp_rate in pois_rates.items()
                ]
            )
        )
        .cumsum()
        .reindex(columns=pois_rates.index)
    )


# def assemble_llr(amnesia, nonamnesia, p0, p1):
#     """deprecated. Use assemble_drug_llr.
#     np.log(L(p1)/L(p0))
#     E[llr|H0] > 0
#     E[llr|H1] < 0
#     """
#     warnings.warn("assemble_llr is deprecated. Use assemble_drug_llr.")
#     return amnesia * np.log(p1/p0) + nonamnesia * np.log((1 - p1) / (1 - p0))


def assemble_drug_llr(counts, p0, p1):
    """
    counts: 2-tuple, amnesia count vec and non amnesia count vec
    p0:
    p1:


    np.log(L(p1)/L(p0))
    E[llr|H0] > 0
    E[llr|H1] < 0
    """
    return counts[0] * np.log(p1 / p0) + counts[1] * np.log((1 - p1) / (1 - p0))


def assemble_binom_llr(bin_count, p0, p1):
    """np.log(L(p1)/L(p0))
    E[llr|H0] > 0
    E[llr|H1] < 0
    """
    pos_count = bin_count
    neg_count = (np.arange(1, len(bin_count) + 1))[:, np.newaxis] - bin_count
    return pos_count * np.log(p1 / p0) + neg_count * np.log((1 - p1) / (1 - p0))


# def assemble_gaussian_llr(samps_noncum, p0, p1):
#     """np.log(L(p1)/L(p0))
#     E[llr|H0] > 0
#     E[llr|H1] < 0
#     """
#     samps_idx = np.arange(1.0, len(samps_noncum) + 1)
#     est_var0 = (1.0 / samps_idx[:, numpy.np.newaxis]) * ((samps_noncum - p0)**2.0).cumsum()
#     est_var1 = (1.0 / samps_idx[:, numpy.np.newaxis]) * ((samps_noncum - p1)**2.0).cumsum()
#     return -.5 * samps_idx[:, numpy.np.newaxis] * (np.log(est_var1) - np.log(est_var0))


def assemble_pois_llr(pois_count, lam0, lam1):
    """np.log(L(p1)/L(p0))
    E[llr|H0] > 0
    E[llr|H1] < 0
    """
    period_idx = pd.DataFrame(
        np.ones(pois_count.shape), columns=pois_count.columns, index=pois_count.index
    ).cumsum()

    return (np.log(lam1 / lam0) * pois_count) - (period_idx * (lam1 - lam0))


def generate_llr(
    dar,
    dnar,
    n_periods,
    rho,
    hyp_type,
    p0,
    p1,
    m1=None,
    rho1=None,
    rand_order=False,
    cummax=False,
):
    if (hyp_type is None) or (hyp_type == "drug"):
        amnesia, nonamnesia = simulate_correlated_reactions(
            dar, dnar, n_periods, rho, m1, rho1, rand_order=rand_order
        )
        llr = assemble_drug_llr((amnesia, nonamnesia), p0, p1)
    elif hyp_type == "binom":
        event_count = simulate_correlated_binom(
            dar, n_periods, rho, m1, rho1, rand_order=rand_order
        )
        llr = assemble_binom_llr(event_count, p0, p1)
    elif hyp_type == "pois":
        event_count = simulate_correlated_pois(
            dar, n_periods, rho, m1, rho1, rand_order=rand_order
        )
        llr = assemble_pois_llr(event_count, p0, p1)
    elif hyp_type == "gaussian":
        raise NotImplementedError("Gaussian LLR not implemented yet.")
        # gvals = simulate_correlated_gaussian_noncum(dar, dnar, n_periods,
        #                                                        rho, m1, rho1, rand_order=rand_order)
        # llr = assemble_gaussian_llr(gvals, p0, p1)
    else:
        raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
    if cummax:
        return llr.cummax()
    else:
        return llr


def whole_data_p0(N_amnesia, N_reports):
    """
    Estimate of average proportion of side effects that are amnesia for all
    drugs combined.
    """
    return float(N_amnesia) / N_reports


def whole_data_p1(N_amnesia, N_reports, p0, n_se=2.0):
    """Get p1 above p0 based on total reports SE for p0 estimate"""
    return p0 + n_se * np.sqrt(p0 * (1 - p0) / N_reports)


def drug_data_p1(N_drugs, p0, n_se=2.0):
    """Get p1 above p0 based on total drugs SE for p0 estimate"""
    return p0 + n_se * np.sqrt(p0 * (1 - p0) / N_drugs)


def am_prop_percentile_p0_p1(dar, dnar, p0_pctl, p1_pctl):
    """Get p0 and p1 as percentiles of amnesia proportion"""
    am_prop = dar / (dar + dnar)
    return np.percentile(am_prop, [100 * p0_pctl, 100 * p1_pctl])


def toep_corr_matrix(m, rho, m1=None, rho1=None, rand_order=False):
    if m1 is None:
        raw_corr_mat = toeplitz(rho ** abs(np.arange(m)))
        if rand_order:
            ordering = numpy.random.permutation(np.arange(m))
            print(ordering)
            corr_mat = (raw_corr_mat[ordering, :])[:, ordering]
            return corr_mat
        else:
            return raw_corr_mat
    else:
        if rand_order:
            warnings.warn(
                "Both seperate correlation structures and random "
                "ordering were passed.\n Ignoring random ordering."
            )
        m0 = m - m1
        rho0 = rho
        if rho1 is None:
            rho1 = rho
        UL = toep_corr_matrix(m0, rho0)
        BR = toep_corr_matrix(m1, rho1)
        UR = np.zeros((m0, m1))
        BL = np.zeros((m1, m0))
        outmat = np.vstack([np.hstack([UL, UR]), np.hstack([BL, BR])])
        #        print(outmat.round(1))
        return outmat


def simulate_correlated_reactions(
    drug_amnesia_rate,
    drug_nonamnesia_rate,
    n_periods,
    rho,
    m1=None,
    rho1=None,
    rand_order=False,
):
    """ """
    if rho is None:
        return simulate_reactions(drug_amnesia_rate, drug_nonamnesia_rate, n_periods)
    else:
        # TODO: fix
        cov_mat = toep_corr_matrix(
            len(drug_amnesia_rate), rho, m1, rho1, rand_order=rand_order
        )
        # toeplitz(rho** abs(np.arange(len(drug_amnesia_rate))))
        # cov_mat[np.arange(0, N_reports, 5)[:,np.newaxis], np.arange(0, N_reports, 5)] = cov_mat[np.arange(0, N_reports, 5)[:,np.newaxis], np.arange(0, N_reports, 5)]**0.5
        uuA = pd.DataFrame(
            norm.cdf(multivariate_normal(cov=cov_mat).rvs(size=n_periods)),
            columns=drug_amnesia_rate.index,
        )
        uuB = pd.DataFrame(
            norm.cdf(multivariate_normal(cov=cov_mat).rvs(size=n_periods)),
            columns=drug_amnesia_rate.index,
        )

        # Running total of reaction reports
        # Iterates through drugs, generating random samples of size n_periods for
        # each drug with nonzero rate.
        sim_amnesia_reactions = pd.DataFrame(
            dict(
                [
                    (drug_name, poisson.ppf(uuA[drug_name], individual_drug_rate))
                    if individual_drug_rate > 0
                    else (drug_name, np.zeros(n_periods))
                    for drug_name, individual_drug_rate in drug_amnesia_rate.items()
                ]
            )
        ).cumsum()

        sim_nonamnesia_reactions = pd.DataFrame(
            dict(
                [
                    (drug_name, poisson.ppf(uuB[drug_name], individual_drug_rate))
                    if individual_drug_rate > 0
                    else (drug_name, np.zeros(n_periods))
                    for drug_name, individual_drug_rate in drug_nonamnesia_rate.items()
                ]
            )
        ).cumsum()
        return (
            sim_amnesia_reactions.reindex(columns=drug_amnesia_rate.index),
            sim_nonamnesia_reactions.reindex(columns=drug_nonamnesia_rate.index),
        )


# def simulate_correlated_gaussian_noncum_internal(gauss_moments_df, n_periods, rho,
#                                                  m1=None, rho1=None, rand_order=False):
#     sdvec = gauss_moments_df["sd"].values
#     var_mat = sdvec[:, numpy.np.newaxis] * sdvec[numpy.np.newaxis, :]
#     cov_mat = toep_corr_matrix(len(gauss_moments_df), rho, m1, rho1, rand_order=rand_order) * var_mat
#     #toeplitz(rho** abs(np.arange(len(gauss_moments_df))))
#     mean_vec = gauss_moments_df["mean"].values
#     sim_values = multivariate_normal(mean=mean_vec, cov=cov_mat).rvs(size=n_periods)
#     return pd.DataFrame(sim_values, columns = gauss_moments_df.index)

# def simulate_correlated_gaussian_noncum(dar, dnar, n_periods, rho,
#                                         m1=None, rho1=None, rand_order=False):
#     return simulate_correlated_gaussian_noncum_internal(pd.DataFrame({"mean":dar, "sd":dnar}), n_periods, rho,
#                                                         m1=m1, rho1=rho1, rand_order=rand_order)


def simulate_correlated_binom(
    bin_props, n_periods, rho, m1=None, rho1=None, rand_order=False
):
    cov_mat = toep_corr_matrix(len(bin_props), rho, m1, rho1, rand_order=rand_order)
    # toeplitz(rho** abs(np.arange(len(bin_props))))
    # cov_mat[np.arange(0, N_reports, 5)[:,np.newaxis], np.arange(0, N_reports, 5)] = cov_mat[np.arange(0, N_reports, 5)[:,np.newaxis], np.arange(0, N_reports, 5)]**0.5
    uuA = pd.DataFrame(
        norm.cdf(multivariate_normal(cov=cov_mat).rvs(size=n_periods)),
        columns=bin_props.index,
    )
    return (
        pd.DataFrame(
            dict(
                [
                    (hyp_name, binom.ppf(uuA[hyp_name], 1, hyp_prop))
                    for hyp_name, hyp_prop in bin_props.items()
                ]
            )
        )
        .cumsum()
        .reindex(columns=bin_props.index)
    )


def simulate_correlated_pois(
    pois_rates, n_periods, rho, m1=None, rho1=None, rand_order=False
):
    cov_mat = toep_corr_matrix(len(pois_rates), rho, m1, rho1, rand_order=rand_order)
    # toeplitz(rho** abs(np.arange(len(pois_rates))))
    # cov_mat[np.arange(0, N_reports, 5)[:,np.newaxis], np.arange(0, N_reports, 5)] = cov_mat[np.arange(0, N_reports, 5)[:,np.newaxis], np.arange(0, N_reports, 5)]**0.5
    uuA = pd.DataFrame(
        norm.cdf(multivariate_normal(cov=cov_mat).rvs(size=n_periods)),
        columns=pois_rates.index,
    )
    return (
        pd.DataFrame(
            dict(
                [
                    (hyp_name, poisson.ppf(uuA[hyp_name], hyp_rate))
                    for hyp_name, hyp_rate in pois_rates.items()
                ]
            )
        )
        .cumsum()
        .reindex(columns=pois_rates.index)
    )


# %% streaming code
# TODO(mhankin): make .df a reference to the updated dataframe... somehow
class online_data(object):
    """streaming data interface class"""

    def __init__(self, col_list: List[str], dgp):
        self._dgp = dgp
        if isinstance(col_list, pd.Index):
            self._columns = col_list
        else:
            self._columns = pd.Index(col_list)

        self._dead_cols = []

    def __len__(self):
        return len(self._dgp)

    def _get_columns(self):
        return self._columns.copy()

    def drop(self, col_list, *args, **kwargs):
        extra_cols = set(col_list).difference(set(self._columns))
        if extra_cols:
            raise ValueError("Unknown columns: {0}".format(list(extra_cols)))
        else:
            self._dead_cols.extend(col_list)
            self._columns = self._columns.drop(col_list)

    def iterrows(self):
        return online_data_generator(self, self._dgp)

    columns = property(_get_columns)


class online_data_generator(object):
    """generator class for yielding data subject to the active columns in its parent

    only used by online_data class
    """

    def __init__(self, parent:online_data, dgp: Callable[[List[str]], pd.Series]):
        self._parent = parent
        self._dgp = dgp
        self._current_index = -1

    def __iter__(self) -> "online_data_generator":
        return self

    def __next__(self) -> Tuple[int, pd.Series]:
        return self.next()

    def next(self) -> Tuple[int, pd.Series]:
        self._current_index = self._current_index + 1
        idx = self._current_index
        data_ser = self._dgp(self._parent.columns)
        return (idx, data_ser)


class df_dgp_wrapper(object):
    """wraps a pandas DataFrame to make it fit the dgp interface"""

    def __init__(self, df: pd.DataFrame):
        self._df = df
        self._iter_rows = df.iterrows()

    def __call__(self, col_list: List[str]) -> pd.Series:
        _, data_ser = next(self._iter_rows)
        if data_ser is None:
            StopIteration()
        return data_ser[col_list]

    def get_data_record(self) -> pd.DataFrame:
        return self._df

    def __len__(self) -> int:
        return len(self._df)


def df_generator(df: pd.DataFrame) -> online_data:
    """Wraps a DataFrame in an online_data object, ready to go"""
    return online_data(df.columns.copy(), df_dgp_wrapper(df.copy()))


class infinite_dgp_wrapper(df_dgp_wrapper):
    """Creates a dgp that will continuously generate minibatches of data as needed"""

    def __init__(self, gen_llr_kwargs, drop_old_data=True):
        self._gen_kwargs = gen_llr_kwargs
        self._df = generate_llr(**gen_llr_kwargs)
        self._iter_rows = self._df.iterrows()
        self._drop_old_data = drop_old_data

    def __call__(self, col_list):
        try:
            _, data_ser = next(self._iter_rows)
        except StopIteration as ex:
            last_val = self._df.iloc[-1]
            new_df = generate_llr(**self._gen_kwargs) + last_val
            if self._drop_old_data:
                self._df = new_df
            else:
                self._df = pd.concat((self._df, new_df))
                self._df.reset_index(inplace=True, drop=True)
            self._iter_rows = new_df.iterrows()
            _, data_ser = next(self._iter_rows)
        return data_ser[col_list]

    def get_data_record(self):
        print("get_data_record")
        if self._drop_old_data:
            raise ValueError("DGP drops old data. Cannot return full record.")
        else:
            return self._df
