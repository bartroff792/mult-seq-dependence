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

from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Protocol, Tuple, Union, runtime_checkable
import pandas as pd
import datetime, os, string, warnings
import numpy as np
import numpy.random
from scipy.linalg import toeplitz
import itertools
# from scipy.stats import norm, multivariate_normal, poisson, binom
from scipy import stats
import hashlib
# import logging
from dataclasses import dataclass


def gen_names(n_hyps: int) -> pd.Index:
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
    return pd.Index(
        [
            ("{0:0=" + str(digits) + "d}-{1}").format(
                i, hashlib.sha1(str(i).encode("utf-8")).hexdigest()[:STR_LEN]
            )
            for i in range(n_hyps)
        ],
        name="hyp_name",
    )


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


# New versions
def construct_dgp(
    m_null: int,
    m_alt: int,
    theta0: float,
    theta1: float,
    interleaved: bool,
    hyp_type: Literal["binom", "pois"],
    extra_params: Optional[Dict[str, float]] = None,
) -> Tuple[Dict[str, pd.Series], pd.Series]:
    """Construct generating parameters and ground truth for a given hypothesis type.

    Args:
        m_null (int): number of true null hypotheses
        m_alt (int): number of true alternative hypotheses
        theta0 (float): test parameter value for nulls.
        theta1 (float): test parameter value for alternatives.
        interleaved (bool): whether to interleave null and alternative hypotheses. if true, hyp0
            will but true, hyp1 will be false, etc until 2 * min(m_null, m_alt), etc. Otherwise, 0 through m_null-1 will be true and the rest false.
        hyp_type (Literal[&quot;drug&quot;, &quot;binom&quot;, &quot;pois&quot;]): only supports binom or pois for now.
        extra_params (Dict[str, Union[float, pd.Series]]): any additional parameters.

    Returns:
        Tuple[Dict[str, pd.Series], pd.Series]: _description_
    """
    if extra_params is None:
        extra_params = {}
    if m_alt is None:
        m_alt = m_null
    hyp_names = gen_names(m_null + m_alt)

    if interleaved:
        ground_truth = pd.Series(
            build_interleaved_ground_truth(m_null, m_alt), index=hyp_names
        )
    else:
        ground_truth = pd.Series(
            np.repeat(np.array([True, False]), [m_null, m_alt]), index=hyp_names
        )
    if hyp_type == "binom":
        binom_probs = theta0 * ground_truth + theta1 * ~ground_truth
        n_events = pd.Series(
            np.repeat(int(extra_params["n"]), len(binom_probs)), index=hyp_names
        )
        if len(extra_params) > 1:
            warnings.warn("Extra parameters not used.")
        return {"p": binom_probs, "n": n_events}, ground_truth
    elif hyp_type == "pois":
        pois_rate = theta0 * ground_truth + theta1 * ~ground_truth
        if len(extra_params) > 0:
            warnings.warn("Extra parameters not used.", UserWarning)
        return {"mu": pois_rate}, ground_truth
    elif hyp_type == "drug":
        total_reaction_rate = pd.Series(
            np.repeat(extra_params["mu"], m_null + m_alt), index=hyp_names
        )
        proportion_amnesia = theta0 * ground_truth + theta1 * ~ground_truth
        if len(extra_params) > 1:
            warnings.warn("Extra parameters not used.", UserWarning)
        return {"mu": total_reaction_rate, "p": proportion_amnesia}, ground_truth
    else:
        raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))


def check_params(
    hyp_type: Literal["drug", "pois", "binom"], params: Dict[str, pd.Series]
) -> pd.Index:
    """Confirms that the right keys are present in the params dictionary, and returns the names of the hypotheses.

    Args:
        hyp_type (Literal[&quot;drug&quot;, &quot;pois&quot;, &quot;binom&quot;]): _description_
        params (Dict[str, pd.Series]): _description_

    Raises:
        ValueError: _description_

    Returns:
        pd.Index: index of hypothesis names
    """
    if hyp_type == "drug":
        assert len(params) == 2, "Drug type must have exactly 2 parameters."
        assert (
            "mu" in params
        ), "Drug type must have an overall reaction rate 'mu' parameter."
        assert (
            "p" in params
        ), "Drug type must have a proportion of reactions at that are relevant 'p' parameter."
        assert len(params["mu"]) == len(
            params["p"]
        ), "amnesia_rate and non_amnesia_rate must have the same length."
        return (params["mu"]).index
    elif hyp_type == "pois":
        assert len(params) == 1, "Poisson type must have exactly 1 parameter."
        assert "mu" in params, "Poisson type must have a lambda parameter."
        return (params["mu"]).index
    elif hyp_type == "binom":
        assert len(params) == 2, "Binomial type must have exactly 2 parameters."
        assert "p" in params, "Binomial type must have a p parameter."
        assert "n" in params, "Binomial type must have a n parameter."
        assert len(params["p"]) == len(
            params["n"]
        ), "n and p must have the same length."
        return (params["p"]).index
    else:
        raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))


def simple_toeplitz_corr_mat(
    rho: float, m: int, rand_order: bool = False
) -> np.ndarray:
    """Construct correlation matrix with symmetric neighboring connections."""
    raw_corr_mat = toeplitz(rho ** abs(np.arange(m)))
    if rand_order:
        ordering = numpy.random.permutation(np.arange(m))
        corr_mat = (raw_corr_mat[ordering, :])[:, ordering]
        return corr_mat
    else:
        return raw_corr_mat

@dataclass
class CorrelationContainer:
    rho: pd.Series # maps group numbers to correlation coefficients
    group_ser: pd.Series # maps hypothesis names to group numbers

def copula_draw(
    hyp_idx: pd.Index,
    n_periods: int, 
    rho: Union[float, CorrelationContainer], 
    rand_order: bool = False,
) -> pd.DataFrame:
    """Draws from a gaussian copula with a given correlation matrix.

    Args:
        hyp_idx (pd.Index): index of hypothesis names.
        n_periods (int): number of periods to simulate.
        rho (Union[float, CorrelationContainer]): correlation coefficient. If a float, the correlation
            matrix will be a simple toeplitz matrix with this value. If a CorrelationContainer, the
            correlation matrix will be block diagonal with each block having the same correlation
            coefficient. The blocks are defined by the group_ser attribute of the CorrelationContainer.
        rand_order (bool, optional): whether to randomly order the drugs. Defaults to False.

    Returns:
        pd.DataFrame: draws from a multivariate normal with the specified correlation matrix.
    """
    if isinstance(rho, float):
        cov_mat = simple_toeplitz_corr_mat(rho, len(hyp_idx), rand_order)
        unif_draw = pd.DataFrame(
            stats.norm.cdf(stats.multivariate_normal(cov=cov_mat).rvs(size=n_periods)),
            columns=hyp_idx,
        )
    elif isinstance(rho, CorrelationContainer):
        unif_draw_list = []
        for group_number, rho_val in rho.rho.items():
            group_hyps = rho.group_ser[rho.group_ser==group_number].index
            group_size = len(group_hyps)
            cov_mat = simple_toeplitz_corr_mat(rho_val, group_size, rand_order)
            group_unif_draw = pd.DataFrame(
                stats.norm.cdf(stats.multivariate_normal(cov=cov_mat).rvs(size=n_periods)),
                columns=group_hyps,
            )
            unif_draw_list.append(group_unif_draw)
        unif_draw = pd.concat(unif_draw_list, axis=1)
    else:
        raise ValueError("Unrecognized correlation type: {0}".format(rho))
    unif_draw.index.name = "period"
    return unif_draw

def simulate_correlated_observations(
    params: Dict[str, pd.Series],
    n_periods: int,
    rho: Union[float, CorrelationContainer],
    hyp_type: Literal["binom", "pois", "drug"],
    rand_order: bool = False,
) -> Dict[str, pd.DataFrame]:
    # Get full index of hypothesis names
    hyp_idx = check_params(hyp_type, params)
    # Draw from the copula.
    unif_draw = copula_draw(hyp_idx, n_periods, rho, rand_order)
    if hyp_type == "binom":
        dist = stats.binom(**params)
        obs = {
            "obs": pd.DataFrame(
                dist.ppf(unif_draw), columns=hyp_idx, index=unif_draw.index,
            )
        }
    elif hyp_type == "pois":
        dist = stats.poisson(**params)
        obs = {
            "obs": pd.DataFrame(
                dist.ppf(unif_draw), columns=hyp_idx, index=unif_draw.index,
            )
        }
    elif hyp_type == "drug":
        total_rate_dist = stats.poisson(params["mu"])
        # Copula draw for the rates
        rate_unif_draw = copula_draw(hyp_idx, n_periods, rho, rand_order)
        total_obs = pd.DataFrame(
            total_rate_dist.ppf(rate_unif_draw), columns=hyp_idx, index=unif_draw.index
        )
        # Use the main copula draws for the proportion of relevant events
        relevant_events_dist = stats.binom(total_obs.astype(int), params["p"])
        relevant_events_counts = pd.DataFrame(
            relevant_events_dist.ppf(unif_draw),
            columns=hyp_idx,
            index=unif_draw.index,
        )
        non_relevant_events_counts = total_obs - relevant_events_counts
        obs = {
            "relevant_events": relevant_events_counts,
            "non_relevant_events": non_relevant_events_counts,
        }
    else:
        raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))

    return obs


def compute_llr(
    observed_data: Dict[str, pd.DataFrame],
    hyp_type: Literal["drug", "binom", "pois", "gaussian"],
    params0: Dict[str, pd.Series],
    params1: Dict[str, pd.Series],
    cumulative=True,
) -> pd.DataFrame:
    """Computes the log likelihood ratio for each hypothesis at each timestep.

    Computes manually using hard derived formulae. Warning: this *always* expects the entries in
    the dataframes in observed_data to be the new observations only, NOT the cumulative observations.
    The `cumulative` argument means that the cumulative LLR should be returned (default). When
    False, the entries in the returned data frame are the stepwise LLRs.

    Args:
        observed_data (pd.DataFrame): Observed data. New observation at each timestep, not cumulative.
        hyp_type (Literal["drug", "binom", "pois", "gaussian"]): Type of hypothesis to generate data for.
        theta0 (Union[float, pd.Series]): Null hypothesis parameter.
        theta1 (Union[float, pd.Series]): Alternative hypothesis parameter.
        extra_params (Optional[Dict[str, Any]]): Extra parameters for the hypothesis.
        cumulative (bool, optional): Whether to return cumulative LLR. Defaults to True.

    Returns:
        pd.DataFrame: LLR paths. Index is the period, columns are the hypotheses.
    """
    # Confirm that all observation dataframes have the same shape
    if len(observed_data)>1:
        for entryA, entryB in itertools.combinations(observed_data.items(), 2):
            kkA, vvA = entryA
            kkB, vvB = entryB
            assert vvA.shape==vvB.shape, f"Shapes of {kkA} and {kkB} do not match: {vvA.shape} and {vvB.shape}."
    
    # Grab an arbitrary entry to get the time index
    obs_entry = next(iter(observed_data.values()))
    time_idx = pd.DataFrame(
        pd.Series(np.arange(1, len(obs_entry) + 1), index=obs_entry.index)
    )
    # Extract the observations and parameters for each type of test and compute the LLR.
    if hyp_type == "binom":
        if cumulative:
            pos_obs = observed_data["obs"].cumsum()
            total_obs = time_idx.dot(pd.DataFrame(params0["n"]).T)
            neg_obs = total_obs - pos_obs
        else:
            pos_obs = observed_data["obs"]
            neg_obs = params0["n"].subtract(pos_obs)
            assert (
                pos_obs.shape == neg_obs.shape
            ), f"Positive and negative observations shapes do not match: {pos_obs.shape} and {neg_obs.shape}."
        llr = pos_obs.multiply(np.log(params1["p"] / params0["p"])) + neg_obs.multiply(
            np.log((1 - params1["p"]) / (1 - params0["p"]))
        )
    elif hyp_type == "drug":
        if cumulative:
            pos_obs = observed_data["relevant_events"].cumsum()
            neg_obs = observed_data["non_relevant_events"].cumsum()
            assert (pos_obs.shape == neg_obs.shape), f"Positive and negative observations shapes do not match: {pos_obs.shape} and {neg_obs.shape}."
        else:
            pos_obs = observed_data["relevant_events"]
            neg_obs = observed_data["non_relevant_events"]
            assert (
                pos_obs.shape == neg_obs.shape
            ), f"Positive and negative observations shapes do not match: {pos_obs.shape} and {neg_obs.shape}."
        llr = pos_obs.multiply(np.log(params1["p"] / params0["p"])) + neg_obs.multiply(
            np.log((1 - params1["p"]) / (1 - params0["p"]))
        )
    elif hyp_type == "pois":
        if cumulative:
            obs = observed_data["obs"].cumsum()
            scaled_rate = time_idx.dot(pd.DataFrame(params1["mu"] - params0["mu"]).T)
        else:
            obs = observed_data["obs"]
            scaled_rate = params1["mu"] - params0["mu"]
        llr = obs.multiply(np.log(params1["mu"] / params0["mu"])).subtract(scaled_rate)
    else:
        raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
    for observed_entry_name, observed_entry_df in observed_data.items():
        assert (
            llr.shape == observed_entry_df.shape
        ), f"LLR and observed data {observed_entry_name} shapes do not match: {llr.shape} and {observed_entry_df.shape}."
    return llr


def generate_llr(
    params: Dict[str, pd.Series],
    n_periods: int,
    rho: Optional[float],
    hyp_type: Literal["drug", "binom", "pois", "gaussian"],
    params0: Dict[str, pd.Series],
    params1: Dict[str, pd.Series],
    rand_order: bool = False,
    # extra_params: Optional[Dict[str, Any]]=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main func for generating data and llr paths.

    Args:
        params (Dict[str, pd.Series]): Dictionary of relevant parameters for dgp. Must have keys corresponding to the required values in hyp_type.
        n_periods (int): Number of periods to simulate.
        rho (Optional[float]): Correlation coefficient. Defaults to None.
        hyp_type (Literal["drug", "binom", "pois", "gaussian"]): Type of hypothesis to generate data for.
        theta0 (Union[float, pd.Series]): Null hypothesis parameter.
        theta1 (Union[float, pd.Series]): Alternative hypothesis parameter.
        rand_order (bool, optional): Whether to randomly order drugs. Defaults to False.
        extra_params (Optional[Dict[str, Any]], optional): Extra parameters for the hypothesis.
            Constant across hypotheses and true for both null and alternative.

    Returns:
        Tuple of 2 pd.DataFrames: LLR paths and observed data, respectively
    """
    check_params(hyp_type, params)

    observed_data = simulate_correlated_observations(
        params, n_periods, rho, hyp_type, rand_order=rand_order
    )
    llr = compute_llr(
        observed_data, hyp_type, params0=params0, params1=params1, cumulative=True
    )
    for observed_entry_name, observed_entry_df in observed_data.items():

        assert (
            llr.shape == observed_entry_df.shape
        ), f"LLR and observed data {observed_entry_name} shapes do not match: {llr.shape} and {observed_entry_df.shape}."
    return llr, observed_data


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



# %% streaming code
# TODO(mhankin): make .df a reference to the updated dataframe... somehow


# statistics interface
# if not a dataframe must have the following atts
# statistics.columns must list hypothesis names of ACTIVE hypotheses
# statistics.iterrows() must return the generator object that emits (step_number, step_data_series)
# statistics.drop(list_of_cols, **kwargs) must cause the ensuing yield statements
#       to omit list_of_cols, and ignore kwargs

@runtime_checkable
class PareableStatisticsStreamer(Protocol):
    def get_columns(self) -> List[str]:
        """Return list of ACTIVE hypotheses only."""
        pass

    def iterrows(self) -> Iterator[Tuple[int, pd.Series]]:
        """Return generator that emits (step_number, step_data_series)"""
        pass

    def drop(self, col_list: List[str], *args, **kwargs):
        """Cause the ensuing yield iterrows statements to omit list_of_cols, and ignore kwargs"""
        pass

# simulation_orchestration: online_data,infinite_dgp_wrapper, df_dgp_wrapper
# multseq: df_generator
class online_data(object):
    """streaming data interface class
    
    Implements the PareaableStatisticsStreamer protocol."""

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
    
    def get_columns(self):
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

    def __init__(self, parent: online_data, dgp: Callable[[List[str]], pd.Series]):
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
        self._llr_df, self._obs_dict = generate_llr(**gen_llr_kwargs)
        for obs_entry_name, obs_entry_df in self._obs_dict.items():
            assert (
                self._llr_df.shape == obs_entry_df.shape
            ), "LLR and observed data must have the same shape."
        self._iter_rows = self._llr_df.iterrows()
        self._drop_old_data = drop_old_data

    def __call__(self, col_list):
        try:
            _, data_ser = next(self._iter_rows)
        except StopIteration as ex:
            # TODO: This seems sketchy... what's happening here
            final_llr_row = self._llr_df.iloc[-1]
            llr_df, obs_data_dict = generate_llr(**self._gen_kwargs)
            llr_df = llr_df + final_llr_row
            if self._drop_old_data:
                self._llr_df = llr_df
            else:
                self._llr_df = pd.concat((self._llr_df, llr_df))
                self._llr_df.reset_index(inplace=True, drop=True)
            self._iter_rows = llr_df.iterrows()
            _, data_ser = next(self._iter_rows)
        return data_ser[col_list]

    def get_data_record(self):
        print("get_data_record")
        if self._drop_old_data:
            raise ValueError("DGP drops old data. Cannot return full record.")
        else:
            return self._llr_df
