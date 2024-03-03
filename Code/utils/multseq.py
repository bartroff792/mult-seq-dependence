"""Main MultSeq code

This module contains the main functions for running sequential test of multiple hypotheses.

The main functions are:
    fellouris_synch_sprt: Fellouris synchronous SPRT. Currently broken.
    msprt: The main function for running multipl hypothesis SPRTs.
    naive_barrier_test: at a given step, takes the statistics of all 
        active hypotheses and all live barriers, and returns a boolean 
        array indicating for barrier i if i or more test statistics exceeded 
        it.
    step_up_elimination: runs a single step of a step-up elimination procedure.
    step_down_elimination: runs a single step of a step-down elimination procedure.


"""
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass

from . import cutoff_funcs, data_funcs

logger = logging.getLogger()
logger.setLevel(min((logging.WARNING, logger.level)))

HighLow = Union[Literal["high"], Literal["low"]]


def RejAccOtherFunc(hyp_term_data):
    if not np.isnan(hyp_term_data["rejLevel"]):
        return "rej"
    elif not np.isnan(hyp_term_data["accLevel"]):
        return "acc"
    else:
        return np.NaN


# %% Fellouris synchronous SPRT
# def fellouris_synch_sprt(llr, alpha=.1, beta=.35, m0=None, m0_range=(None, None), record_interval: int = 10, verbose=True):
#     """
#     Args:
#         llr (pd.DataFrame): Columns are hypotheses to be tested, rows are time steps,
#             and values are log likelihood ratio values
#         alpha (float): desired FDR (or pFDR, see pfdr arg) value
#         beta (float): desired FNR (or pFNR) value
#         m0: (int) number of true null hypotheses. If None, then m0_range must
#             be a valid 2-tuple.
#         m0_range: (2-tuple of ints) lower and upper bounds (inclusive) on the
#             number of true null hypotheses. If None, then m0 must be a valid
#             int.
#         record_interval (int): frequency at which the terminations should be reported and recorded
#         verbose (bool): whether to print progress updates
#     Return
#         fine_grained_outcomes (dict):
#         remaining: list of hypotheses never terminated
#         accepted: dictionary mapping steps to lists of hypotheses accepted at that step
#     """
#     m_hyps = len(llr.columns)

#     if m0 is not None:
#         c_val = np.min([np.log(m0 / alpha), np.log((m_hyps - m0) / beta)])
#     elif isinstance(m0_range[0], int) and isinstance(m0_range[1], int):
#         if 0 <= m0_range[0] and m0_range[0] <= m0_range[1] and m0_range[1] <= m_hyps:
#             # TODO
#             raise Exception("NOT YET IMPLEMENTED")
#         else:
#             raise ValueError("Bounds must be 0<=lower <= upper <= {0}. Received {1}".format(m_hyps, m0_range))
#     else:
#         raise ValueError("Must either pass m0 or m0_range")


#     # Step through each timestep, performing rejections and acceptances
#     for step in range(len(llr)):

#         # Record diagnostic data
#         if step %  record_interval == (record_interval - 1):
#             if verbose:
#                 # TODO: fix reporting
#                 logging.info("On step {0}. Accept: {1}. Reject: {2}, prop:{3}".format(step, j_accepted, j_rejected,
#                                                                            float(j_accepted+ j_rejected)/m_hyps))
#             step_record.append(step)
#             num_rejected_record.append(j_rejected)
#             num_accepted_record.append(j_accepted)
#             prop_terminated_record.append(float(j_accepted+ j_rejected) / m_hyps)

#         # Alias data for timestep
#         data_ser = llr.ix[step].sort_values()
#         num_pos = (data_ser>0).sum()
#         gap_stat = data_ser.ix[m0 + 1] - data_ser.ix[m0]

#     # No remaining, hypothesisTerminationData, or levelTripData kv pairs
#     fine_grained_outcomes = {'accepted':llr_accepted,
#                              'rejected':llr_rejected}
#     # Instead of terminated, this now captures just the number of positive and
#     # negative.
#     pos_neg_time_series =  pd.DataFrame(
#         {'neg':num_accepted_record, 'pos':num_rejected_record},
#             index=step_record)
#     # TODO: fill in cutoff returns
#     cutoff_output = (None,)
#     return  (fine_grained_outcomes,
#             termination_time_series, cutoff_output)


# %% Asynchronous SPRT

HypType = Any


@dataclass
class FineGrainedMSPRTOut:
    """Finegrained output from a multiple hypothesis SPRT

    Attrs:
        remaining (List[HypType]): list of hypotheses never terminated
        accepted (Dict[int, List[HypType]]): dictionary mapping steps to lists
             of hypotheses accepted at that step
        rejected (Dict[int, List[HypType]]): maps step numbers to lists of
            hypotheses rejected at that step
        hypTerminationData (pd.DataFrame): rows for each hypothesis, specifying
            its outcome (column `ar0`),
            the step at which it was accepted or rejected (np.NaN if neither),
            the significance level at which it was terminated.
        levelTripData (pd.DataFrame): `acc` and `rej` columns and rows for each level.
            Values are the step at which that level was tripped.
    """

    remaining: List[HypType]
    accepted: Dict[int, List[HypType]]
    rejected: Dict[int, List[HypType]]
    hypTerminationData: pd.DataFrame
    levelTripData: pd.DataFrame


@dataclass
class MSPRTOut:
    """High level output structure from an MSPRT run.

    Attrs:
        cutoffs (cutoff_funcs.CutoffDF): the pvalue and llr cutoffs used in
            the MSPRT procedure.
        fine_grained (FineGrainedMSPRTOut): details on the outcomes of the
            procedure.
        termination_ts (pd.DataFrame): record of terminations by time step
            index: step, frequency depends on record_interval
            Accepted: number of nulls accepted by timestep
            Rejected: number of nulls rejected by timestep
            'Prop Terminated': Accepted + Rejected / m_hyps

        record_interval (int): frequency at which terminations are assessed and reported.
        stepup (bool): whether the procedure was a stepup (or stepdown when false)
        rejective (bool): whether it was rejective only or also allowed acceptances.
    """

    cutoffs: cutoff_funcs.CutoffDF
    fine_grained: FineGrainedMSPRTOut
    termination_ts: pd.DataFrame  # TODO: replace with type alias, if not checked.
    record_interval: int
    stepup: bool
    rejective: bool
    full_llr: Optional[pd.DataFrame] = None


def msprt(
    statistics: pd.DataFrame,
    cutoffs: cutoff_funcs.CutoffDF,
    record_interval: int = 100,
    stepup: bool = False,
    rejective: bool = False,
    verbose: bool = True,
) -> MSPRTOut:
    """Runs a multiple hypothesis SPRT procedure.

    Args:
        statistics (pd.DataFrame): Columns are hypotheses to be tested, rows are time steps,
            and values are statistic values (generally log likelihood ratio values).
        cutoffs (cutoff_funcs.CutoffDF): statistic cutoffs. must always have `A` populated. `B` must
            also be populated if rejective is false. A>0, B<0.
        record_interval (int): frequency at which the terminations should be reported and recorded
        stepup (bool): Forces the test to use stepup rejection and acceptance,
            although cutoffs were set to control FDR for stepdown procedures

    Returns:
        A MSPRTOut object. See that class for details.

    """
    if isinstance(statistics, pd.DataFrame):
        # Copy data to prevent deletion
        statistics = data_funcs.df_generator(statistics.copy())
    else:
        assert isinstance(statistics, data_funcs.PareableStatisticsStreamer), "statistics must be a dataframe or implement the PareableStatisticsStreamer interface"

    if (not rejective) and (cutoffs["B"]).isna().any():
        raise ValueError("No B acceptance vector passed for acceptive-rejective test")
    if rejective: # and (B_vec is not None):
        # raise ValueError("B acceptance vector passed for rejective test")
        # TODO: raise a warning here
        pass

    # statistics interface:
    # statistics.get_columns() must list hypothesis names of ACTIVE hypotheses
    # statistics.iterrows() must return the generator object that emits (step_number, step_data_series)
    # statistics.drop(list_of_cols, **kwargs) must cause the ensuing yield statements
    #       to omit list_of_cols, and ignore kwargs

    # Calculate cutoffs
    hyp_names = statistics.get_columns()
    m_hyps = len(hyp_names)
    llr_iter = statistics.iterrows()

    # Diagnostics
    step_record = []
    num_rejected_record = []
    num_accepted_record = []
    prop_terminated_record = []

    # other setup
    termination_level = pd.DataFrame(
        {
            "step": np.NaN,
            "rejLevel": np.NaN,
            "accLevel": np.NaN,
        },
        index=hyp_names,
    )
    # For recording the time step at which each level was tripped
    acc_level_term = pd.Series(np.inf, index=np.arange(m_hyps))
    rej_level_term = pd.Series(np.inf, index=np.arange(m_hyps))
    llr_accepted = dict()
    llr_rejected = dict()
    j_accepted = 0
    j_rejected = 0

    # Step through each timestep, performing rejections and acceptances
    # data_ser is alias for timestep data series
    for step, data_ser in llr_iter:
        # Record diagnostic data
        if step % record_interval == (record_interval - 1):
            if verbose:
                logging.info(
                    "On step {0}. Accept: {1}. Reject: {2}, prop:{3}".format(
                        step,
                        j_accepted,
                        j_rejected,
                        float(j_accepted + j_rejected) / m_hyps,
                    ),
                )
            step_record.append(step)
            num_rejected_record.append(j_rejected)
            num_accepted_record.append(j_accepted)
            prop_terminated_record.append(float(j_accepted + j_rejected) / m_hyps)

        if not rejective:
            if stepup:
                accept_cols, num_new_accepts = step_up_elimination(
                    data_ser,
                    cutoffs["B"],
                    j_accepted,
                    highlow="low",
                )
            else:
                accept_cols, num_new_accepts = step_down_elimination(
                    data_ser,
                    cutoffs["B"],
                    j_accepted,
                    highlow="low",
                )
            if num_new_accepts > 0:
                if verbose:
                    logging.info("num new accepts >0" + str(step))
                acc_level_term[j_accepted : (j_accepted + num_new_accepts)] = step
                j_accepted = j_accepted + num_new_accepts
                # For each accepted hypothesis, record the step and level at which 
                # it was accepted
                termination_level.loc[accept_cols, "accLevel"] = j_accepted
                termination_level.loc[accept_cols, "step"] = step
                # Record all the hypotheses accepted at this step
                llr_accepted[step] = list(accept_cols.values)
                statistics.drop(accept_cols, axis=1, inplace=True)

                # Reset data_ser
                data_ser.drop(accept_cols, inplace=True)

        if stepup:
            reject_cols, num_new_rejects = step_up_elimination(
                data_ser,
                cutoffs["A"],
                j_rejected,
                highlow="high",
            )
        else:
            reject_cols, num_new_rejects = step_down_elimination(
                data_ser,
                cutoffs["A"],
                j_rejected,
                highlow="high",
            )

        if num_new_rejects > 0:
            if verbose:
                logging.info("num new rejects >0" + str(step))
            # Log the steps at which this these rejections occurred
            rej_level_term[j_rejected : (j_rejected + num_new_rejects)] = step
            j_rejected = j_rejected + num_new_rejects
            # For each rejected hypothesis, record the step and level at which it was rejected
            termination_level.loc[reject_cols, "rejLevel"] = j_rejected
            termination_level.loc[reject_cols, "step"] = step
            # Record all the hypotheses rejected at this step
            llr_rejected[step] = list(reject_cols.values)
            statistics.drop(reject_cols, axis=1, inplace=True)

            data_ser.drop(reject_cols, inplace=True)

        if len(statistics.get_columns()) == 0:
            logging.debug("Stopping early on step " + str(step))
            #            print("end\n", data_ser, "\n", j_rejected, j_accepted)
            break
    
    # WARNING! might break by skipping this
    # rej_level_term.replace(np.inf, step + 1, inplace=True)
    # acc_level_term.replace(np.inf, step + 1, inplace=True)
        
    # Record final diagnostic data
    if verbose:
        logging.info(
            "Final step {0}. Accept: {1}. Reject: {2}, prop:{3}".format(
                step + 1,
                j_accepted,
                j_rejected,
                float(j_accepted + j_rejected) / m_hyps,
            )
        )
    step_record.append(step)
    num_rejected_record.append(j_rejected)
    num_accepted_record.append(j_accepted)
    prop_terminated_record.append(float(j_accepted + j_rejected) / m_hyps)

    termination_level["ar0"] = termination_level.apply(RejAccOtherFunc, axis=1)
    if rejective:
        termination_level.loc[pd.isnull(termination_level["ar0"]), "ar0"] = "acc"

    fine_grained_outcomes = FineGrainedMSPRTOut(
        **{
            "remaining": list(statistics.get_columns()),
            "accepted": llr_accepted,
            "rejected": llr_rejected,
            #   'drugTerminationData':termination_level,
            "hypTerminationData": termination_level,
            "levelTripData": pd.DataFrame(
                {"acc": acc_level_term, "rej": rej_level_term}
            ),
        }
    )
    termination_time_series = pd.DataFrame(
        {
            "Accepted": num_accepted_record,
            "Rejected": num_rejected_record,
            "Prop Terminated": prop_terminated_record,
        },
        index=step_record,
    )
    return MSPRTOut(
        fine_grained=fine_grained_outcomes,
        termination_ts=termination_time_series,
        cutoffs=cutoffs,
        record_interval=record_interval,
        stepup=stepup,
        rejective=rejective,
    )


def naive_barrier_trips(
    data_ser: NDArray[np.float32],
    cutoff_vec: NDArray[np.float32],
    num_eliminated: int,
    highlow: HighLow,
) -> NDArray[np.bool_]:
    """Returns a boolean vector indicating whether or not the jth most
    significant active statistic tripped the jth active barrier."""
    #    print(data_ser[:, None])
    #    print(cutoff_vec[None, num_eliminated:])
    #    raise ValueError()
    # Calculate the number #{j: p_{j} > or < alpha_i } for each i
    if isinstance(data_ser, pd.Series):
        data_ser = data_ser.values
    if isinstance(cutoff_vec, pd.Series):
        cutoff_vec = cutoff_vec.values
    remaining_cutoffs = cutoff_vec[None, num_eliminated:]
    if highlow == "high":
        num_crossing_barrier = (data_ser[:, None] > remaining_cutoffs).sum(0)
    elif highlow == "low":
        num_crossing_barrier = (data_ser[:, None] < remaining_cutoffs).sum(0)
    else:
        raise ValueError("Unknown rejection direction: " + str(highlow))
    # Inner comparison tests p_{(i)} < alpha_i
    naive_barrier_tripped = (
        num_crossing_barrier >= np.arange(1, 1 + len(num_crossing_barrier))
    ).astype(bool)
    return naive_barrier_tripped


# %% step up and step down tests
def step_up_elimination(
    data_ser: pd.Series,
    cutoff_vec: NDArray[np.float32],
    num_eliminated: int,
    highlow: HighLow = "high",
) -> Tuple[List[Any], int]:
    """Runs a rejection or acceptance phase of a sequential stepdown procedure.

    Args:
        data_ser: series of test statistics for active hypotheses at given
            phase.
        cutoff_vec: vector of statistic cutoff values, with 0 index being the
            most significant. Passed whole, regardless of previous rejections.
            TODO: possibly change that
        num_elminated: integer number of
    """
    # Calculate number of active test statistics that have tripped each barrier
    barrier_crossed_mask = naive_barrier_trips(
        data_ser, cutoff_vec, num_eliminated, highlow
    )

    # Continue
    if np.any(barrier_crossed_mask):
        # Count of newly rejected (or accepted) hypotheses at this stage under stepup
        num_new_hyps = np.max(np.where(barrier_crossed_mask)[0]) + 1
        if highlow == "high":
            trip_cols = data_ser.index[
                data_ser > cutoff_vec[num_eliminated + num_new_hyps - 1]
            ]
        elif highlow == "low":
            trip_cols = data_ser.index[
                data_ser < cutoff_vec[num_eliminated + num_new_hyps - 1]
            ]
        return trip_cols, num_new_hyps
    else:
        # TODO:
        return [], 0


def step_down_elimination(data_ser, cutoff_vec, num_eliminated, highlow="high"):
    """Runs a rejection or acceptance phase of a sequential stepdown procedure.

    Args:
        data_ser: series of test statistics for active hypotheses at given
            phase.
        cutoff_vec: vector of statistic cutoff values, with 0 index being the
            most significant. Passed whole, regardless of previous rejections.
            TODO: possibly change that
        num_elminated: integer number of
    """
    assert len(cutoff_vec) - num_eliminated >= len(data_ser), "Too few cutoffs"
    # Calculate number of active test statistics that have tripped each barrier
    naive_barrier_tripped = naive_barrier_trips(
        data_ser, cutoff_vec, num_eliminated, highlow
    )

    # Cumprod sets to 1 everything at or more significant than the highest
    # significance level tripped by a stepdown procedure, and everything lower
    # to 0
    barrier_crossed_mask = np.cumprod(naive_barrier_tripped).astype(bool)
    if np.any(barrier_crossed_mask):
        # Inverts bits of array and searches for the index at which it switches
        # from 0 to 1, which is equivalent to the count of newly rejected (or
        # accepted) hypotheses at this stage under stepdown
        num_new_hyps = (~barrier_crossed_mask).searchsorted([True])[0]
        new_cutoff = cutoff_vec[num_eliminated + num_new_hyps - 1]
        if highlow == "high":
            trip_cols = data_ser.index[data_ser > new_cutoff]
        elif highlow == "low":
            trip_cols = data_ser.index[data_ser < new_cutoff]
        return trip_cols, num_new_hyps
    else:
        # TODO:
        return [], 0


#        # TODO: replace this with a return that sets blah blah blah
#        level_termination_step[num_eliminated:(num_eliminated + num_new_hyps)] = step
#        # Increment total number of total rejections/acceptances
#        num_eliminated = num_eliminated + num_new_hyps
#        termination_level.loc[accept_cols, 'accLevel'] = num_eliminated
#        termination_level.loc[accept_cols, 'step'] = step
#        nllr_accepted[step] = trip_cols.values
#        nllr.drop(accept_cols, axis=1, inplace=True)
#
#        # Reset data_ser
#        data_ser = nllr.ix[step]
#

# def step_illustration(data_ser, cutoff_vec, num_eliminated, highlow="high"):
#     n_remaining = len(cutoff_vec) - num_eliminated
#     cutoff_names = ["CUTOFF{0}".format(u) for u in np.arange(1, n_remaining + 1)]
#     stats_with_cutoffs = pd.concat((data_ser, pd.Series(cutoff_vec[num_eliminated:], index=cutoff_names)))
#     return stats_with_cutoffs.sort_values(ascending=(highlow=="low"))
