"""Main MultSeq code
"""

from pylab import *
from nose.tools import *
import itertools
import numpy
import pandas
import string, re
import pickle, os
import datetime, calendar
from scipy.stats import poisson, chi2_contingency, fisher_exact, binom_test
from scipy.stats import multivariate_normal, poisson, norm
from scipy.linalg import toeplitz
from statsmodels.base.model import GenericLikelihoodModel
import traceback, logging
from utils.common_funcs import log_odds, sigmoid
from utils.cutoff_funcs import fdr_helper, fdr_func, fdr_scaling_func, \
    calculate_mult_sprt_cutoffs, get_pvalue_cutoffs, pfdr_pfnr_cutoffs,\
    finite_horizon_rejective_cutoffs, create_fdr_controlled_alpha
from utils.data_funcs import read_drug_data, simulate_reactions, assemble_llr, simulate_correlated_reactions, df_generator
logger = logging.getLogger()
logger.setLevel(min((logging.WARNING, logger.level)))


  
def RejAccOtherFunc(hyp_term_data):
    if not isnan(hyp_term_data['rejLevel']):
        return "rej"
    elif not isnan(hyp_term_data['accLevel']):
        return "acc"
    else:
        return NaN
        

import pdb
            
            
# %% Fellouris synchronous SPRT
def fellouris_synch_sprt(llr, alpha=.1, beta=.35, m0=None, m0_range=(None, None)):
    """
    Args:
        llr (pandas.DataFrame): Columns are hypotheses to be tested, rows are time steps,
            and values are log likelihood ratio values
        alpha (float): desired FDR (or pFDR, see pfdr arg) value
        beta (float): desired FNR (or pFNR) value
        m0: (int) number of true null hypotheses. If None, then m0_range must
            be a valid 2-tuple.
        m0_range: (2-tuple of ints) lower and upper bounds (inclusive) on the
            number of true null hypotheses. If None, then m0 must be a valid 
            int.
    Return
    """
    m_hyps = len(llr.columns)
    
    if m0 is not None:
        c_val = min([log(m0 / alpha), log((m_hyps - m0) / beta)])
    elif isinstance(m0_range[0], int) and isinstance(m0_range[1], int):
        if 0 <= m0_range[0] and m0_range[0] <= m0_range[1] and m0_range[1] <= m_hyps:
            # TODO
            raise Exception("NOT YET IMPLEMENTED")
        else:
            raise ValueError("Bounds must be 0<=lower <= upper <= {0}. Received {1}".format(m_hyps, m0_range))
    else:
        raise ValueError("Must either pass m0 or m0_range")
        
    
    # Step through each timestep, performing rejections and acceptances
    for step in range(len(llr)):
        
        # Record diagnostic data
        if mod(step, record_interval)== (record_interval - 1):
            if verbose:
                # TODO: fix reporting
                logging.info("On step {0}. Accept: {1}. Reject: {2}, prop:{3}".format(step, j_accepted, j_rejected, 
                                                                           float(j_accepted+ j_rejected)/m_hyps))
            step_record.append(step)
            num_rejected_record.append(j_rejected)
            num_accepted_record.append(j_accepted)
            prop_terminated_record.append(float(j_accepted+ j_rejected) / m_hyps)

        # Alias data for timestep
        data_ser = llr.ix[step].sort_values()
        num_pos = (data_ser>0).sum()
        gap_stat = data_ser.ix[m0 + 1] - data_ser.ix[m0]
        
    # No remaining, hypothesisTerminationData, or levelTripData kv pairs
    fine_grained_outcomes = {'accepted':llr_accepted, 
                             'rejected':llr_rejected}
    # Instead of terminated, this now captures just the number of positive and
    # negative.
    pos_neg_time_series =  pandas.DataFrame(
        {'neg':num_accepted_record, 'pos':num_rejected_record}, 
            index=step_record)
    # TODO: fill in cutoff returns
    cutoff_output = (None,)
    return  (fine_grained_outcomes,
            termination_time_series, cutoff_output)
    
    

            
# %% Asynchronous SPRT

def modular_sprt_test(llr, A_vec, B_vec, record_interval=100, stepup=False, 
                      rejective=False, verbose=True):
    """
    Args:
        llr (pandas.DataFrame): Columns are hypotheses to be tested, rows are time steps,
            and values are log likelihood ratio values
        alpha (float): desired FDR (or pFDR, see pfdr arg) value
        beta (float): desired FNR (or pFNR) value
        rho (float): Wald approximation adjustment. Leave as default .583 for best results.
        BH (bool): Use Benjamini Hochberg style cutoffs vs Holm style cutoffs
        record_interval (int): frequency at which the terminations should be reported and recorded
        pfdr (bool): Specifies whether to use FDR+FNR or pFDR+pFNR. Defaults False to FDR. Requires m0 
            be set to a valid integer between 0 and m_hyps inclusive
        m0 (int): Only used if pfdr is True. 
        stepup (bool): Forces the test to use stepup rejection and acceptance, 
            although cutoffs were set to control FDR for stepdown procedures
            
    Returns:
        fine_grained_outcomes (dict):
            remaining: list of hypotheses never terminated
            accepted: dictionary mapping steps to lists of hypotheses accepted at that step
            rejected:dictionary mapping steps to lists of hypotheses rejected at that step
            hypTerminationData: DF with row for each hypothesis, specifying its outcome (ar0),
                the step at which it was accepted or rejected (NaN if neither),
                and the significance level at which it was terminated.
            levelTripData: DF with acc and rej columns and rows for each level. 
                Values are the step at which that level was tripped.
        termination_time_series (pandas.DataFrame): record of terminations by time step
            index: step, frequency depends on record_interval
            Accepted: number of nulls accepted by timestep 
            Rejected: number of nulls rejected by timestep
            'Prop Terminated': Accepted + Rejected / m_hyps
        cutoff_output (pandas.DataFrame): Cutoff values
            alpha:
            beta:
            A:
            B:
        
    """
    if isinstance(llr, pandas.DataFrame):
        # Copy data to prevent deletion
        llr = df_generator(llr.copy())

    if B_vec is None:
        if not rejective:
            raise ValueError("No B acceptance vector passed for acceptive-rejective test")
    if rejective:
        if B_vec is not None:
            raise ValueError("B acceptance vector passed for rejective test")

    # llr interface
    # if not a dataframe must have the following atts
    # llr.columns must list hypothesis names of ACTIVE hypotheses
    # llr.iterrows() must return the generator object that emits (step_number, step_data_series)
    # llr.drop(list_of_cols, **kwargs) must cause the ensuing yield statements 
    #       to omit list_of_cols, and ignore kwargs 
    
    # Calculate cutoffs
    m_hyps = len(llr.columns)
    hyp_names = llr.columns
    llr_iter = llr.iterrows()
    
    # Diagnostics
    step_record = []
    num_rejected_record = []
    num_accepted_record = []
    prop_terminated_record = []

    # other setup
    termination_level = pandas.DataFrame({'step':NaN, 'rejLevel':NaN, 'accLevel':NaN}, index=hyp_names)
    # For recording the time step at which each level was tripped
    acc_level_term = pandas.Series(np.inf, index=arange(m_hyps))
    rej_level_term = pandas.Series(np.inf, index=arange(m_hyps))
    llr_accepted = dict()
    llr_rejected = dict()
    j_accepted = 0
    j_rejected = 0
    
#
#    for thing in llr.iterrows():
#        print(thing)
    
    
    # Step through each timestep, performing rejections and acceptances
    # data_ser is alias for timestep data series
    for step, data_ser in llr_iter:
        
        # Record diagnostic data
        if mod(step, record_interval)== (record_interval - 1):
            if verbose:
                logging.info("On step {0}. Accept: {1}. Reject: {2}, prop:{3}".format(step, j_accepted, j_rejected, 
                                                                           float(j_accepted+ j_rejected)/m_hyps))
            step_record.append(step)
            num_rejected_record.append(j_rejected)
            num_accepted_record.append(j_accepted)
            prop_terminated_record.append(float(j_accepted+ j_rejected) / m_hyps)

        


        if not rejective:
            if stepup:
                accept_cols, num_new_accepts = step_up_elimination(data_ser, B_vec, j_accepted, highlow="low")
            else:
                accept_cols, num_new_accepts = step_down_elimination(data_ser, B_vec, j_accepted, highlow="low")
            if num_new_accepts>0:
                if verbose:
                    logging.info( "num new accepts >0" + str( step))
                acc_level_term[j_accepted:(j_accepted+num_new_accepts)] = step
                j_accepted = j_accepted + num_new_accepts
                termination_level.loc[accept_cols, 'accLevel'] = j_accepted
                termination_level.loc[accept_cols, 'step'] = step
                llr_accepted[step] = list(accept_cols.values)
                llr.drop(accept_cols, axis=1, inplace=True)
        
                # Reset data_ser
                data_ser.drop(accept_cols, inplace=True)
        
        if stepup:
            reject_cols, num_new_rejects = step_up_elimination(data_ser, A_vec, j_rejected, highlow="high")
        else:
            reject_cols, num_new_rejects = step_down_elimination(data_ser, A_vec, j_rejected, highlow="high")
            
        if num_new_rejects>0:
            if verbose:
                logging.info("num new rejects >0" + str( step))
            rej_level_term[j_rejected:(j_rejected+num_new_rejects)] = step
            j_rejected = j_rejected + num_new_rejects
            termination_level.loc[reject_cols, 'rejLevel'] = j_rejected
            termination_level.loc[reject_cols, 'step'] = step
            llr_rejected[step] = list(reject_cols.values)
            llr.drop(reject_cols, axis=1, inplace=True)
            
            data_ser.drop(reject_cols, inplace=True)
            
        if len(llr.columns)==0:
            logging.debug("Stopping early on step " + str(step))
#            print("end\n", data_ser, "\n", j_rejected, j_accepted)
            break
            
    rej_level_term.replace(np.inf, step+1, inplace=True)
    acc_level_term.replace(np.inf, step+1, inplace=True)
    # Record final diagnostic data
    if verbose:
        logging.info( "Final step {0}. Accept: {1}. Reject: {2}, prop:{3}".format(step + 1, 
            j_accepted, j_rejected, float(j_accepted+ j_rejected)/m_hyps))
    step_record.append(step)
    num_rejected_record.append(j_rejected)
    num_accepted_record.append(j_accepted)
    prop_terminated_record.append(float(j_accepted+ j_rejected) / m_hyps)

            
    termination_level['ar0'] = termination_level.apply(RejAccOtherFunc, axis=1)
    if rejective:
        termination_level.ix[pandas.isnull(termination_level['ar0']), 'ar0'] = "acc"

    
            
    fine_grained_outcomes = {'remaining':list(llr.columns), 
                             'accepted':llr_accepted, 
                             'rejected':llr_rejected, 
              'drugTerminationData':termination_level, 
              'hypTerminationData':termination_level, 
              'levelTripData':pandas.DataFrame({'acc':acc_level_term, 'rej':rej_level_term})}
    termination_time_series =  pandas.DataFrame(
        {'Accepted':num_accepted_record, 'Rejected':num_rejected_record, 
         'Prop Terminated':prop_terminated_record}, index=step_record)
    return  (fine_grained_outcomes, termination_time_series)



def naive_barrier_trips(data_ser, cutoff_vec, num_eliminated, highlow):
    """Returns a boolean vector indicating whether or not the jth most 
    significant active statistic tripped the jth active barrier. """
#    print(data_ser[:, None])
#    print(cutoff_vec[None, num_eliminated:])
#    raise ValueError()
    if highlow=="high":        
        num_crossing_barrier = (data_ser[:, None] > cutoff_vec[None, num_eliminated:]).sum(0)
    elif highlow=="low":
        num_crossing_barrier = (data_ser[:, None] < cutoff_vec[None, num_eliminated:]).sum(0)
    else:
        raise ValueError("Unknown rejection direction: "+str(highlow))    
    # Inner comparison tests p_{(i)} < alpha_i
    naive_barrier_tripped = (num_crossing_barrier >= arange(1, 1 + len(num_crossing_barrier))).astype(bool)
    return naive_barrier_tripped
    
# %% step up and step down tests
def step_up_elimination(data_ser, cutoff_vec, num_eliminated, highlow="high"):
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
    barrier_crossed_mask = naive_barrier_trips(data_ser, cutoff_vec, 
                                                num_eliminated, highlow)
     
    # Continue
    if any(barrier_crossed_mask):
        # Count of newly rejected (or accepted) hypotheses at this stage under stepup
        num_new_hyps = max(where(barrier_crossed_mask)[0])  + 1
        if highlow=="high":
            trip_cols = data_ser.index[data_ser > cutoff_vec[num_eliminated + num_new_hyps - 1]]
        elif highlow=="low":
            trip_cols = data_ser.index[data_ser < cutoff_vec[num_eliminated + num_new_hyps - 1]]
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
    naive_barrier_tripped = naive_barrier_trips(data_ser, cutoff_vec, 
                                                num_eliminated, highlow)
    
    # Cumprod sets to 1 everything at or more significant than the highest 
    # significance level tripped by a stepdown procedure, and everything lower
    # to 0
    barrier_crossed_mask = cumprod(naive_barrier_tripped).astype(bool)    
    # Continue
    if any(barrier_crossed_mask):
        # Inverts bits of array and searches for the index at which it switches
        # from 0 to 1, which is equivalent to the count of newly rejected (or 
        # accepted) hypotheses at this stage under stepdown
        num_new_hyps = (~barrier_crossed_mask).searchsorted([True])[0]
        if highlow=="high":
            trip_cols = data_ser.index[data_ser > cutoff_vec[num_eliminated + num_new_hyps - 1]]
        elif highlow=="low":
            trip_cols = data_ser.index[data_ser < cutoff_vec[num_eliminated + num_new_hyps - 1]]
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

def step_illustration(data_ser, cutoff_vec, num_eliminated, highlow="high"):
    n_remaining = len(cutoff_vec) - num_eliminated
    cutoff_names = ["CUTOFF{0}".format(u) for u in arange(1, n_remaining + 1)]
    stats_with_cutoffs = pandas.concat((data_ser, pandas.Series(cutoff_vec[num_eliminated:], index=cutoff_names)))
    return stats_with_cutoffs.sort_values(ascending=(highlow=="low"))
    

    
# analysis block
        
from IPython.display import display
def display_results(tout):
    for el in tout:
        if isinstance(el, dict):
            for k,v in el.items():
                print(k)
                display(v)
        else:
            display(el)
        print("\n")