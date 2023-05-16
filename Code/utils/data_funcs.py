# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 23:21:28 2016

@author: mike
"""
import pandas
import datetime, os, string, warnings
import numpy
import numpy.random
from numpy import array, linspace, repeat, tile, newaxis, exp, ones, hstack, vstack
from numpy import zeros, log, sqrt, arange, percentile, logical_and, isnan
from scipy.linalg import toeplitz
from scipy.stats import norm, multivariate_normal, poisson, binom
import hashlib
import logging

def gen_names(n_hyps):
    STR_LEN = 3
    digits = int(numpy.ceil(numpy.log10(n_hyps)))
    for i in range(n_hyps):
        return [("{0:0=" + str(digits) + "d}-{1}").format(i, hashlib.sha1(str(i).encode("utf-8")).hexdigest()[:STR_LEN]) for i in range(n_hyps)]
        


TOTAL_AMNESIA_CNAME = "Total All"
TOTAL_REACTS_CNAME = "total_reactions"
DATA_FILE_PATH = os.path.expanduser('~/Dropbox/Research/MultSeq/Data/YellowcardData.csv')

def gen_skew_prescreen(min_am=1, min_tot=20, aug_am=1, aug_non=1):
    """Generates a drug prescreening function to be passed to read_drug_data."""
    def skew_prescreen(reacts_df, am_col, tot_col):
        screened_df = reacts_df[(reacts_df[am_col]>min_am) | (reacts_df[tot_col]>min_tot)].copy()
        screened_df[TOTAL_AMNESIA_CNAME] = screened_df[TOTAL_AMNESIA_CNAME] + aug_am
        screened_df[TOTAL_REACTS_CNAME] = screened_df[TOTAL_REACTS_CNAME] + aug_am + aug_non
        return screened_df
    return skew_prescreen

def read_drug_data(prescreen=None, aug=1):
    """Read drug reaction rates and some metadata from file.
    """
    # sum total days over all drugs, divide total reports by that 
    # emit reports from each drug and total drugs with given rates
    # can't be formulated as sprt.... look at bartroff's???
    

    reacts_df = pandas.read_csv(DATA_FILE_PATH, index_col=0, 
                                parse_dates=["end_date","start_date", "first_react"])

    # Here either exclude drugs without first report date or fill in report start date
    first_react_unknown_mask = reacts_df.isnull()['first_react']
    reacts_df.loc[first_react_unknown_mask, 'first_react'] = reacts_df[first_react_unknown_mask]['start_date']
    
    # Optional prescreening
    if prescreen:
        reacts_df = prescreen(reacts_df, TOTAL_AMNESIA_CNAME, TOTAL_REACTS_CNAME)

    amnesia_reacts = reacts_df[TOTAL_AMNESIA_CNAME]
    secs_per_year = datetime.timedelta(365).total_seconds()
    date_range_col = (reacts_df["end_date"]-reacts_df["first_react"]).apply(
        lambda drange: drange/numpy.timedelta64(1,'s'))/secs_per_year
    N_reports = reacts_df[TOTAL_REACTS_CNAME].sum()
    N_amnesia = reacts_df[TOTAL_AMNESIA_CNAME].sum()
    # Calculate rates
    drug_amnesia_rate = (aug + reacts_df[TOTAL_AMNESIA_CNAME])/date_range_col
    drug_reacts_rate = (2 * aug + reacts_df[TOTAL_REACTS_CNAME])/date_range_col
    drug_nonamnesia_rate = drug_reacts_rate - drug_amnesia_rate
    return drug_amnesia_rate, drug_nonamnesia_rate, (N_amnesia, N_reports)
    

def assemble_fake_drugs(max_magnitude, m_null, interleaved, p0, p1):
    """Assembles dar and dnar for fake drugs.
    
    """
    mag_vec = linspace(1, max_magnitude, m_null)
    
    
    drug_names = gen_names(2 * m_null)
    
    # Create null/alternative masks, and total magnitude series
    if interleaved:
        ground_truth = pandas.Series(tile(array([True, False]), m_null), 
                                     index=drug_names)
        drr = pandas.Series(repeat(mag_vec, 2), index=drug_names)
    else:        
        ground_truth = pandas.Series(repeat(array([True, False]), m_null), 
                                     index=drug_names)
        drr = pandas.Series(tile(mag_vec, 2), index=drug_names)
        
    # Create (non) amensia magnitude
    dar = (p0 * ground_truth + p1 * ~ground_truth) * drr
    dnar = drr - dar
    return dar, dnar, ground_truth

    
def assemble_fake_gaussian(max_magnitude, m_null, p0, p1, m_alt=None):
    """Assembles dar and dnar for fake gaussian.
    
    """
    var_vec_true = linspace(1, max_magnitude, m_null)
    var_vec_false = linspace(1, max_magnitude, m_alt)
    #drr = repeat(mag_vec, 2)
    #dar = concatenate((p0 * mag_vec, p1 * mag_vec)) 
    
    
    drug_names = gen_names(m_null + m_alt)
#    list(map(lambda u,v: u + v, 
#                     array(list(string.ascii_letters))[arange(0, 4 * m_null, 2)], 
#                     array(list(string.ascii_letters))[arange(1, 4 * m_null, 2)]))
    
    # Create null/alternative masks, and total magnitude series    
    if m_alt is None:
        ground_truth = pandas.Series(repeat(array([True, False]), m_null), 
                                 index=drug_names)
    else:
        ground_truth = pandas.Series(repeat(array([True, False]), [m_null, m_alt]), 
                                 index=drug_names)
    mean_vec = (p0 * ground_truth + p1 * ~ground_truth) 
    sd_vec = pandas.Series(sqrt(numpy.concatenate((var_vec_true, var_vec_false))), index=mean_vec.index)
    
    return mean_vec, sd_vec, ground_truth
    
def assemble_fake_binom(m_null, interleaved, p0, p1, m_alt=None):
    """Assembles dar and dnar for fake drugs.
    
    """
    if m_alt is None:
        drug_names = gen_names(2 * m_null)
    else:
        drug_names = gen_names(m_null + m_alt)
#    list(map(lambda u,v: u + v, 
#                     array(list(string.ascii_letters))[arange(0, 4 * m_null, 2)], 
#                     array(list(string.ascii_letters))[arange(1, 4 * m_null, 2)]))
#    
    # Create null/alternative masks, and total magnitude series
    if interleaved:
        if m_alt is None:
            ground_truth = pandas.Series(tile(array([True, False]), m_null), 
                                     index=drug_names)
        else:
            ground_truth = pandas.Series(tile(array([True, False]), [m_null, m_alt]), 
                                     index=drug_names)
        
    else:        
        if m_alt is None:
            ground_truth = pandas.Series(repeat(array([True, False]), m_null), 
                                     index=drug_names)
        else:
            ground_truth = pandas.Series(repeat(array([True, False]), [m_null, m_alt]), 
                                     index=drug_names)
        
    # Create (non) amensia magnitude
    dar = (p0 * ground_truth + p1 * ~ground_truth)
    return dar, ground_truth
    

def assemble_fake_pois(m_null, interleaved, p0, p1, m_alt=None):
    """Assembles dar and dnar for fake drugs.
    
    """
    if m_alt is None:
        drug_names = gen_names(2 * m_null)
    else:
        drug_names = gen_names(m_null + m_alt)
#    drug_names = list(map(lambda u,v: u + v, 
#                     array(list(string.ascii_letters))[arange(0, 4 * m_null, 2)], 
#                     array(list(string.ascii_letters))[arange(1, 4 * m_null, 2)]))
    
    # Create null/alternative masks, and total magnitude series
    if interleaved:
        if m_alt is None:
            ground_truth = pandas.Series(tile(array([True, False]), m_null), 
                                     index=drug_names)
        else:
            ground_truth = pandas.Series(tile(array([True, False]), [m_null, m_alt]), 
                                     index=drug_names)
        
    else:        
        if m_alt is None:
            ground_truth = pandas.Series(repeat(array([True, False]), m_null), 
                                     index=drug_names)
        else:
            ground_truth = pandas.Series(repeat(array([True, False]), [m_null, m_alt]), 
                                     index=drug_names)
                                         
        
    # Create (non) amensia magnitude
    dar = (p0 * ground_truth + p1 * ~ground_truth)
    return dar, ground_truth


def assemble_fake_pois_grad(m_null, p0, p1, m_alt=None):
    """Assembles dar and dnar for fake drugs.
    
    """
    if m_alt is None:
        m_alt = m_null
    lam0 = min((p0, p1))
    lam1 = max((p0, p1))
    lam_ratio = lam0 / lam1
    log_low_lam = log(lam0) + log(lam_ratio)
    log_high_lam = log(lam1) - log(lam_ratio)
    dar_vals = exp(linspace(log_low_lam, log_high_lam, m_null + m_alt))
    drug_names = gen_names(m_null + m_alt)
#    drug_names = list(map(lambda u,v: u + v, 
#                     array(list(string.ascii_letters))[arange(0, 4 * m_null, 2)], 
#                     array(list(string.ascii_letters))[arange(1, 4 * m_null, 2)]))
    
    return pandas.Series(dar_vals, index=drug_names)

    
def simulate_reactions(drug_amnesia_rate, drug_nonamnesia_rate, n_periods):
    """Given series of drug amnesia and non-amnesia rates, gerenates data.
    
    args:
        drug_amnesia_rate: Series
        drug_nonamnesia_rate: Series
        n_periods: int
    return:
        tuple of DataFrames of reactions. Columns are drugs, rows are periods.
    """
    sim_amnesia_reactions = pandas.DataFrame(dict([
                (drug_name, poisson.rvs(individual_drug_rate, size=n_periods)) if individual_drug_rate>0
                else (drug_name, zeros(n_periods))
                for drug_name, individual_drug_rate in drug_amnesia_rate.iteritems()])).cumsum()
    
    sim_nonamnesia_reactions = pandas.DataFrame(dict([
                (drug_name, poisson.rvs(individual_drug_rate, size=n_periods)) if individual_drug_rate>0
                else (drug_name, zeros(n_periods))
                for drug_name, individual_drug_rate in drug_nonamnesia_rate.iteritems()])).cumsum()
    return (sim_amnesia_reactions.reindex(columns=drug_amnesia_rate.index), 
            sim_nonamnesia_reactions.reindex(columns=drug_nonamnesia_rate.index))

    
def simulate_gaussian_noncum_internal(gauss_moments_df, n_periods):
    return pandas.DataFrame(dict([(hyp_name, hyp_data["mean"] + hyp_data["sd"]*numpy.random.randn(n_periods))
        for hyp_name, hyp_data in gauss_moments_df.iterrows()])).reindex(columns=gauss_moments_df.index)
    
def simulate_gaussian_noncum(dar, dnar, n_periods):
    return simulate_gaussian_noncum_internal(pandas.DataFrame({"mean":dar, "sd":dnar}), n_periods)

def simulate_binom(bin_props, n_periods):
    return pandas.DataFrame(dict([(hyp_name, binom.rvs(1, hyp_prop, size=n_periods))
        for hyp_name, hyp_prop in bin_props.iteritems()])).cumsum().reindex(columns=bin_props.index)

def simulate_pois(pois_rates, n_periods):
    return pandas.DataFrame(dict([(hyp_name, poisson.rvs(hyp_rate, size=n_periods))
        for hyp_name, hyp_rate in pois_rates.iteritems()])).cumsum().reindex(columns=pois_rates.index)    

    
        
def assemble_llr(amnesia, nonamnesia, p0, p1):
    """deprecated. Use assemble_drug_llr.
    log(L(p1)/L(p0))
    E[llr|H0] > 0
    E[llr|H1] < 0
    """
    warnings.warn("assemble_llr is deprecated. Use assemble_drug_llr.")
    return amnesia * log(p1/p0) + nonamnesia * log((1 - p1) / (1 - p0))


def assemble_drug_llr(counts, p0, p1):
    """
    counts: 2-tuple, amnesia count vec and non amnesia count vec
    p0: 
    p1:
        
        
    log(L(p1)/L(p0))
    E[llr|H0] > 0
    E[llr|H1] < 0
    """
    return counts[0] * log(p1/p0) + counts[1] * log((1 - p1) / (1 - p0))


def assemble_binom_llr(bin_count, p0, p1):
    """log(L(p1)/L(p0))
    E[llr|H0] > 0
    E[llr|H1] < 0
    """
    pos_count = bin_count
    neg_count = (arange(1, len(bin_count) + 1))[:, newaxis] - bin_count
    return pos_count * log(p1/p0) + neg_count * log((1 - p1) / (1 - p0))
    
    
def assemble_gaussian_llr(samps_noncum, p0, p1):
    """log(L(p1)/L(p0))
    E[llr|H0] > 0
    E[llr|H1] < 0
    """
    samps_idx = arange(1.0, len(samps_noncum) + 1)
    est_var0 = (1.0 / samps_idx[:, numpy.newaxis]) * ((samps_noncum - p0)**2.0).cumsum()
    est_var1 = (1.0 / samps_idx[:, numpy.newaxis]) * ((samps_noncum - p1)**2.0).cumsum()
    return -.5 * samps_idx[:, numpy.newaxis] * (log(est_var1) - log(est_var0))


def assemble_pois_llr(pois_count, lam0, lam1):
    """log(L(p1)/L(p0))
    E[llr|H0] > 0
    E[llr|H1] < 0
    """
    period_idx = pandas.DataFrame(ones(pois_count.shape), columns=pois_count.columns,
                                  index=pois_count.index).cumsum()

    return (log(lam1 / lam0) * pois_count) - (period_idx * (lam1 - lam0))

def generate_llr(dar, dnar, n_periods, rho, hyp_type, p0, p1, 
                 m1=None, rho1=None, rand_order=False, cummax=False):
    if (hyp_type is None) or (hyp_type=="drug"):
        amnesia, nonamnesia = simulate_correlated_reactions(dar, dnar, n_periods, rho, 
                                                            m1, rho1, rand_order=rand_order)
        llr = assemble_drug_llr((amnesia, nonamnesia), p0, p1)
    elif hyp_type == "binom":
        event_count = simulate_correlated_binom(dar, n_periods, rho, 
                                                           m1, rho1, rand_order=rand_order)
        llr = assemble_binom_llr(event_count, p0, p1)
    elif hyp_type == "pois":
        event_count = simulate_correlated_pois(dar, n_periods, rho, 
                                                          m1, rho1, rand_order=rand_order)
        llr = assemble_pois_llr(event_count, p0, p1)
    elif hyp_type == "gaussian":
        gvals = simulate_correlated_gaussian_noncum(dar, dnar, n_periods, 
                                                               rho, m1, rho1, rand_order=rand_order)
        llr = assemble_gaussian_llr(gvals, p0, p1)
    else:
        raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
    if cummax:
        return llr.cummax()
    else:
        return llr


def imp_sample_drug_weight(counts, p0, p1, drr0=None, drr1=None):
    amcount = counts[0].iloc[-1, :] # Final step is all thats important
    nonamcount = counts[1].iloc[-1, :]
    am_factor = p0 / p1
    nonam_factor = (1 - p0) / (1 - p1)
    log_weight = amcount * log(am_factor) + nonamcount * log(nonam_factor)
    raw_weight = exp(log_weight)
    if isnan(raw_weight).any():
        logger = logging.getLogger()
        logger.debug("NaN weights: {0}".format(raw_weight[isnan(raw_weight)]))
        logger.debug("Log weights: {0}".format(log_weight[isnan(raw_weight)]))
#    print(raw_weight)
#    print(raw_weight[isnan(raw_weight)])
#    print(drr0[isnan(raw_weight)])
#    print(counts[0].iloc[-1, :][isnan(raw_weight)])
#    print(counts[1].iloc[-1, :][isnan(raw_weight)])
#    print(counts[0].iloc[-1, :].head())
    if drr0 is not None and drr1 is not None:
        T = counts[0].shape[0]
        weight = raw_weight * ((drr0 / drr1).astype('float128') ** (amcount + nonamcount)) * exp(-T * (drr0 - drr1)).astype('float128')
    else:
        weight = raw_weight
    return weight

def imp_sample_binom_weight(counts, p0, p1):
    events = counts.iloc[-1, :] # Final step is all thats important
    fail_events = counts.shape[0] - events
    event_factor = p0 / p1
    fail_factor = (1 - p0) / (1 - p1)
    log_weight = events * log(event_factor) + fail_events * log(fail_factor)
    return exp(log_weight)

def imp_sample_pois_weight(counts, p0, p1):
    events = counts.iloc[-1, :] # Final step is all thats important
    n = counts.shape[0]
    factor = p0 / p1
    # irrelevant factor 
    # stupid = exp(-counts.shape[0]*(po - p1))
    log_weight = events * log(factor) - n * (p0 - p1)
    return exp(log_weight)


def imp_sample_gaussian_weight(counts, p0, p1, drr):
    raise Exception("Not implemented yet. Composite or simple? SD or VAR?")
    
    

def whole_data_p0(N_amnesia, N_reports):
    """
    Estimate of average proportion of side effects that are amnesia for all 
    drugs combined.
    """
    return float(N_amnesia)/N_reports


def whole_data_p1(N_amnesia, N_reports, p0, n_se=2.0):
    """Get p1 above p0 based on total reports SE for p0 estimate"""
    return p0 + n_se * sqrt(p0 * (1-p0) / N_reports)
    
def drug_data_p1(N_drugs, p0, n_se=2.0):
    """Get p1 above p0 based on total drugs SE for p0 estimate"""
    return p0 + n_se * sqrt(p0 * (1-p0) / N_drugs)
    
def am_prop_percentile_p0_p1(dar, dnar, p0_pctl, p1_pctl):
    """Get p0 and p1 as percentiles of amnesia proportion"""
    am_prop = dar / (dar + dnar)
    return numpy.percentile(am_prop, [100*p0_pctl, 100*p1_pctl])
    
def prescreen_abs(min_am_reacts, min_total_reacts):
    """Creates a prescreening function based on absolute reaction counts.
    For use in read_data
    
    """
    def prescreen(reacts_df, TOTAL_AMNESIA_CNAME, TOTAL_REACTS_CNAME):
        mask = logical_and(reacts_df[TOTAL_AMNESIA_CNAME] >= min_am_reacts,
                           reacts_df[TOTAL_REACTS_CNAME] >= min_total_reacts)
        return reacts_df[mask]
    return prescreen
        

def prescreen_rel(min_am_reacts_percentile, min_total_reacts_percentile):
    """Creates a prescreening function based on percentile reaction counts.
    For use in read_data
    
    """
    def prescreen(reacts_df, TOTAL_AMNESIA_CNAME, TOTAL_REACTS_CNAME):
        min_am_reacts = percentile(reacts_df[TOTAL_AMNESIA_CNAME], 
                                   min_am_reacts_percentile)
        min_total_reacts = percentile(reacts_df[TOTAL_REACTS_CNAME], 
                                      min_total_reacts_percentile)
        mask = logical_and(reacts_df[TOTAL_AMNESIA_CNAME] >= min_am_reacts,
                           reacts_df[TOTAL_REACTS_CNAME] >= min_total_reacts)
        return reacts_df[mask]
    return prescreen    
    
    
def toep_corr_matrix(m, rho, m1=None, rho1=None, rand_order=False):
    
    
    if m1 is None:
        raw_corr_mat = toeplitz(rho ** abs(arange(m)))
        if rand_order:
            ordering = numpy.random.permutation(arange(m))
            print(ordering)
            corr_mat = (raw_corr_mat[ordering, :])[:, ordering]
            return corr_mat
        else:
            return raw_corr_mat
    else:
        if rand_order:
            warnings.warn("Both seperate correlation structures and random "
                          "ordering were passed.\n Ignoring random ordering.")
        m0 = m - m1
        rho0 = rho
        if rho1 is None:
            rho1 = rho
        UL = toep_corr_matrix(m0, rho0)
        BR = toep_corr_matrix(m1, rho1)
        UR = zeros((m0, m1))
        BL = zeros((m1, m0))
        outmat = vstack([hstack([UL, UR]), hstack([BL, BR])])
#        print(outmat.round(1))
        return outmat
        
        
    

def simulate_correlated_reactions(drug_amnesia_rate, drug_nonamnesia_rate, n_periods, rho,
                                  m1=None, rho1=None, rand_order=False):
    """
    """
    if rho is None:
        return simulate_reactions(drug_amnesia_rate, drug_nonamnesia_rate, n_periods)
    else:
            
        # TODO: fix 
        cov_mat = toep_corr_matrix(len(drug_amnesia_rate), rho, m1, rho1, rand_order=rand_order) 
        #toeplitz(rho** abs(arange(len(drug_amnesia_rate))))
        #cov_mat[arange(0, N_reports, 5)[:,newaxis], arange(0, N_reports, 5)] = cov_mat[arange(0, N_reports, 5)[:,newaxis], arange(0, N_reports, 5)]**0.5
        uuA = pandas.DataFrame(norm.cdf(multivariate_normal(cov=cov_mat).rvs(size=n_periods)), 
                              columns = drug_amnesia_rate.index)
        uuB = pandas.DataFrame(norm.cdf(multivariate_normal(cov=cov_mat).rvs(size=n_periods)), 
                              columns = drug_amnesia_rate.index)
    
        # Running total of reaction reports
        # Iterates through drugs, generating random samples of size n_periods for 
        # each drug with nonzero rate.
        sim_amnesia_reactions = pandas.DataFrame(dict([
                    (drug_name, poisson.ppf(uuA[drug_name], individual_drug_rate)) if individual_drug_rate>0
                    else (drug_name, zeros(n_periods))
                    for drug_name, individual_drug_rate in drug_amnesia_rate.items()])).cumsum()
        
        sim_nonamnesia_reactions = pandas.DataFrame(dict([
                    (drug_name, poisson.ppf(uuB[drug_name], individual_drug_rate)) if individual_drug_rate>0
                    else (drug_name, zeros(n_periods))
                    for drug_name, individual_drug_rate in drug_nonamnesia_rate.items()])).cumsum()
        return (sim_amnesia_reactions.reindex(columns=drug_amnesia_rate.index), 
                sim_nonamnesia_reactions.reindex(columns=drug_nonamnesia_rate.index))
    

    
    
def simulate_correlated_gaussian_noncum_internal(gauss_moments_df, n_periods, rho, 
                                                 m1=None, rho1=None, rand_order=False):
    sdvec = gauss_moments_df["sd"].values    
    var_mat = sdvec[:, numpy.newaxis] * sdvec[numpy.newaxis, :]
    cov_mat = toep_corr_matrix(len(gauss_moments_df), rho, m1, rho1, rand_order=rand_order) * var_mat
    #toeplitz(rho** abs(arange(len(gauss_moments_df)))) 
    mean_vec = gauss_moments_df["mean"].values
    sim_values = multivariate_normal(mean=mean_vec, cov=cov_mat).rvs(size=n_periods)
    return pandas.DataFrame(sim_values, columns = gauss_moments_df.index)
    
def simulate_correlated_gaussian_noncum(dar, dnar, n_periods, rho, 
                                        m1=None, rho1=None, rand_order=False):
    return simulate_correlated_gaussian_noncum_internal(pandas.DataFrame({"mean":dar, "sd":dnar}), n_periods, rho, 
                                                        m1=m1, rho1=rho1, rand_order=rand_order)

def simulate_correlated_binom(bin_props, n_periods, rho, 
                              m1=None, rho1=None, rand_order=False):    
    cov_mat = toep_corr_matrix(len(bin_props), rho, m1, rho1, rand_order=rand_order)
    #toeplitz(rho** abs(arange(len(bin_props))))
    #cov_mat[arange(0, N_reports, 5)[:,newaxis], arange(0, N_reports, 5)] = cov_mat[arange(0, N_reports, 5)[:,newaxis], arange(0, N_reports, 5)]**0.5
    uuA = pandas.DataFrame(norm.cdf(multivariate_normal(cov=cov_mat).rvs(size=n_periods)), 
                          columns = bin_props.index)
    return pandas.DataFrame(dict([(hyp_name, binom.ppf(uuA[hyp_name], 1, hyp_prop))
        for hyp_name, hyp_prop in bin_props.iteritems()])).cumsum().reindex(columns=bin_props.index)

def simulate_correlated_pois(pois_rates, n_periods, rho, 
                             m1=None, rho1=None, rand_order=False):
    
    cov_mat = toep_corr_matrix(len(pois_rates), rho, m1, rho1, 
                               rand_order=rand_order)
    # toeplitz(rho** abs(arange(len(pois_rates))))
    #cov_mat[arange(0, N_reports, 5)[:,newaxis], arange(0, N_reports, 5)] = cov_mat[arange(0, N_reports, 5)[:,newaxis], arange(0, N_reports, 5)]**0.5
    uuA = pandas.DataFrame(norm.cdf(multivariate_normal(cov=cov_mat).rvs(size=n_periods)), 
                          columns = pois_rates.index)
    return pandas.DataFrame(dict([(hyp_name, poisson.ppf(uuA[hyp_name], hyp_rate))
        for hyp_name, hyp_rate in pois_rates.iteritems()])).cumsum().reindex(columns=pois_rates.index)

class drug_data_metadata(object):
    def __init__(self, N_reports, N_amnesia, p0, p1, n_periods):
        self.N_reports = N_reports 
        self.N_amnesia = N_amnesia
        self.p0 = p0 
        self.p1 = p1
        self.n_periods = n_periods
        
    def __str__(self):
        outstr = ("N_reports: {0}\n"
                  "N_amnesia: {1}\n" 
                  "p0: {2}\n"
                  "p1: {3}\n"
                  "n_periods: {4}").format(self.N_reports, 
                                           self.N_amnesia, 
                                           self.p0, 
                                           self.p1, 
                                           self.n_periods)
        return outstr
    
# %% streaming code
# TODO(mhankin): make .df a reference to the updated dataframe... somehow
class online_data(object):
    """streaming data interface class"""
    
    def __init__(self, col_list, dgp):
        self._dgp = dgp
        if isinstance(col_list, pandas.Index):
            self._columns = col_list
        else:
            self._columns = pandas.Index(col_list)
            
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
    def __init__(self, parent, dgp):
        self._parent = parent
        self._dgp = dgp
        self._current_index = -1
        
    def __iter__(self):
        return self
        
    def __next__(self):
        return self.next()

    def next(self):
        self._current_index = self._current_index + 1
        idx = self._current_index
        data_ser = self._dgp(self._parent.columns)
        return (idx, data_ser)
    

        
    
class df_dgp_wrapper(object):
    """wraps a pandas DataFrame to make it fit the dgp interface"""
    def __init__(self, df):
        self._df = df
        self._iter_rows = df.iterrows()
        
    def __call__(self, col_list):
        _, data_ser = next(self._iter_rows)
        if data_ser is None:
            StopIteration()
        return data_ser[col_list]
    
    def get_data_record(self):
        return self._df
    
    def __len__(self):
        return len(self._df)
    
def df_generator(df):
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
                self._df = pandas.concat((self._df, new_df))
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
        
    