"""
Created on Wed Feb  8 15:03:03 2017

@author: mhankin
"""
from scipy.special import factorial

import multseq

print(multseq.__file__)
from multseq import step_down_elimination, step_up_elimination
from . import cutoff_funcs
from . import data_funcs
from scipy import stats
import pandas as pd
import numpy as np
import tqdm
import os
import seaborn as sns
import visualizations
import configparser
import statsmodels.formula.api as sm
from contextlib import closing
import shelve
from IPython.display import display

#alpha = .25
#rho = -.6
#m_null = 5
#n_periods = 25
PERIOD_TEST_POINTS = 10
low_sample = 5
high_sample = 125
period_max_exp = .0
reps_per_period_point = 500
#scale_fdr=True
#max_magnitude = 4.0

def single_binom_test(x, n:int, p0:float, p1:float, halfp:bool=False):
    if p1>p0:
        if halfp:
            return (1 - .5 * stats.binomcdf(x-1, n, p0) - .5 * stats.binomcdf(x, n, p0))
        else:
            return (1 - stats.binomcdf(x-1, n, p0))
    else:
        # p1 < p0
        if halfp:
            return (.5 * stats.binomcdf(x, n, p0) + .5 * stats.binomcdf(x-1, n, p0))
        else:
            return stats.binomcdf(x, n, p0)
    
def single_pois_test(x, lam0, lam1, halfp=False):
    if lam1>lam0:
        if halfp:
            return (1 - .5 * stats.poisson.cdf(x-1, lam0) - .5 * stats.poisson.cdf(x, lam0))
        else:
            return (1 - stats.poisson.cdf(x-1, lam0))
    else:
        # p1 < p0
        if halfp:
            return (.5 * stats.poisson.cdf(x, lam0) + .5 * stats.poisson.cdf(x-1, lam0))
        else:
            return stats.poisson.cdf(x, lam0)
        
def single_test(x, param0, param1, prod_dist, param_comparison, halfp=False):
    if param_comparison(param0, param1):
        if halfp:
            return (1 - .5 * prod_dist.cdf(x-1, *param0) - .5 * prod_dist.cdf(x, *param0))
        else:
            return (1 - prod_dist.cdf(x-1, *param0))
    else:
        # p1 < p0
        if halfp:
            return (.5 * prod_dist.cdf(x, *param0) + .5 * prod_dist.cdf(x-1, *param0))
        else:
            return prod_dist.cdf(x, *param0)
        
def mixed_test_points(low, high, num, mix=.7):
    """Mix of linear and log scale grid points between values.
    mix value controls how many log vs linear.
    """
    
    
    log_num = int(ceil(mix * num))
    lin_num = int(ceil((1.0 - mix) * num)) + 2
    lin_list = np.linspace(low, high, lin_num)[1:-1].tolist()
    log_list = logspace(log10(low), log10(high), log_num).tolist()
    full_ar = np.array(lin_list + log_list)
    full_ar.sort()
    return full_ar.astype(int)
    
    

def get_oc_range_wrapper(p0, p1, 
                         alpha, # = alpha, 
                         rho,# = rho, 
                         m_null, m_alt,
                         period_test_points = PERIOD_TEST_POINTS, 
                         low_sample = low_sample, 
                         high_sample = high_sample,
                         period_max_exp = period_max_exp,
                         reps_per_period_point = reps_per_period_point, 
                         scale_fdr=True, 
                         max_magnitude=None, 
                         hyp_type="drug", halfp=False, dbg=False,
                            cut_type="BL"):
    """Stepdown error ests on 1d grid of fixed sample sizes.
    Wrapper builds vector of sample sizes that mixes linear and log scale.
    See get_oc_range and mixed_test_points doc for more info.
    
    """

#     n_period_vec = (base_periods * (2.0 ** np.linspace(0.0, period_max_exp, period_test_points))).astype(int)
    n_period_vec = mixed_test_points(low_sample, high_sample, period_test_points)
    return get_oc_range(p0, p1, 
                        alpha = alpha, 
                        rho = rho, 
                        m_null = m_null, m_alt=m_alt, 
                        n_period_vec=n_period_vec, 
                        reps_per_period_point = reps_per_period_point, 
                        scale_fdr=scale_fdr, 
                        max_magnitude=max_magnitude, 
                        hyp_type=hyp_type, halfp=halfp, dbg=dbg)

    
def fixed_sample_pval(hyp_type, dar, dnar, drr, rho, p0, p1, n_periods, halfp):
                # Generate data
    if (hyp_type is None) or (hyp_type=="drug"):
        amnesia_ts, nonamnesia_ts = data_funcs.simulate_correlated_reactions(n_periods * dar, n_periods * dnar, 2, rho, halfp)
        llr_ts = data_funcs.assemble_llr(amnesia_ts, nonamnesia_ts, p0, p1)
        llr = llr_ts.iloc[-1]
        amnesia = amnesia_ts.iloc[0]
        tot_reacts = nonamnesia_ts.iloc[0] + amnesia
        nrm_approx = cutoff_funcs.llr_term_moments(drr, p0, p1) * n_periods

        Z_scores = (llr - nrm_approx["term_mean"]) / np.sqrt(nrm_approx["term_var"])
        p_val = pd.Series(1 - stats.norm.cdf(Z_scores), index=dar.index)
    elif hyp_type == "binom":
        amnesia_ts = data_funcs.simulate_correlated_binom(dar, n_periods, rho)
        amnesia = pd.DataFrame(amnesia_ts).iloc[-1]
        p_val = pd.Series(dict([(drug_name, single_binom_test(amnesia_val, n_periods, p0, p1, halfp=halfp)) for drug_name, amnesia_val in amnesia.items()]))
        #llr = data_funcs.assemble_binom_llr(amnesia, p0, p1).iloc[0]
        #nrm_approx = llr_binom_term_moments(p0, p1) * n_periods
    elif hyp_type == "pois":
        amnesia_ts = data_funcs.simulate_correlated_pois(dar, n_periods, rho)
        amnesia = pd.DataFrame(amnesia_ts).iloc[-1]
        
        p_val = pd.Series(dict([(drug_name, single_pois_test(amnesia_val, n_periods * p0, n_periods * p1, halfp=halfp)) for drug_name, amnesia_val in amnesia.items()]))
        
        #llr = data_funcs.assemble_pois_llr(amnesia, p0, p1).iloc[0]
        #nrm_approx = llr_pois_term_moments(p0, p1) * n_periods
    else:
        raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))
    return p_val
    
def get_oc_range(p0, p1, 
                 alpha, 
                 rho, 
                 m_null,
                 m_alt,
                 n_period_vec = [],
                 reps_per_period_point = reps_per_period_point, 
                 scale_fdr=True, 
                 max_magnitude=None, 
                 hyp_type="drug", halfp=False, dbg=False, stepup=False, cut_type="BL"):
    """Step down error control estimates via MC sim for fixed sample test.
    FDR, FNR, pFDR, pFNR at each fixed sample size specified by n_period_vec.
    
    """
    
    
    #################    
    period_test_points = len(n_period_vec)
    
    if m_alt is None:
        m_alt = m_null
    
    if (hyp_type is None) or (hyp_type=="drug"):
        if m_alt:
            raise ValueError("m_alt for drugs...")
        dar, dnar, ground_truth = data_funcs.assemble_fake_drugs(max_magnitude, m_null, False, p0, p1)
        drr = dar + dnar
    elif hyp_type == "binom":
        dar, ground_truth = data_funcs.assemble_fake_binom(m_null, False, p0, p1, m_alt=m_alt)
        drr = pd.Series(np.ones(len(dar)), index=dar.index)
        dnar = None
    elif hyp_type == "pois":
        dar, ground_truth = data_funcs.assemble_fake_pois(m_null, False, p0, p1, m_alt=m_alt)
        drr = pd.Series(np.ones(len(dar)), index=dar.index)
        dnar = None
    else:
        raise ValueError("Unrecognized hypothesis type: {0}".format(hyp_type))


    # Calculate alpha cutoffs
    m_hyps = m_null + m_alt
        
    if cut_type=="BY" or "BL":
        alpha_vec_raw = cutoff_funcs.create_fdr_controlled_bl_alpha_indpt(alpha, m_hyps)
    elif cut_type=="BH":
        alpha_vec_raw = alpha * arange(1, 1+m_hyps) / float(m_hyps)
    # Holm
    elif cut_type=="HOLM":        
        alpha_vec_raw = alpha / (float(m_hyps) - arange(m_hyps))
    else:
        raise Exception("Not implemented yet")
        
    if scale_fdr:
        if stepup:
            scaled_alpha_vec = alpha_vec_raw / np.log(m_hyps)
        else:
#            if m0_known:
#                scaled_alpha_vec = alpha * alpha_vec_raw / cutoff_funcs.fdr_helper(alpha_vec_raw, m_total)
#            else:
                
            scaled_alpha_vec = cutoff_funcs.apply_fdr_controlled_alpha(alpha, alpha_vec_raw)
    else: 
        scaled_alpha_vec = alpha_vec_raw
    

    # Set up records
    if dbg:
        dbg_record = dict()
    reps_list = []
    fdr_var_vec = np.zeros(period_test_points)
    fnr_var_vec = np.zeros(period_test_points)
    pfdr_var_vec = np.zeros(period_test_points)
    pfnr_var_vec = np.zeros(period_test_points)
    
    fdr_vec = np.zeros(period_test_points)
    fnr_vec = np.zeros(period_test_points)
    pfdr_vec = np.zeros(period_test_points)
    pfnr_vec = np.zeros(period_test_points)
    
    
    
    for j in tqdm.tqdm_notebook(range(period_test_points), 
                                desc="sample size for {0}".format(hyp_type),
                                position=1):
        # Hack to address high variability for small sample sizes
        n_periods = n_period_vec[j]
        if n_periods < 15:
            reps_per_period_point_ish = reps_per_period_point * 10
        elif (n_periods < 30) and (n_periods >= 20):
            reps_per_period_point_ish = reps_per_period_point * 5
        else:
            reps_per_period_point_ish = reps_per_period_point

        reps_list.append(reps_per_period_point_ish)
        if dbg:
            
            rej_rec = pd.DataFrame(np.zeros((reps_per_period_point_ish, (m_null + m_alt))).astype(float), columns=dar.index)
            pval_rec = pd.DataFrame(np.zeros((reps_per_period_point_ish, (m_null + m_alt))).astype(float), columns=dar.index)
        
        fdp_rec = np.zeros(reps_per_period_point_ish)
        fnp_rec = np.zeros(reps_per_period_point_ish)
        
        # Perform MC simulations for a given fixed sample size
        for i in tqdm.tqdm_notebook(range(reps_per_period_point_ish), desc="reps for {0} periods".format(n_periods),
                                   leave=False, position=2):
            
            
            #######
            
            p_val = fixed_sample_pval(hyp_type, dar, dnar, drr, rho, p0, p1, n_periods, halfp)
                
            # Perform fixed sample stepdown test
            if stepup:
                rej_idx, num_rej = step_up_elimination(p_val, scaled_alpha_vec, 0, 'low')
            else:
                rej_idx, num_rej = step_down_elimination(p_val, scaled_alpha_vec, 0, 'low')
            if dbg:
                pval_rec.loc[i, :] = p_val
                rej_rec.loc[i, rej_idx] = 1.0
                
            
            # Store performance metrics for a given simulation    
            fdp_rec[i] = ground_truth[rej_idx].sum() / max((1, num_rej))
            fnp_rec[i] = ((~ground_truth[~ground_truth.index.isin(rej_idx)]).sum()) / max((1, (m_null + m_alt) - num_rej))
            

        
        # Summarize performance for all MC sims for a given sample size
        if dbg:
            dbg_record[n_periods] = {"pval":pval_rec, "rej":rej_rec}
        fdr_var_vec[j] = fdp_rec.var()
        fnr_var_vec[j] = fnp_rec.var()
        pfdr_var_vec[j] = fdp_rec[fdp_rec > 0].var()
        pfnr_var_vec[j] = fnp_rec[fnp_rec > 0].var()
        fdr_vec[j] = fdp_rec.mean()
        fnr_vec[j] = fnp_rec.mean()
        pfdr_vec[j] = fdp_rec[fdp_rec > 0].mean()
        pfnr_vec[j] = fnp_rec[fnp_rec > 0].mean()
        
        
    if dbg:
        return (pd.DataFrame({"fdr":fdr_vec, "fnr":fnr_vec, 
                                 "pfdr":pfdr_vec, "pfnr":pfnr_vec, 
                                 "fdr_sd":np.sqrt(fdr_var_vec), "fnr_sd":np.sqrt(fnr_var_vec), 
                                 "pfdr_sd":np.sqrt(pfdr_var_vec), "pfnr_sd":np.sqrt(pfnr_var_vec), 
                                 "logfdr":np.log(fdr_vec), "logfnr":np.log(fnr_vec), 
                                 "samplesize":n_period_vec,
                                 "reps":reps_list}), dbg_record)
    else:
        return pd.DataFrame({"fdr":fdr_vec, "fnr":fnr_vec, 
                                 "pfdr":pfdr_vec, "pfnr":pfnr_vec, 
                                 "fdr_sd":np.sqrt(fdr_var_vec), "fnr_sd":np.sqrt(fnr_var_vec), 
                                 "pfdr_sd":np.sqrt(pfdr_var_vec), "pfnr_sd":np.sqrt(pfnr_var_vec), 
                                 "logfdr":np.log(fdr_vec), "logfnr":np.log(fnr_vec), 
                                 "samplesize":n_period_vec,
                                 "reps":reps_list})
    
def error_plots(oc_df, n_se = 2.0, nominal_fdr = .25, main_title = "Poisson (5,5) $\\alpha$=.25",
                output_filepath = "/home/mhankin/Dropbox/Research/MultSeq/Images/fixed_poisson_5_5_25.png"):

    colors = sns.color_palette()
    fdr_color = colors[0]
    fnr_color = colors[1]
    oc_df["pfdr_se"] = n_se*oc_df["pfdr_sd"] / np.sqrt(oc_df["reps"])
    oc_df["pfnr_se"] = n_se*oc_df["pfnr_sd"] / np.sqrt(oc_df["reps"])
    oc_df["fdr_se"] = n_se*oc_df["fdr_sd"] / np.sqrt(oc_df["reps"])
    oc_df["fnr_se"] = n_se*oc_df["fnr_sd"] / np.sqrt(oc_df["reps"])

    error_fig, error_axes = subplots(1, 2)
    fdr_fnr_ax, pfdr_pfnr_ax = error_axes
    oc_df.plot(ax=fdr_fnr_ax, kind="line", x="samplesize", y="fdr", logy=True, yerr="fdr_se", color=fdr_color)
    oc_df.plot(ax=fdr_fnr_ax, kind="line", x="samplesize", y="fnr", logy=True, yerr="fnr_se", color=fnr_color)
    fdr_fnr_ax.legend(["FDR", "FNR"])
    fdr_fnr_ax.axhline(nominal_fdr, color=fdr_color, linestyle="-.")
    fdr_fnr_ax.set_ylabel("Error Metrics (Log Scale)", size=18)
    fdr_fnr_ax.set_xlabel("Sample Size (Fixed Sample)", size=18)
    for tick in fdr_fnr_ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(16)
    for tick in fdr_fnr_ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)

    oc_df.plot(ax=pfdr_pfnr_ax, kind="line", x="samplesize", y="pfdr", logy=True, yerr="pfdr_se", color=fdr_color)
    oc_df.plot(ax=pfdr_pfnr_ax, kind="line", x="samplesize", y="pfnr", logy=True, yerr="pfnr_se", color=fnr_color)
    pfdr_pfnr_ax.legend(["pFDR", "pFNR"])
    pfdr_pfnr_ax.set_ylabel("Error Metrics (Log Scale)", size=18)
    pfdr_pfnr_ax.set_xlabel("Sample Size (Fixed Sample)", size=18)
    for tick in pfdr_pfnr_ax.xaxis.get_major_ticks():
                tick.label.set_fontsize(16)
    for tick in pfdr_pfnr_ax.yaxis.get_major_ticks():
                tick.label.set_fontsize(16)

    error_fig.set_figwidth(14)
    error_fig.suptitle(main_title, size=23)
    error_fig.savefig(output_filepath)
    return error_fig
    
    
import statsmodels.formula.api as sm
def est_equiv_sample_size(alpha, hyp_type, p0, p1, m_null, 
                          seqlogfnr_gen, seqlogfnr_rej, seqlogfnr_gen_nominal, rho, 
                          low_sample=low_sample, high_sample=high_sample,
                          m_alt=None, verbose=True, period_test_points=10, do_exact=True,
                            cut_type="BL", scale_fdr=True, dbg=False, match_nominal=False):
    """Find fixed sample equivalent for given sequential test.
    """
    
    # Get coarse grid of error metrics
    oc_range = get_oc_range_wrapper(alpha=alpha, p0=p0, p1=p1, hyp_type=hyp_type, 
                                    low_sample=low_sample, high_sample=high_sample,
                            m_null=m_null, m_alt=m_alt, period_test_points=period_test_points,
                            cut_type=cut_type, scale_fdr=scale_fdr, rho=rho)
    if verbose:
        print(oc_range)
        
    # Perform regression to estimate equivalent sample size
    ols_result = sm.ols(formula="logfnr ~ 1 + samplesize", data=oc_range).fit().params
    regression_estimates = pd.Series({"gen":(seqlogfnr_gen - ols_result["Intercept"]) / ols_result["samplesize"], 
                          #"rej":(seqlogfnr_rej - ols_result["Intercept"]) / ols_result["samplesize"],
                          "gen_nominal":(seqlogfnr_gen_nominal - ols_result["Intercept"]) / ols_result["samplesize"]}) 
    
    if do_exact:
        # Get fine grid of error metrics near estimate to get exact equivalent.
        n_period_vec = arange(int(regression_estimates["gen"] - 5), int(regression_estimates["gen"] + 7))
        narrow_oc_range = get_oc_range(alpha=alpha, p0=p0, p1=p1, hyp_type=hyp_type, n_period_vec=n_period_vec, 
                                m_null=m_null, m_alt=m_alt, rho=rho,
                            cut_type=cut_type, scale_fdr=scale_fdr)
        
        if match_nominal:
            n_period_vec_nominal = arange(int(regression_estimates["gen_nominal"] - 5), int(regression_estimates["gen_nominal"] + 7))
            narrow_oc_range_nominal = get_oc_range(alpha=alpha, p0=p0, p1=p1, hyp_type=hyp_type, n_period_vec=n_period_vec_nominal, 
                                    m_null=m_null, m_alt=m_alt, rho=rho,
                                cut_type=cut_type, scale_fdr=scale_fdr)
            
            oc_range = pd.concat((oc_range, narrow_oc_range, narrow_oc_range_nominal)).sort_values("samplesize").reset_index(drop=True)
            idx_min_nominal = abs(oc_range["logfnr"] - seqlogfnr_gen_nominal).argmin()
            idx_min = abs(oc_range["logfnr"] - seqlogfnr_gen).argmin()
#            fine_est = pd.Series({"gen":int(oc_range.loc[idx_min, "samplesize"]),
#                                                "gen_nominal":int(oc_range.loc[idx_min_nominal, "samplesize"])})
            fine_est = pd.DataFrame({"gen":oc_range.loc[idx_min, :],
                                                "gen_nominal":oc_range.loc[idx_min_nominal, :]})
        else:
            oc_range = pd.concat((oc_range, narrow_oc_range)).sort_values("samplesize").reset_index(drop=True)
            idx_min = abs(oc_range["logfnr"] - seqlogfnr_gen).argmin()
            
            fine_est = pd.Series(oc_range.loc[idx_min, :])
        


        
        
        if dbg:
            return (fine_est, 
                    regression_estimates,  oc_range)
        else:
            return fine_est
    else:
        if dbg:
            return regression_estimates, oc_range
        else:
            return regression_estimates
        
    
def get_ave_samp_ser(fullrec, hyp_type, m0, rejective=True):
    """ For infinite horizon, build series before matching"""
    if rejective:
        term_time = fullrec[hyp_type].fillna(fullrec["Horizon"])
        rej_ser = pd.Series({
                     "H0 ASN":term_time.iloc[:,:m0].mean().mean(),
                     "Ha ASN":term_time.iloc[:,m0:].mean().mean(),
                     "Seq Achieved FDR":fullrec["FDR"][hyp_type]["rej_FDR"],
                     "Seq Achieved FNR":fullrec["FDR"][hyp_type]["rej_FNR"],
                     "Seq Achieved pFDR":fullrec["FDR"][hyp_type]["rej_pFDR"],
                     "Seq Achieved pFNR":fullrec["FDR"][hyp_type]["rej_pFNR"],
                     "Nominal FDR":fullrec["config"].getfloat("alpha_rejective"),
                     "Horizon":fullrec["Horizon"]})
        if "rej_FDRse" in fullrec["FDR"][hyp_type]:
            se_ser = pd.Series({"Seq FDR se":fullrec["FDR"][hyp_type]["rej_FDRse"],
                     "Seq FNR se":fullrec["FDR"][hyp_type]["rej_FNRse"],
                     "Seq pFDR se":fullrec["FDR"][hyp_type]["rej_pFDRse"],
                     "Seq pFNR se":fullrec["FDR"][hyp_type]["rej_pFNRse"]})
            rej_ser = pd.concat((rej_ser, se_ser))
#        rej_ser = pd.concat((rej_ser, fullrec[hyp_type].mean(),
#                                 (fullrec[hyp_type]<50).mean()))
#        print(fullrec[hyp_type])
#        raise Exception()
        return rej_ser
    else:
        gen_ser =  pd.Series({
                     "H0 ASN":fullrec[hyp_type].iloc[:,:m0].mean().mean(),
                     "Ha ASN":fullrec[hyp_type].iloc[:,m0:].mean().mean(),
                     "Seq Achieved FDR":fullrec["FDR"][hyp_type]["gen_FDR"],
                     "Seq Achieved FNR":fullrec["FDR"][hyp_type]["gen_FNR"],
                     "Nominal FDR":fullrec["config"].getfloat("alpha_general"),
                     "Nominal FNR":fullrec["config"].getfloat("beta_general")})
        if "gen_FDRse" in fullrec["FDR"][hyp_type]:
            se_ser = pd.Series({"Seq FDR se":fullrec["FDR"][hyp_type]["gen_FDRse"],
                     "Seq FNR se":fullrec["FDR"][hyp_type]["gen_FNRse"],
                     "Seq pFDR se":fullrec["FDR"][hyp_type]["gen_pFDRse"],
                     "Seq pFNR se":fullrec["FDR"][hyp_type]["gen_pFNRse"]})
            gen_ser = pd.concat((gen_ser, se_ser))
        return gen_ser
    
    
def term_time_general(shfp, cfgfp, cfgsect, rejective=False):
    """Retrieve termination time and error metric of seq table from shelve file
    """
    rec_filepath = os.path.expanduser(shfp)
    with closing(shelve.open(rec_filepath)) as shf:
        FDR_table = shf["FDR_table"]
        seq_pois_data_general = shf["synth_pois_data_general"]
        seq_pois_unscaled_data_general = shf["synth_pois_unscaled_data_general"]
        seq_binom_data_general = shf["synth_binom_data_general"]
        
        seq_pois_data_rejective = shf["synth_pois_data_rejective"]
        seq_pois_unscaled_data_rejective = shf["synth_pois_unscaled_data_rejective"]
        seq_binom_data_rejective = shf["synth_binom_data_rejective"]
        
    config = configparser.ConfigParser(inline_comment_prefixes=["#"], default_section="default")
    config.read([os.path.expanduser(cfgfp)])
    config_section = config[cfgsect]
    m0 = config_section.getint("m_null")
    m1 = config_section.getint("m_alt")
    if rejective:
        fullrec = {"pois":seq_pois_data_rejective[1], 
                   "pois_unscaled":seq_pois_unscaled_data_rejective[1],
                   "binom":seq_binom_data_rejective[1], 
                   "FDR":FDR_table, "config":config_section, "m0":m0, "m1":m1,
                   "Horizon":config_section.getint("n_periods_rejective")}
    else:
        fullrec = {"pois":seq_pois_data_general[1], 
                   "pois_unscaled":seq_pois_unscaled_data_general[1],
                   "binom":seq_binom_data_general[1], 
                   "FDR":FDR_table, "config":config_section, "m0":m0, "m1":m1}
    fullrec["ave_samp_pois"] = get_ave_samp_ser(fullrec, "pois", m0, rejective=rejective)
    fullrec["ave_samp_pois_unscaled"] = get_ave_samp_ser(fullrec, "pois_unscaled", m0, rejective=rejective)
    fullrec["ave_samp_binom"] = get_ave_samp_ser(fullrec, "binom", m0, rejective=rejective)
    return fullrec
#%% Finite horizon
def build_fs_metric_ser(fdp_ser, fnp_ser):
    print(fnp_ser.shape)
    fdr_var = fdp_ser.var()
    fnr_var = fnp_ser.var()
    pfdr_var = fdp_ser[fdp_ser > 0].var()
    pfnr_var = fnp_ser[fnp_ser > 0].var()
    
    fdr_se = np.sqrt(fdr_var / len(fdp_ser))
    fnr_se = np.sqrt(fnr_var / len(fnp_ser))
    pfdr_se = np.sqrt(pfdr_var / sum(fdp_ser>0))
    pfnr_se = np.sqrt(pfnr_var / sum(fnp_ser>0))
    
    
    fdr = fdp_ser.mean()
    fnr = fnp_ser.mean()
    pfdr = fdp_ser[fdp_ser > 0].mean()
    pfnr = fnp_ser[fnp_ser > 0].mean()
    
    metric_ser = pd.Series({"fdr":fdr, "fnr":fnr, 
                                 "pfdr":pfdr, "pfnr":pfnr, 
                                 "fdr_se":fdr_se, "fnr_se":fnr_se, 
                                 "pfdr_se":pfdr_se, "pfnr_se":pfnr_se, 
                                 "logfdr":np.log(fdr), "logfnr":np.log(fnr)})
    return metric_ser

def finite_horizon_equivalent_oc(shfp, cfgfp, cfgsect, n_reps=100, halfp=True):
    fullrec = term_time_general(shfp, cfgfp, cfgsect, rejective=True)
    n_periods = fullrec["config"].getint("n_periods_rejective")
    m0 = fullrec["m0"]
    m1 = fullrec["m1"]
    rho = fullrec["config"].getfloat("rho")
    alpha = fullrec["config"].getfloat("alpha_rejective")
    m_hyps = m0 + m1
    if "stepup" in fullrec["config"]:
        stepup = fullrec["config"].getboolean("stepup")
    else:
        stepup = False
        
    if "scale_fdr" in fullrec["config"]:
        scale_fdr = fullrec["config"].getboolean("scale_fdr")
    else:
        scale_fdr = True
        
    cut_type = fullrec["config"].get("cut_type_general")
        
    if (cut_type=="BY") or (cut_type=="BL"):
        alpha_vec_raw = cutoff_funcs.create_fdr_controlled_bl_alpha_indpt(alpha, m_hyps)
    elif cut_type=="BH":
        alpha_vec_raw = alpha * arange(1, 1+m_hyps) / float(m_hyps)
    # Holm
    elif cut_type=="HOLM":        
        alpha_vec_raw = alpha / (float(m_hyps) - arange(m_hyps))
    else:
        raise Exception("Not implemented yet: "+str(cut_type))
        
    if scale_fdr:
        if stepup:
            scaled_alpha_vec = alpha_vec_raw / np.log(m_hyps)
        else:
#            if m0_known:
#                scaled_alpha_vec = alpha * alpha_vec_raw / cutoff_funcs.fdr_helper(alpha_vec_raw, m_total)
#            else:
                
            scaled_alpha_vec = cutoff_funcs.apply_fdr_controlled_alpha(alpha, alpha_vec_raw)
    else: 
        scaled_alpha_vec = alpha_vec_raw
    
    #pois
    fdp_rec_pois = np.zeros(n_reps, float)
    fnp_rec_pois = np.zeros(n_reps, float)
    fdp_rec_pois_unscaled = np.zeros(n_reps, float)
    fnp_rec_pois_unscaled = np.zeros(n_reps, float)
    lam0 = fullrec["config"].getfloat("lam0")
    lam1 = fullrec["config"].getfloat("lam1")
    dar, ground_truth = data_funcs.assemble_fake_pois(m0, False, lam0, lam1, m_alt=m1)
    drr = pd.Series(np.ones(len(dar)), index=dar.index)
    dnar = None
    for i in tqdm.tqdm(range(n_reps), "pois FH reps"):
        p_val = fixed_sample_pval("pois", dar, dnar, drr, rho, lam0, lam1, n_periods, halfp)

                    # Perform fixed sample stepdown test
        if stepup:
            rej_idx, num_rej = step_up_elimination(p_val, scaled_alpha_vec, 0, 'low')
            rej_idx_unscaled, num_rej_unscaled = step_up_elimination(p_val, alpha_vec_raw, 0, 'low')
        else:
            rej_idx, num_rej = step_down_elimination(p_val, scaled_alpha_vec, 0, 'low')
            rej_idx_unscaled, num_rej_unscaled = step_down_elimination(p_val, alpha_vec_raw, 0, 'low')
#        if dbg:
#            pval_rec.loc[i, :] = p_val
#            rej_rec.loc[i, rej_idx] = 1.0
            
        
        # Store performance metrics for a given simulation    
        fdp_rec_pois[i] = ground_truth[rej_idx].sum() / max((1, num_rej))
        fnp_rec_pois[i] = ((~ground_truth[~ground_truth.index.isin(rej_idx)]).sum()) / max((1, m_hyps - num_rej))
        fdp_rec_pois_unscaled[i] = ground_truth[rej_idx_unscaled].sum() / max((1, num_rej_unscaled))
        fnp_rec_pois_unscaled[i] = ((~ground_truth[~ground_truth.index.isin(rej_idx_unscaled)]).sum()) / max((1, m_hyps - num_rej_unscaled))
     
    pois_ser = build_fs_metric_ser(fdp_rec_pois, fnp_rec_pois)
    pois_ser.index = map(lambda metric: "fs_"+metric, pois_ser.index)
    pois_ser = pd.concat((pois_ser, fullrec["ave_samp_pois"]))

    pois_unscaled_ser = build_fs_metric_ser(fdp_rec_pois_unscaled, fnp_rec_pois_unscaled)
    pois_unscaled_ser.index = map(lambda metric: "fs_"+metric, pois_unscaled_ser.index)
    pois_unscaled_ser = pd.concat((pois_unscaled_ser, fullrec["ave_samp_pois_unscaled"]))



    #binom
    fdp_rec_binom = np.zeros(n_reps, float)
    fnp_rec_binom = np.zeros(n_reps, float)
    p0 = fullrec["config"].getfloat("p0")
    p1 = fullrec["config"].getfloat("p1")
    dar, ground_truth = data_funcs.assemble_fake_binom(m0, False, p0, p1, m_alt=m1)
    drr = pd.Series(np.ones(len(dar)), index=dar.index)
    dnar = None
    for i in tqdm.tqdm(range(n_reps), "binom FH reps"):
        p_val = fixed_sample_pval("binom", dar, dnar, drr, rho, p0, p1, n_periods, halfp)
                    # Perform fixed sample stepdown test
        if stepup:
            rej_idx, num_rej = step_up_elimination(p_val, scaled_alpha_vec, 0, 'low')
        else:
            rej_idx, num_rej = step_down_elimination(p_val, scaled_alpha_vec, 0, 'low')
#        if dbg:
#            pval_rec.loc[i, :] = p_val
#            rej_rec.loc[i, rej_idx] = 1.0
            
        
        # Store performance metrics for a given simulation    
        fdp_rec_binom[i] = ground_truth[rej_idx].sum() / max((1, num_rej))
        fnp_rec_binom[i] = ((~ground_truth[~ground_truth.index.isin(rej_idx)]).sum()) / max((1, m_hyps - num_rej))
        

    binom_ser =  build_fs_metric_ser(fdp_rec_binom, fnp_rec_binom)
    binom_ser.index = map(lambda metric: "fs_"+metric, binom_ser.index)
    binom_ser = pd.concat((binom_ser, fullrec["ave_samp_binom"]))
        
    print(cfgsect, pois_ser)

    return {"pois":pois_ser,"pois_unscaled":pois_unscaled_ser, "binom":binom_ser, "m0":m0, "m1":m1, "Horizon":n_periods, "FDR":fullrec["FDR"]}
        

        
#%% Infinite horizon section
def infinite_horizon_equivalent(shfp, cfgfp,  cfgsect, do_exact=False, dbg=False, verbose=False):
    fullrec = term_time_general(shfp, cfgfp, cfgsect)
    cut_type = fullrec["config"].get("cut_type_general")
    
    m0 = fullrec["m0"]
    m1 = fullrec["m1"]
    if "scale_fdr" in fullrec["config"]:
        scale_fdr = fullrec["config"].getboolean("scale_fdr")
    else:
        scale_fdr = True
        
    rho = fullrec["config"].getfloat("rho")
    
        
    pois_df = est_equiv_sample_size(fullrec["config"].getfloat("alpha_general"), 
                          hyp_type="pois", p0=fullrec["config"].getfloat("lam0"), 
                          p1=fullrec["config"].getfloat("lam1"),
                          m_null=fullrec["m0"], m_alt=fullrec["m1"], 
                          low_sample=10, high_sample=125,
                          seqlogfnr_gen=np.log(fullrec["FDR"]["pois"]["gen_FNR"]), 
                          seqlogfnr_rej=np.log(fullrec["FDR"]["pois"]["rej_FNR"]), 
                          seqlogfnr_gen_nominal=np.log(fullrec["config"].getfloat("beta_general")),
                          do_exact=do_exact, cut_type=cut_type, scale_fdr=scale_fdr, rho=rho,
                          dbg=dbg, verbose=verbose)
    binom_df = est_equiv_sample_size(fullrec["config"].getfloat("alpha_general"), 
                          hyp_type="binom", p0=fullrec["config"].getfloat("p0"), 
                          p1=fullrec["config"].getfloat("p1"),
                          m_null=fullrec["m0"], m_alt=fullrec["m1"], 
                          low_sample=10, high_sample=125,
                          seqlogfnr_gen=np.log(fullrec["FDR"]["binom"]["gen_FNR"]), 
                          seqlogfnr_rej=np.log(fullrec["FDR"]["binom"]["rej_FNR"]), 
                          seqlogfnr_gen_nominal=np.log(fullrec["config"].getfloat("beta_general")),
                          do_exact=do_exact, cut_type=cut_type, scale_fdr=scale_fdr, rho=rho,
                          dbg=dbg, verbose=verbose)

    if dbg:
        return {"pois":pois_df[0], "binom":binom_df[0], "m0":m0, "m1":m1, 
                "pois_dbg":pois_df[1:], "binom_dbg":binom_df[1:],
                "seq_pois":fullrec["ave_samp_pois"],
                "seq_binom":fullrec["ave_samp_binom"]}
    else:
        return {"pois":pois_df, "binom":binom_df, "m0":m0, "m1":m1,
                "seq_pois":fullrec["ave_samp_pois"],
                "seq_binom":fullrec["ave_samp_binom"]}

