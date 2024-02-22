#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 26 21:57:48 2016

@author: mhankin
"""

#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK
"""
run_sims runs the main simulations
Created on Thu Oct 27 12:12:55 2016

@author: mike
"""
# %% Arg imports
import argparse, argcomplete, os
# Parallelization
parser = argparse.ArgumentParser()
parser.add_argument('--singlecore', action='store_true', help="Dont parallelize main sims")
parser.add_argument('--skipsynth', action='store_true', help="skip synthetic sims and load data")
parser.add_argument('--skiprej', action='store_true', help="skip rejective procedures")
parser.add_argument('--skipgen', action='store_true', help="skip general procedures")
parser.add_argument('--skipfixed', action='store_true', help="skip fixed sample equivalents")
parser.add_argument('--doviz', action='store_true', help="do visualizations")
parser.add_argument('--shelvepath', default="~/Dropbox/Research/MultSeq/Data/binpois.shelve", help="shelve record path")
parser.add_argument('--vizpath', default="~/Dropbox/Research/MultSeq/Data/binpois.html", help="Visualization html path")
parser.add_argument('--cfgpath', default="~/Dropbox/Research/MultSeq/Data/sim.cfg", help="Simulation configuration file")
parser.add_argument('--cfgsect', default="binpois", help="Simulation configuration section")
parser.add_argument('--usesect', action='store_true', help="Use default roots with cfgsect name")
parser.add_argument('--scomponly', action='store_true', help="Only plot scaled-unscaled comparison")

parser.add_argument('--stifle', action='store_true', help="don't actually show viz")


argcomplete.autocomplete(parser)
cmd_args = parser.parse_args()

if cmd_args.usesect:
    cmd_args.vizpath = "~/Dropbox/Research/MultSeq/Data/{0}.html".format(cmd_args.cfgsect)
    cmd_args.shelvepath = "~/Dropbox/Research/MultSeq/Data/{0}.shelve".format(cmd_args.cfgsect)


# %% Main imports
from pylab import *
    
import multseq, visualizations
from utils import cutoff_funcs, data_funcs, common_funcs, simulation_funcs, sim_analysis

from IPython.display import display
import itertools, functools
import pandas
import shelve
import sys
import numpy
from contextlib import closing
from configparser import ConfigParser

import logging
import tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# %% Reload after code changes

class TqdmLoggingHandler (logging.Handler):
    def __init__ (self, level = logging.NOTSET):
        super (self.__class__, self).__init__ (level)

    def emit (self, record):
        try:
            msg = self.format (record)
            tqdm.tqdm.write (msg)
            self.flush ()
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record) 

logger.addHandler(TqdmLoggingHandler())


# %% Global parameter settings


# store data
rec_filepath = os.path.expanduser(cmd_args.shelvepath)


config = ConfigParser(inline_comment_prefixes=["#"], default_section="default")
config.read([os.path.expanduser(cmd_args.cfgpath)])
config_section = config[cmd_args.cfgsect]

# Drug screening 
# config_section.getint("")
min_am = config_section.getint("min_am")
min_tot = config_section.getint("min_tot")
with closing(shelve.open(rec_filepath)) as shf:
    shf["min_am"] = min_am
    shf["min_tot"] = min_tot
    shf["config_dict"] = dict(config_section.items())

plot_synth = config_section.getboolean("plot_synth")
plot_drugs = config_section.getboolean("plot_drugs")
glyph_size = config_section.getint("glyph_size")
# Drug Rejective
alpha_rejective = config_section.getfloat("alpha_rejective") 
cut_type_rejective = config_section.get("cut_type_rejective")
undershoot_drug_rejective =  config_section.getfloat("undershoot_drug_rejective")
undershoot_synth_rejective = config_section.getfloat("undershoot_synth_rejective")
sim_reps = config_section.getint("sim_reps")
n_periods_rejective = config_section.getint("n_periods_rejective")

# Drug General
do_iter_MC = config_section.getboolean("do_iter_MC")
alpha_general = config_section.getfloat("alpha_general")
beta_general = config_section.getfloat("beta_general")
cut_type_general = config_section.get("cut_type_general")
undershoot_drug_general = config_section.getfloat("undershoot_drug_general")
undershoot_synth_general = config_section.getfloat("undershoot_synth_general")
n_se = config_section.getfloat("n_se")
am_prop_pctl = (config_section.getfloat("am_prop_pctl_low"),
                config_section.getfloat("am_prop_pctl_high"))

m_null = config_section.getint("m_null")
if "m_alt" in config_section:
    m_alt = config_section.getint("m_alt")
else:
    m_alt = m_null

max_mag = config_section.getfloat("max_mag")


use_am_prop = config_section.getboolean("use_am_prop")

rho = config_section.getfloat("rho")

if "split_corr" in config_section:
    split_corr = config_section.getboolean("split_corr")
    if "rho1" in config_section:
        rho1 = config_section.getfloat("rho1")
    else:
        rho1 = rho
else:
    split_corr = False
    rho1 = None
    
if "rand_order" in config_section:    
    rand_order = config_section.getboolean("rand_order")
else:
    rand_order = False

p0 = config_section.getfloat("p0")
p1 = config_section.getfloat("p1")

lam0 = config_section.getfloat("lam0")
lam1 = config_section.getfloat("lam1")

if "stepup" in config_section:    
    stepup = config_section.getboolean("stepup")
else:
    stepup = False
    
    
if "cummax" in config_section:    
    cummax= config_section.getboolean("cummax")
else:
    cummax = False
    
if "fh_cutoff_imp_sample" in config_section:
    fh_cutoff_imp_sample = config_section.getboolean("fh_cutoff_imp_sample")
else: 
    fh_cutoff_imp_sample =False 
    
if "fh_cutoff_imp_sample_prop" in config_section:
    fh_cutoff_imp_sample_prop = config_section.getfloat("fh_cutoff_imp_sample_prop")
else:
    fh_cutoff_imp_sample_prop = .5
    
if "fh_cutoff_normal_approx" in config_section:
    fh_cutoff_normal_approx = config_section.getboolean("fh_cutoff_normal_approx")
else:
    fh_cutoff_normal_approx = False
    
if fh_cutoff_normal_approx and fh_cutoff_imp_sample:
    raise Exception("Can't use normal approx and importance sampling")
    
if "fh_sleep_time" in config_section:
    fh_sleep_time = config_section.getint("fh_sleep_time")
else:
    fh_sleep_time = 20
    
    
    
    
def compute_fdr(rej_rec, truth_mask): #=numpy.repeat([True, False], m_null)):
    fdp_rec = pandas.DataFrame(dict(
    FDP = ((truth_mask & (rej_rec.isin(["rej"]))).sum(1) / (rej_rec.isin(["rej"]).sum(1))).fillna(0),
    FNP = ((~truth_mask & (rej_rec.isin(["acc"]))).sum(1) / (rej_rec.isin(["acc"]).sum(1))).fillna(0)))
    pfdp_rec = fdp_rec.copy()
    pfdp_rec[pfdp_rec==0.0] = np.NaN
    
    some_rej_mask = (rej_rec.isin(["rej"])).sum(1) > 0
    some_acc_mask = (rej_rec.isin(["acc"])).sum(1) > 0
#    fdp_rec["pFDR"] = ((truth_mask & (rej_rec[some_rej_mask].isin(["rej"]))).sum(1) / (rej_rec[some_rej_mask].isin(["rej"]).sum(1))).fillna(0)
#    fdp_rec["pFNR"] = ((~truth_mask & (rej_rec[some_acc_mask].isin(["acc"]))).sum(1) / (rej_rec[some_acc_mask].isin(["acc"]).sum(1))).fillna(0)
#    fdp_rec.loc[fdp_rec["pFDR"]==0.0, "pFDR"] = NaN
#    fdp_rec.loc[fdp_rec["pFNR"]==0.0, "pFNR"] = NaN
    

    total_with_rej = some_rej_mask.sum()
    total_with_acc = some_acc_mask.sum()
    
    
    fdr_ser = pandas.Series(dict(
    pFDR=pfdp_rec["FDP"].mean(skipna=True),
    pFNR=pfdp_rec["FNP"].mean(skipna=True),
    FDR=fdp_rec["FDP"].fillna(0.0).mean(),
    FNR=fdp_rec["FNP"].fillna(0.0).mean(),
    pFDRse=sqrt(pfdp_rec["FDP"].var(skipna=True)/total_with_rej),
    pFNRse=sqrt(pfdp_rec["FNP"].mean(skipna=True)/total_with_acc),
    FDRse=sqrt(fdp_rec["FDP"].fillna(0.0).var()/len(fdp_rec)),
    FNRse=sqrt(fdp_rec["FNP"].fillna(0.0).var()/len(fdp_rec))))
    return fdr_ser




print("Starting computation for ", cmd_args.cfgsect)
# %% Synthetic data

with closing(shelve.open(rec_filepath)) as shf:
    shf["m_null"] = m_null
    shf["max_mag"] = max_mag
    

if not cmd_args.skipgen:
    figure(1)
    synth_binom_data_general = simulation_funcs.synth_data_sim(alpha=alpha_general, beta=beta_general, 
                                   max_magnitude=max_mag, m_null=m_null, m0_known=False, 
                                   rho=rho, interleaved=False, 
                                   do_parallel=not cmd_args.singlecore, 
                                   undershoot_prob=undershoot_synth_general, 
                                   sim_reps=sim_reps, rej_hist=True,
                                   hyp_type="binom", p0=p0, p1=p1, do_iterative_cutoff_MC_calc=do_iter_MC,
                                   m_alt=m_alt, stepup=stepup, 
                                   cut_type=cut_type_general, split_corr=split_corr, 
                                   rho1=rho1, rand_order=rand_order)
                                   
    with closing(shelve.open(rec_filepath)) as shf:
        shf["synth_binom_data_general"] = synth_binom_data_general
        shf["synth_binom_fdp_general"] = compute_fdr(rej_rec = synth_binom_data_general[0], truth_mask=synth_binom_data_general[2])
        
    synth_pois_grad_data_general = simulation_funcs.synth_data_sim(alpha=alpha_general, beta=beta_general, 
                                   max_magnitude=max_mag, m_null=m_null, m0_known=False, 
                                   rho=rho, interleaved=False, m_alt=m_alt,
                                   do_parallel=not cmd_args.singlecore, 
                                   undershoot_prob=undershoot_synth_general, 
                                   sim_reps=sim_reps, rej_hist=True,
                                   hyp_type="pois_grad", p0=lam0, p1=lam1, 
                                   do_iterative_cutoff_MC_calc=do_iter_MC, 
                                   stepup=stepup, cut_type=cut_type_general, 
                                   split_corr=split_corr, rho1=rho1, 
                                   rand_order=rand_order)
                                   
    with closing(shelve.open(rec_filepath)) as shf:
        shf["synth_pois_grad_data_general"] = synth_pois_grad_data_general
        
    synth_pois_data_general = simulation_funcs.synth_data_sim(alpha=alpha_general, beta=beta_general, 
                                   max_magnitude=max_mag, m_null=m_null, m0_known=False, 
                                   rho=rho, interleaved=False, 
                                   do_parallel=not cmd_args.singlecore, 
                                   undershoot_prob=undershoot_synth_general, 
                                   sim_reps=sim_reps, rej_hist=True,
                                   hyp_type="pois", p0=lam0, p1=lam1, do_iterative_cutoff_MC_calc=do_iter_MC,
                                   m_alt=m_alt, stepup=stepup, cut_type=cut_type_general, 
                                   split_corr=split_corr, rho1=rho1, 
                                   rand_order=rand_order)
                                   
    with closing(shelve.open(rec_filepath)) as shf:
        shf["synth_pois_data_general"] = synth_pois_data_general
        shf["synth_pois_fdp_general"] = compute_fdr(rej_rec = synth_pois_data_general[0], truth_mask=synth_pois_data_general[2])
        
    synth_pois_unscaled_data_general = simulation_funcs.synth_data_sim(alpha=alpha_general, beta=beta_general, 
                                   max_magnitude=max_mag, m_null=m_null, m0_known=False, 
                                   rho=rho, interleaved=False, scale_fdr=False,
                                   do_parallel=not cmd_args.singlecore, 
                                   undershoot_prob=undershoot_synth_general, 
                                   sim_reps=sim_reps, rej_hist=True,
                                   hyp_type="pois", p0=lam0, p1=lam1, do_iterative_cutoff_MC_calc=do_iter_MC,
                                   m_alt=m_alt, stepup=stepup, cut_type=cut_type_general, 
                                   split_corr=split_corr, rho1=rho1, 
                                   rand_order=rand_order)
                                   
    with closing(shelve.open(rec_filepath)) as shf:
        shf["synth_pois_unscaled_data_general"] = synth_pois_unscaled_data_general
        shf["synth_poisunscaled_fdp_general"] = compute_fdr(rej_rec = synth_pois_unscaled_data_general[0], 
                                                            truth_mask=synth_pois_unscaled_data_general[2])
        
if not cmd_args.skiprej:
    # %% 
    synth_binom_data_rejective = simulation_funcs.synth_data_sim(alpha=alpha_rejective, beta=None, 
                                   max_magnitude=max_mag, m_null=m_null, m0_known=False, 
                                   rho=rho, interleaved=False, 
                                   undershoot_prob=undershoot_synth_rejective, 
                                   sim_reps=sim_reps, rej_hist=True,
                                   do_parallel=not cmd_args.singlecore, fin_par=not cmd_args.singlecore, 
                                   n_periods=n_periods_rejective,
                                   hyp_type="binom", p0=p0, p1=p1, fh_sleep_time=fh_sleep_time, 
                                   m_alt=m_alt, stepup=stepup, cut_type=cut_type_general, 
                                   fh_cutoff_normal_approx=fh_cutoff_normal_approx,
                                   fh_cutoff_imp_sample=fh_cutoff_imp_sample,
                                   fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop,
                                   split_corr=split_corr, rho1=rho1, 
                                   rand_order=rand_order, cummax=cummax)
                                   
    with closing(shelve.open(rec_filepath)) as shf:
        shf["synth_binom_data_rejective"] = synth_binom_data_rejective
        shf["synth_binom_fdp_rejective"] = compute_fdr(rej_rec = synth_binom_data_rejective[0], 
                                                            truth_mask=synth_binom_data_rejective[2])
        shf["undershoot_synth_rejective"] = undershoot_synth_rejective

    synth_pois_grad_data_rejective = simulation_funcs.synth_data_sim(alpha=alpha_rejective, beta=None, 
                                   max_magnitude=max_mag, m_null=m_null, m0_known=False, 
                                   rho=rho, interleaved=False, m_alt=m_alt,
                                   undershoot_prob=undershoot_synth_rejective, 
                                   sim_reps=sim_reps, rej_hist=True,
                                   do_parallel=not cmd_args.singlecore, fin_par=not cmd_args.singlecore, 
                                   n_periods=n_periods_rejective,
                                   hyp_type="pois_grad", p0=lam0, p1=lam1, 
                                   do_viz=False, fh_sleep_time=fh_sleep_time, stepup=stepup, cut_type=cut_type_general, 
                                   fh_cutoff_normal_approx=fh_cutoff_normal_approx,
                                   fh_cutoff_imp_sample=fh_cutoff_imp_sample,
                                   fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop,
                                   split_corr=split_corr, rho1=rho1, 
                                   rand_order=rand_order, cummax=cummax)
    
    with closing(shelve.open(rec_filepath)) as shf:
        shf["synth_pois_grad_data_rejective"] = synth_pois_grad_data_rejective
        
    synth_pois_data_rejective = simulation_funcs.synth_data_sim(alpha=alpha_rejective, beta=None, 
                                   max_magnitude=max_mag, m_null=m_null, m0_known=False, 
                                   rho=rho, interleaved=False, 
                                   undershoot_prob=undershoot_synth_rejective, 
                                   sim_reps=sim_reps, rej_hist=True,
                                   do_parallel=not cmd_args.singlecore, fin_par=not cmd_args.singlecore, 
                                   n_periods=n_periods_rejective,
                                   hyp_type="pois", p0=lam0, p1=lam1, do_viz=False, 
                                   fh_sleep_time=fh_sleep_time,
                                   m_alt=m_alt, stepup=stepup, cut_type=cut_type_general, 
                                   fh_cutoff_normal_approx=fh_cutoff_normal_approx,
                                   fh_cutoff_imp_sample=fh_cutoff_imp_sample,
                                   fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop, 
                                   split_corr=split_corr, rho1=rho1, 
                                   rand_order=rand_order, cummax=cummax)
    
    with closing(shelve.open(rec_filepath)) as shf:
        shf["synth_pois_data_rejective"] = synth_pois_data_rejective
        shf["synth_pois_fdp_rejective"] = compute_fdr(rej_rec = synth_pois_data_rejective[0], 
                                                            truth_mask=synth_pois_data_rejective[2])
            
    synth_pois_unscaled_data_rejective = simulation_funcs.synth_data_sim(alpha=alpha_rejective, beta=None, 
                                   max_magnitude=max_mag, m_null=m_null, m0_known=False, 
                                   rho=rho, interleaved=False, 
                                   undershoot_prob=undershoot_synth_rejective/m_null, # Kludgey fix for the undersampling issue
                                   sim_reps=sim_reps, rej_hist=True,
                                   do_parallel=not cmd_args.singlecore, fin_par=not cmd_args.singlecore, 
                                   n_periods=n_periods_rejective, scale_fdr=False,
                                   hyp_type="pois", p0=lam0, p1=lam1, do_viz=False, 
                                   fh_sleep_time=fh_sleep_time,
                                   m_alt=m_alt, stepup=stepup, cut_type=cut_type_general, 
                                   fh_cutoff_normal_approx=fh_cutoff_normal_approx,
                                   fh_cutoff_imp_sample=fh_cutoff_imp_sample,
                                   fh_cutoff_imp_sample_prop=fh_cutoff_imp_sample_prop,
                                   fh_cutoff_imp_sample_hedge=.5, 
                                   split_corr=split_corr, rho1=rho1, 
                                   rand_order=rand_order, cummax=cummax)
    
    with closing(shelve.open(rec_filepath)) as shf:
        shf["synth_pois_unscaled_data_rejective"] = synth_pois_unscaled_data_rejective
        shf["synth_pois_unscaled_fdp_rejective"] = compute_fdr(rej_rec = synth_pois_unscaled_data_rejective[0], 
                                                            truth_mask=synth_pois_unscaled_data_rejective[2])
                               
with closing(shelve.open(rec_filepath)) as shf:
    synth_pois_data_rejective = shf["synth_pois_data_rejective"]
    synth_pois_data_general = shf["synth_pois_data_general"]
    synth_pois_grad_data_rejective = shf["synth_pois_grad_data_rejective"]
    synth_pois_grad_data_general = shf["synth_pois_grad_data_general"]
    synth_pois_unscaled_data_rejective = shf["synth_pois_unscaled_data_rejective"]
    synth_pois_unscaled_data_general = shf["synth_pois_unscaled_data_general"]
    synth_binom_data_rejective = shf["synth_binom_data_rejective"]
    synth_binom_data_general = shf["synth_binom_data_general"]
#        figure(1)
#        synth_gen_res = simulation_funcs.synth_data_sim(alpha=alpha_general, beta=beta_general, 
#                               max_magnitude=max_mag, m_null=m_null, m0_known=False, 
#                               rho=-.2, interleaved=False, 
#                               do_parallel=not cmd_args.singlecore, 
#                               undershoot_prob=undershoot_synth_general, 
#                               sim_reps=None, rej_hist=True,
#                               hyp_type="pois", p0=lam0, p1=lam1, do_viz=True)

# %% Fixed sample equivalencies


# %% visualizations

if cmd_args.skipfixed:
    pass
else:
    pass

from bokeh.plotting import output_file
from bokeh.models import Range1d, ColumnDataSource
from bokeh.layouts import gridplot, widgetbox, column
from bokeh.models.widgets import DataTable, StringFormatter, TableColumn, Paragraph, Panel, Div
from bokeh.io import show as bokeh_show
from bokeh.io import save as bokeh_save
import pickle

if True or cmd_args.doviz:
    output_file(os.path.expanduser(cmd_args.vizpath), mode="inline") # autosave=True,
    
    
    # build header text
    header_text = ("<h1>Rejection plots for:</h1>"
                  " <br/>&nbsp;&nbsp;&nbsp;&alpha;={alpha}&nbsp;&nbsp;&beta; = {beta}"
                  "&nbsp;&nbsp;&nbsp;&rho;={rho}"
                  "&nbsp;&nbsp;&nbsp;m<sub>0</sub>={m0}&nbsp;&nbsp;m<sub>1</sub>={m1}"
                  " <br/>&nbsp;&nbsp;&nbsp;&lambda;<sub>0</sub>={lam0}&nbsp;&nbsp;&lambda;<sub>1</sub>={lam1}"
                  "&nbsp;&nbsp;&nbsp;p<sub>0</sub>={p0}&nbsp;&nbsp;p<sub>1</sub>={p1}"
                  " <br/>&nbsp;&nbsp;&nbsp;Step-up = {su}&nbsp;&nbsp;cut style = {cutstyle}"
                  " <br/>&nbsp;&nbsp;&nbsp;Finite Horizon = {horizon}"
                  "").format(alpha=alpha_general, beta=beta_general, 
                                  lam0=lam0, lam1=lam1,
                                  p0=p0, p1=p1, rho=rho,
                                  m0=m_null, m1=m_alt,
                                  su=stepup, cutstyle=cut_type_general,
                                  horizon=n_periods_rejective)
    if split_corr:
        split_corr_text = (
                  " <br/>&nbsp;&nbsp;&nbsp;Split corr &rho;<sub>1</sub> = {rho1}".format(rho1=rho1))
        header_text = header_text + split_corr_text
    elif rand_order:
        rand_order_text = " <br/>&nbsp;&nbsp;&nbsp;Random Ordering for Corr mat"
        header_text = header_text + rand_order_text
    
    plot_list0 = []
    plot_list1 = []
    plot_list = [[Div(render_as_text=False, text=header_text)], 
                  plot_list0, 
                  plot_list1] 
                      
#                      height=8,
#                            tags=[{"style":"font-size:200%;"}]
    
    
        
    if (not cmd_args.skipsynth) or plot_synth:
        try:
            synth_truth_mask = synth_binom_data_general[2]
        except:
            print("Loading truth mask failed")
            synth_truth_mask = pandas.Series(numpy.repeat([True, False], m_null), index=synth_binom_data_general[0].columns)
            
        color_map = pandas.Series({True:"blue", False:"red"})
        cmapf = lambda u: color_map[u]
        
        synth_binom_fdp_rec = pandas.concat(
               (compute_fdr(synth_binom_data_general[0],
                             synth_binom_data_general[2]).add_prefix("gen_"), 
                 compute_fdr(synth_binom_data_rejective[0],
                             synth_binom_data_rejective[2]).add_prefix("rej_")))
#        pandas.DataFrame(dict(
#        gen_fdp = ((synth_truth_mask * (synth_binom_data_general[0].isin(["rej"]))).sum(1) / (synth_binom_data_general[0].isin(["rej"]).sum(1))).fillna(0),
#        gen_fnp = ((~synth_truth_mask * (synth_binom_data_general[0].isin(["acc"]))).sum(1) / (synth_binom_data_general[0].isin(["acc"]).sum(1))).fillna(0),
#        rej_fdp = ((synth_truth_mask * (synth_binom_data_rejective[0].isin(["rej"]))).sum(1) / (synth_binom_data_rejective[0].isin(["rej"]).sum(1))).fillna(0),
#        rej_fnp = ((~synth_truth_mask * (synth_binom_data_rejective[0].isin(["acc"]))).sum(1) / (synth_binom_data_rejective[0].isin(["acc"]).sum(1))).fillna(0)))
        print("Binom {0}".format(synth_binom_fdp_rec.mean()))
        FDR_table = pandas.DataFrame({"binom":synth_binom_fdp_rec})
        
        synth_outcomes = visualizations.build_comparison_plot_dataframe(synth_binom_data_general[0], 
                                                        synth_binom_data_rejective[0],
                                                        synth_truth_mask, cmapf)
                                                        
        if not cmd_args.scomponly:
            plot_list0.append(visualizations.bokeh_tool_scatter(source_rec=synth_outcomes, x_col="rej_prop_fh", 
                                                               y_col="rej_prop_gen", 
                          tooltips=[("(FH Rejection Rate, General rejection rate)", "(@rej_prop_fh{1.11111}, @rej_prop_gen{1.11111})"), ("H0", "@grund_truth")],
                          title="Binomial rejection rate comparison", color_col="color",
                          x_axis_label="FH", y_axis_label="General",
                          x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), glyph_size=glyph_size))
        #### Normal Poisson
        
        synth_pois_fdp_rec = pandas.concat(
               (compute_fdr(synth_pois_data_general[0],
                             synth_pois_data_general[2]).add_prefix("gen_"), 
                 compute_fdr(synth_pois_data_rejective[0],
                             synth_pois_data_rejective[2]).add_prefix("rej_")))
#        pandas.DataFrame(dict(
#        gen_fdp = ((synth_truth_mask * (synth_pois_data_general[0].isin(["rej"]))).sum(1) / (synth_pois_data_general[0].isin(["rej"]).sum(1))).fillna(0),
#        gen_fnp = ((~synth_truth_mask * (synth_pois_data_general[0].isin(["acc"]))).sum(1) / (synth_pois_data_general[0].isin(["acc"]).sum(1))).fillna(0),
#        rej_fdp = ((synth_truth_mask * (synth_pois_data_rejective[0].isin(["rej"]))).sum(1) / (synth_pois_data_rejective[0].isin(["rej"]).sum(1))).fillna(0),
#        rej_fnp = ((~synth_truth_mask * (synth_pois_data_rejective[0].isin(["acc"]))).sum(1) / (synth_pois_data_rejective[0].isin(["acc"]).sum(1))).fillna(0)))
        print("Pois {0}".format(synth_pois_fdp_rec.mean()))
        FDR_table["pois"] = synth_pois_fdp_rec
        
        
        synth_outcomes = visualizations.build_comparison_plot_dataframe(synth_pois_data_general[0], 
                                                        synth_pois_data_rejective[0],
                                                        synth_truth_mask, cmapf)
        
        plot_list1.append(visualizations.bokeh_tool_scatter(source_rec=synth_outcomes, x_col="rej_prop_fh", 
                                                           y_col="rej_prop_gen", 
                      tooltips=[("(FH Rejection Rate, General rejection rate)", "(@rej_prop_fh{1.11111}, @rej_prop_gen{1.11111})"), ("H0", "@grund_truth")],
                      title="Poisson rejection rate comparison "+ str((m_null, m_alt)), color_col="color",
                      x_axis_label="FH", y_axis_label="General",
                      x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), glyph_size=glyph_size,
                      legend="FDR controlled"))
        
        #### Unscaled Poisson 
        
        synth_pois_unscaled_fdp_rec = pandas.concat((compute_fdr(synth_pois_unscaled_data_general[0],
                                                                 synth_pois_unscaled_data_general[2]).add_prefix("gen_"), 
                                                     compute_fdr(synth_pois_unscaled_data_rejective[0],
                                                                 synth_pois_unscaled_data_rejective[2]).add_prefix("rej_")))
#        pandas.DataFrame(dict(
#        gen_fdp = ((synth_truth_mask * (synth_pois_unscaled_data_general[0].isin(["rej"]))).sum(1) / (synth_pois_unscaled_data_general[0].isin(["rej"]).sum(1))).fillna(0),
#        gen_fnp = ((~synth_truth_mask * (synth_pois_unscaled_data_general[0].isin(["acc"]))).sum(1) / (synth_pois_unscaled_data_general[0].isin(["acc"]).sum(1))).fillna(0),
#        rej_fdp = ((synth_truth_mask * (synth_pois_unscaled_data_rejective[0].isin(["rej"]))).sum(1) / (synth_pois_unscaled_data_rejective[0].isin(["rej"]).sum(1))).fillna(0),
#        rej_fnp = ((~synth_truth_mask * (synth_pois_unscaled_data_rejective[0].isin(["acc"]))).sum(1) / (synth_pois_unscaled_data_rejective[0].isin(["acc"]).sum(1))).fillna(0)))
        print("Unscaled Pois {0}".format(synth_pois_unscaled_fdp_rec.mean()))
        FDR_table["pois_unscaled"] = synth_pois_unscaled_fdp_rec
        
        synth_outcomes = visualizations.build_comparison_plot_dataframe(synth_pois_unscaled_data_general[0], 
                                                        synth_pois_unscaled_data_rejective[0],
                                                        synth_truth_mask, cmapf)

        
        if not cmd_args.scomponly:
            
            plot_list1.append(visualizations.bokeh_tool_scatter(source_rec=synth_outcomes, x_col="rej_prop_fh", 
                                                               y_col="rej_prop_gen", 
                          tooltips=[("(FH Rejection Rate, General rejection rate)", "(@rej_prop_fh{1.11111}, @rej_prop_gen{1.11111})"), ("H0", "@grund_truth")],
                          title="Unscaled Poisson rejection rate comparison ", color_col="color",
                          x_axis_label="FH", y_axis_label="General",
                          x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), glyph_size=glyph_size))
        
        visualizations.bokeh_add_scatter(plot_list1[0], 
                                         source_rec=synth_outcomes, 
                                         x_col="rej_prop_fh", 
                                         y_col="rej_prop_gen", 
                                         color_col="color",
                                         legend="Unscaled", glyph_size=glyph_size)
        
        #### pois grad
        
        synth_outcomes = visualizations.build_comparison_plot_dataframe(synth_pois_grad_data_general[0], 
                                                        synth_pois_grad_data_rejective[0],
                                                        synth_truth_mask, cmapf)
        
    
        dar_pois_grad = data_funcs.assemble_fake_pois_grad(m_null, lam0, lam1,m_alt=m_alt)
        synth_outcomes["pois_rate"] = dar_pois_grad


        grad_color_func = functools.partial(visualizations.grad_color_func, lam0=lam0, lam1=lam1)
            
        synth_outcomes["color"] = synth_outcomes["pois_rate"].apply(grad_color_func)
        synth_outcomes["drug_name"] = synth_outcomes.index
        if not cmd_args.scomponly:
            
            plot_list0.append(visualizations.bokeh_tool_scatter(source_rec=synth_outcomes, x_col="rej_prop_fh", 
                                                               y_col="rej_prop_gen", 
                          tooltips=[("(FH Rejection Rate, General rejection rate)", "(@rej_prop_fh{1.11111}, @rej_prop_fh{1.11111})"), ("pois rate", "@pois_rate")],
                          title="Poisson gradient rejection rate comparison", # \n(H0:lam={0}, Ha:lam={1})".format(lam0, lam1), 
                          x_axis_label="FH", y_axis_label="General", line_alpha_override=.5,
                          x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), color_col="color", glyph_size=glyph_size))
        
    
    with closing(shelve.open(rec_filepath)) as shf:
        shf["FDR_table"] = FDR_table
    FDR_table["pois_eff_ratio"] = FDR_table["pois"] / pandas.Series({"gen_FDR":alpha_general,
                                                                     "gen_FNR":beta_general,
                                                                     "rej_FDR":alpha_rejective,
                                                                     "rej_FNR":np.NaN})
    FDR_table["binom_eff_ratio"] = FDR_table["binom"] / pandas.Series({"gen_FDR":alpha_general,
                                                                       "gen_FNR":beta_general,
                                                                       "rej_FDR":alpha_rejective,
                                                                       "rej_FNR":np.NaN})
    processed_FDR_table = FDR_table.round(4)
    processed_FDR_table["index"] =processed_FDR_table.index
    fdr_source = ColumnDataSource(processed_FDR_table)
    
    columns = [
            TableColumn(field=col_name, title=col_name) for col_name in FDR_table.columns
        ] + [TableColumn(field="index", title="Metric", formatter=StringFormatter())]
    data_table = DataTable(source=fdr_source, columns=columns, width=900, height=280) # 

    plot_list.append([widgetbox(data_table)])
    
    if not cmd_args.scomponly:            
        full_page = gridplot(plot_list)
    else:
        flat_plot_list = [item for sublist in plot_list for item in sublist]
        full_page = column(*flat_plot_list)
        
    if not cmd_args.stifle:
        bokeh_show(full_page )
    else:
        bokeh_save(full_page, os.path.expanduser(cmd_args.vizpath), title="Drug Plots")
        
#    from bokeh.io import export_png
#    relevant_plot = plot_list1[0]
#    print(relevant_plot)
#    
#    export_png(relevant_plot, filename=os.path.expanduser("~/Documents/plot1.png"))
    
    
#    bokeh_show(vplot(*plot_list))
    
    
#synth_fdp_rec.plot("rej_fdp", "rej_fnp", "scatter")
#clf()
#synth_gen_fdp.hist(alpha=.3, bins=linspace(0, 1, 100), normed=True, label="gen fdp")
#synth_rej_fdp.hist(alpha=.3, bins=linspace(0, 1, 100), normed=True, label="rej fdp")
#synth_gen_fnp.hist(alpha=.3, bins=linspace(0, 1, 100), normed=True, label="gen fnp")
#synth_rej_fnp.hist(alpha=.3, bins=linspace(0, 1, 100), normed=True, label="rej fnp") 
#legend()