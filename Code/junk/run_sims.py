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
parser = argparse.ArgumentParser(description='Run simulations')
parser.add_argument('--singlecore', action='store_true', help="Dont parallelize main sims")
parser.add_argument('--skipdrugs', action='store_true', help="skip drugs sims and load data")
#parser.add_argument('--skipsynth', action='store_true', help="skip synthetic sims and load data")
parser.add_argument('--skiprej', action='store_true', help="skip rejective procedures")
parser.add_argument('--skipgen', action='store_true', help="skip general procedures")
parser.add_argument('--doviz', action='store_true', help="do visualizations")
parser.add_argument('--stifle', action='store_true', help="don't actually show viz")
parser.add_argument('--shelvepath', default="~/Dropbox/Research/MultSeq/Data/SimRec.shelve", help="shelve record path")
parser.add_argument('--vizpath', default="~/Dropbox/Research/MultSeq/Data/SimRec.html", help="Visualization html path")
parser.add_argument('--cfgpath', default="~/Dropbox/Research/MultSeq/Data/sim.cfg", help="Simulation configuration file")
parser.add_argument('--cfgsect', default="main", help="Simulation configuration section")

parser.add_argument('--usesect', action='store_true', help="Use default roots with cfgsect name")
argcomplete.autocomplete(parser)
cmd_args = parser.parse_args()


if cmd_args.usesect:
    cmd_args.vizpath = "~/Dropbox/Research/MultSeq/Data/{0}.html".format(cmd_args.cfgsect)
    cmd_args.shelvepath = "~/Dropbox/Research/MultSeq/Data/{0}.shelve".format(cmd_args.cfgsect)


# %% Main imports
import numpy as np
    
import visualizations
from utils import data_funcs,  simulation_funcs

from IPython.display import display
import pandas
import shelve
from contextlib import closing
from configparser import ConfigParser

import logging
import tqdm
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# %% Reload after code changes

class TqdmLoggingHandler (logging.Handler):
    """Logging handler that emit messages friendly to tqdm progress bars"""
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
print("Section {0}".format(cmd_args.cfgsect))
print(dict(config_section))
# Drug screening 
# config_section.getint("")
min_am = config_section.getint("min_am")
min_tot = config_section.getint("min_tot")

am_prop_pctl = (config_section.getfloat("am_prop_pctl_low"),
                config_section.getfloat("am_prop_pctl_high"))

with closing(shelve.open(rec_filepath)) as shf:
    shf["min_am"] = min_am
    shf["min_tot"] = min_tot

plot_drugs = config_section.getboolean("plot_drugs")
print("plot_drugs {0}".format(plot_drugs))

cut_type = config_section.get("cut_type")
if "scale_fdr" in config_section:
    scale_fdr = config_section.getboolean("scale_fdr")
else:
    scale_fdr = True
    
if "cummax" in config_section:
    cummax = config_section.getboolean("cummax")
else:
    cummax = False


stepup = config_section.getboolean("stepup")
alpha = config_section.getfloat("alpha")
sim_reps = config_section.getint("sim_reps")


if "divide_cores" in config_section:
    divide_cores = config_section.getfloat("divide_cores")
else:
    divide_cores = None
# Drug Rejective
undershoot_drug_rejective =  config_section.getfloat("undershoot_drug_rejective")
n_periods_rejective =  config_section.getint("n_periods_rejective")

# Drug General

beta = config_section.getfloat("beta")
undershoot_drug_general = config_section.getfloat("undershoot_drug_general")





# %% Real data rejective

if cmd_args.skipdrugs or cmd_args.skiprej:
    try:
        with closing(shelve.open(rec_filepath)) as shf:
            rejacc_drug_rejective = shf["rejacc_drug_rejective"]
    except:
        logging.warn("Unable to load drug rejective data")
else:
    rejacc_drug_rejective = simulation_funcs.real_data_wrapper(
        alpha, None, undershoot_prob=undershoot_drug_rejective, 
        sim_reps=sim_reps, n_periods=n_periods_rejective, 
        min_am=min_am, min_tot=min_tot, rho=None,  cut_type=cut_type,
        do_parallel=not cmd_args.singlecore, fin_par=not cmd_args.singlecore, 
        am_prop_pctl=am_prop_pctl, stepup=stepup, scale_fdr=scale_fdr, 
        divide_cores=divide_cores, cummax=cummax)
        
    with closing(shelve.open(rec_filepath)) as shf:
        shf["alpha"] = alpha
        shf["undershoot_drug_rejective"] = undershoot_drug_rejective
        shf["n_periods_rejective"] = n_periods_rejective
        shf["sim_reps"] = sim_reps
        shf["rejacc_drug_rejective"] = rejacc_drug_rejective

    
# %% Real data rejective


if cmd_args.skipdrugs or cmd_args.skipgen:
    try:
        with closing(shelve.open(rec_filepath)) as shf:
            rejacc_drug_general = shf["rejacc_drug_general"]
    except:
        logging.warn("Unable to load infinite horizon drug data")     
else:
    rejacc_drug_general = simulation_funcs.real_data_wrapper(
        alpha, beta, rho=None, cut_type=cut_type,
        undershoot_prob=undershoot_drug_general, 
        sim_reps=sim_reps, min_am=min_am, min_tot=min_tot,
        do_parallel=not cmd_args.singlecore, 
        fin_par=not cmd_args.singlecore,
        am_prop_pctl=am_prop_pctl, stepup=stepup, scale_fdr=scale_fdr, 
        divide_cores=divide_cores)
        
    with closing(shelve.open(rec_filepath)) as shf:
        shf["alpha"] = alpha
        shf["beta"] = beta
        shf["undershoot_drug_general"] = undershoot_drug_general
        shf["sim_reps"] = sim_reps
        shf["rejacc_drug_general"] = rejacc_drug_general


# %% visualizations

from bokeh.plotting import output_file
from bokeh.models import Range1d

from bokeh.models.widgets import DataTable, StringFormatter, TableColumn, Paragraph, Panel, Div
from bokeh.layouts import gridplot, widgetbox
from bokeh.io import show as bokeh_show
from bokeh.io import save as bokeh_save
import pickle
import functools
if cmd_args.doviz:
    output_file(os.path.expanduser(cmd_args.vizpath)) #, autosave=True)
    
    plot_list = [[], [], []]
    print("doviz {0}".format(plot_drugs))
    if (not cmd_args.skipdrugs) or plot_drugs:
        
        # %% Raw rata data plot     
        dar, dnar, _ = data_funcs.read_drug_data(data_funcs.gen_skew_prescreen(min_am, min_tot))
        p0, p1 = data_funcs.am_prop_percentile_p0_p1(dar, dnar, *am_prop_pctl)
        grad_color_func = functools.partial(visualizations.grad_color_func, lam0=p0, lam1=p1)
        maindf = pandas.DataFrame({'drug_name':dar.index,"amnesia_rate":dar.values, "other_rate":dnar.values})
        maindf["p"] = maindf["amnesia_rate"] / (maindf["amnesia_rate"] + maindf["other_rate"])
        maindf["color"] = maindf["p"].apply(grad_color_func)
    
        
        # %% Load search data
        main_rec = pandas.read_csv("../Data/GoogleSearchHitData.csv", index_col=0)
        
        
        # Real data rejection rate vs google rate
        main_rec["general_rejection_rate"] = rejacc_drug_general[0].applymap(lambda u: float(u=="rej")).mean()
        main_rec["general_acceptance_rate"] = rejacc_drug_general[0].applymap(lambda u: float(u=="acc")).mean()
        main_rec["general_nonacceptance_rate"] = 1.0 - main_rec["general_acceptance_rate"]
        main_rec["rej_rejection_rate"] = rejacc_drug_rejective[0].applymap(lambda u: float(u=="rej")).mean()
        main_rec["rej_acceptance_rate"] = rejacc_drug_rejective[0].applymap(lambda u: float(u=="acc")).mean()
        main_rec['general_nonterm'] = 1.0 - (main_rec['general_acceptance_rate'] + main_rec["general_rejection_rate"])
        main_rec["drug_name"] = main_rec.index
        maindf_copy = maindf.copy().set_index("drug_name")
        main_rec.dropna(inplace=True)
        main_rec = main_rec.join(maindf_copy, "drug_name", "left", rsuffix="_raw")
        
        xmin_p1_raw, xmax_p1_raw = (maindf["amnesia_rate"].min(), maindf["amnesia_rate"].max())
        ymin_p1_raw, ymax_p1_raw = (maindf["other_rate"].min(), maindf["other_rate"].max())
        xmin1, xmax1 = (xmin_p1_raw - 0.05 * (xmax_p1_raw - xmin_p1_raw), xmax_p1_raw + 0.05 * (xmax_p1_raw - xmin_p1_raw))
        ymin1, ymax1 = (ymin_p1_raw - 0.05 * (ymax_p1_raw - ymin_p1_raw), ymax_p1_raw + 0.05 * (ymax_p1_raw - ymin_p1_raw))
        plot_list[0].append(visualizations.bokeh_tool_scatter(source_rec=maindf, x_col="amnesia_rate", y_col="other_rate", 
                      tooltips=[("Yellowcard  (Amnesia Rate, Non Amnesia Rate)", "($x{1.11111}, $y{1.11111})"), ("p", "@p")],
                      title="Yellowcard Side Effect Rates", 
                      x_axis_label="Amensia Rate", y_axis_label="Non Amnesia Rate",
                      x_range=Range1d(xmin1, xmax1), y_range=Range1d(ymin1, ymax1), color_col="color"))
        
    
        plot_list[0].append(visualizations.bokeh_tool_scatter(source_rec=main_rec, x_col="ratio", y_col="general_rejection_rate", 
                      tooltips=[("(Google Hit Amnesia Ratio, Rejection Rate)", "($x{1.11111}, $y{1.11111})"),],
                      title="Drug Rejection Rate (general) vs Google Amnesia Hit Ratio", 
                      x_axis_label="Google", y_axis_label="MultSPRT Rejection Rate",
                      x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), color_col="color"))
                      
        plot_list[1].append(visualizations.bokeh_tool_scatter(source_rec=main_rec, x_col="ratio", y_col="rej_rejection_rate", 
                      tooltips=[("(Google Hit Amnesia Ratio, Rejection Rate)", "($x{1.11111}, $y{1.11111})"),],
                      title="Drug Rejection Rate (FH) vs Google Amnesia Hit Ratio", 
                      x_axis_label="Google", y_axis_label="MultSPRT FH Rejection Rate",
                      x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), color_col="color")    )
        
        plot_list[1].append(visualizations.bokeh_tool_scatter(source_rec=main_rec, x_col="amnesia", y_col="general_rejection_rate", 
                      tooltips=[("(Google Amnesia Hits, Rejection Rate)", "($x{1.11111}, $y{1.11111})"),],
                      title="Drug Rejection Rate (general) vs Google Amnesia Hits", 
                      x_axis_label="Google", y_axis_label="MultSPRT Rejection Rate",
                      x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), color_col="color"))
                      
        plot_list[2].append(visualizations.bokeh_tool_scatter(source_rec=main_rec, x_col="rej_rejection_rate", y_col="general_rejection_rate", 
                      tooltips=[("(FH Rejection Rate, General Rejection Rate)", "($x{1.11111}, $y{1.11111})"),],
                      title="Drug Rejection Rate comparison", 
                      x_axis_label="FH", y_axis_label="General",
                      x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), color_col="color"))
        
        plot_list[2].append(Div(render_as_text=False, text=
                  ("<h1>Drug Rejection plots for:</h1>"
                  " <br/>&nbsp;&nbsp;&nbsp;&alpha;={alpha}&nbsp;&nbsp;&beta; = {beta}"
                  " <br/>&nbsp;&nbsp;&nbsp;min am={min_am}&nbsp;&nbsp;min total={min_tot}"
                  "&nbsp;&nbsp;drugs screened={num_drugs}"
                  " <br/>&nbsp;&nbsp;&nbsp;am prop for hyps={am_prop_pctl}"
                  " <br/>&nbsp;&nbsp;&nbsp;Step-up = {su}&nbsp;&nbsp;cut style = {cutstyle}"
                  " <br/>&nbsp;&nbsp;&nbsp;Finite Horizon = {horizon} &nbsp;&nbsp;&nbsp;sim reps= {sim_reps}"
                  "").format(alpha=alpha, beta=beta, 
                                  min_am=min_am, min_tot=min_tot,
                                  am_prop_pctl=am_prop_pctl,
                                  su=stepup, cutstyle=cut_type,
                                  horizon=n_periods_rejective,
                                  sim_reps=sim_reps, num_drugs=len(dar)),
                            ))
                      
    full_page = gridplot(plot_list)
    if not cmd_args.stifle:
        bokeh_show(full_page )
    else:
        bokeh_save(full_page, os.path.expanduser(cmd_args.vizpath), title="Drug Plots")
    
