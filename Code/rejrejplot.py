#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 11:23:32 2017

@author: mhankin
"""
import argparse
# Parallelization
parser = argparse.ArgumentParser()
parser.add_argument('--cfgpath', default="~/Dropbox/Research/MultSeq/Data/sim.cfg", help="Simulation configuration file")
parser.add_argument('--cfgpath1')
parser.add_argument('--cfgsect0', help="Simulation configuration section")
parser.add_argument('--cfgsect1', help="Simulation configuration section")
parser.add_argument('--name0', default=None, help="Simulation configuration section")
parser.add_argument('--name1', default=None, help="Simulation configuration section")

parser.add_argument('--drug', action='store_true', help="Use default roots with cfgsect name")
parser.add_argument('--logit', action='store_true', help="plot logit scale")




cmd_args = parser.parse_args()

cfgsect0 = cmd_args.cfgsect0
cfgsect1 = cmd_args.cfgsect1
if cmd_args.name0 is None:
    name0 = cfgsect0
else:
    name0 = cmd_args.name0

if cmd_args.name1 is None:
    name1 = cfgsect1
else:
    name1 = cmd_args.name1
    
if cmd_args.cfgpath1 is None:
    cmd_args.cfgpath1 = cmd_args.cfgpath

vizpath = "~/Dropbox/Research/MultSeq/Data/compare-{0}-{1}.html".format(cmd_args.cfgsect0, cmd_args.cfgsect1)
shelvepath0 = "~/Dropbox/Research/MultSeq/Data/{0}.shelve".format(cfgsect0)
shelvepath1 = "~/Dropbox/Research/MultSeq/Data/{0}.shelve".format(cfgsect1)



from utils.common_funcs import sigmoid, log_odds
import shelve, visualizations, os, configparser, pandas
import functools
from utils import data_funcs
from contextlib import closing
from bokeh.plotting import output_file
from bokeh.models import Range1d, ColumnDataSource
from bokeh.layouts import gridplot, widgetbox, column
from bokeh.models.widgets import DataTable, StringFormatter, TableColumn, Paragraph, Panel, Div
from bokeh.io import show as bokeh_show
from bokeh.io import save as bokeh_save
import pickle
from numpy import array
import numpy as np


from bokeh.models import TickFormatter

JS_CODE = """
import {TickFormatter} from "models/formatters/tick_formatter"

export class MyFormatter extends TickFormatter
  type: "MyFormatter"

  # TickFormatters should implement this method, which accepts a lisst
  # of numbers (ticks) and returns a list of strings
  doFormat: (ticks) ->
    # format the first tick as-is
    formatted = ["p=#{(1/(1+Math.exp(-ticks[0]))).toExponential(2)}"]

    # format the remaining ticks as a difference from the first
    for i in [1...ticks.length]
       formatted.push("p=#{(1/(1+Math.exp(-ticks[i]))).toExponential(2)}")

    return formatted
"""

class MyFormatter(TickFormatter):

    __implementation__ = JS_CODE






if True or cmd_args.doviz:
    output_file(os.path.expanduser(vizpath), mode="inline") # autosave=True,

drug = cmd_args.drug
rec_filepath0 = os.path.expanduser(shelvepath0)
rec_filepath1 = os.path.expanduser(shelvepath1)


configfl0 = configparser.ConfigParser(inline_comment_prefixes=["#"], default_section="default")
configfl0.read([os.path.expanduser(cmd_args.cfgpath)])
config0 = configfl0[cfgsect0]
configfl1 = configparser.ConfigParser(inline_comment_prefixes=["#"], default_section="default")
configfl1.read([os.path.expanduser(cmd_args.cfgpath1)])
config1 = configfl1[cfgsect1]

if drug:
    min_am = config0.getint("min_am")
    min_tot = config0.getint("min_tot")
    
    am_prop_pctl = (config0.getfloat("am_prop_pctl_low"),
                    config0.getfloat("am_prop_pctl_high"))

glyph_size = config0.getint("glyph_size")
m_null = config0.getint("m_null")
m_alt = config0.getint("m_alt")
assert m_null == config1.getint("m_null"), "Wrong nulls"
assert m_alt == config1.getint("m_alt"), "wrong alts"

# Check they match and build synth_truth mask

with closing(shelve.open(rec_filepath0)) as shf:
    print(list(shf.keys()))
    if drug:
        raw_rec0 = shf["rejacc_drug_rejective"][0]
        rec0 = pandas.Series(raw_rec0.applymap(lambda u: float(u=="rej")).mean())
    else:
        rec0 = shf["synth_pois_data_rejective"][0]
        rec0_unscaled = shf["synth_pois_unscaled_data_rejective"][0]
        synth_truth_mask = shf["synth_pois_data_general"][2]
    
with closing(shelve.open(rec_filepath1)) as shf:
    if drug:
        raw_rec1 = shf["rejacc_drug_rejective"][0]
        rec1 = pandas.Series(raw_rec1.applymap(lambda u: float(u=="rej")).mean())
    else:
        rec1 = shf["synth_pois_data_rejective"][0]
        rec1_unscaled = shf["synth_pois_unscaled_data_rejective"][0]
    
    
if drug:
    clean_name0 = name0.replace(" ", "_")
    clean_name1 = name1.replace(" ", "_")
    drug_outcomes = pandas.DataFrame({clean_name0:rec0, clean_name1:rec1})
    dar, dnar, _ = data_funcs.read_drug_data(data_funcs.gen_skew_prescreen(min_am, min_tot))
    p0, p1 = data_funcs.am_prop_percentile_p0_p1(dar, dnar, *am_prop_pctl)
    grad_color_func = functools.partial(visualizations.grad_color_func, lam0=p0, lam1=p1)
    maindf = pandas.DataFrame({'drug_name':dar.index,"amnesia_rate":dar.values, "other_rate":dnar.values})
    maindf["p"] = maindf["amnesia_rate"] / (maindf["amnesia_rate"] + maindf["other_rate"])
    maindf["color"] = maindf["p"].apply(grad_color_func)
    maindf.set_index("drug_name", inplace=True, drop=False)
    drug_outcomes = pandas.merge(drug_outcomes, maindf, 
                                 left_index=True, right_index=True)
#    drug_outcomes["{0} - {1}".format(name0, name1)] = drug_outcomes[name0] - drug_outcomes[name1]

    print(drug_outcomes.head())
#    raise Exception()
    
    p0 = visualizations.bokeh_tool_scatter(source_rec=drug_outcomes, x_col=clean_name0, 
                                                               y_col=clean_name1, 
                          tooltips=[("({0} Rejection Rate, {1} Rejection rate)".format(name0, name1), 
                                     "(@"+clean_name0+"{1.11111}, @"+clean_name1+"{1.11111})"), 
        ("H0", "@color")],
                          title="Drug rejection rate comparison", color_col="color",
                          x_axis_label=name0, y_axis_label=name1,
                          x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), glyph_size=glyph_size)
    p = p0
#    p1 = visualizations.bokeh_tool_scatter(source_rec=drug_outcomes, x_col=name0, 
#                                                               y_col=name0, 
#                          tooltips=[("({0} Rejection Rate, {1} Rejection rate)".format(name0, name1), 
#                                     "(@"+name0+"{1.11111}, @"+name1+"{1.11111})"), 
#        ("H0", "@color")],
#                          title="Drug rejection rate comparison", color_col="color",
#                          x_axis_label=name0, y_axis_label=name1,
#                          x_range=Range1d(-0.1, 1.1), y_range=Range1d(-0.1, 1.1), glyph_size=glyph_size)
else:
    color_map = pandas.Series({True:"blue", False:"red"})
    cmapf = lambda u: color_map[u]
        

    synth_outcomes = visualizations.build_comparison_plot_dataframe(
            rec0, rec1, synth_truth_mask, cmapf, gen_suffix="_"+cfgsect0, rej_suffix="_"+cfgsect1)
            
    synth_outcomes_unscaled = visualizations.build_comparison_plot_dataframe(
            rec0_unscaled, rec1_unscaled, synth_truth_mask, cmapf, gen_suffix="_"+cfgsect0, rej_suffix="_"+cfgsect1)
    
    if cmd_args.logit:
        prefix = "rej_logit_"
        synth_outcomes[prefix + cfgsect0] = log_odds(synth_outcomes["rej_prop_" + cfgsect0])
        synth_outcomes[prefix + cfgsect1] = log_odds(synth_outcomes["rej_prop_" + cfgsect1])
        synth_outcomes_unscaled[prefix + cfgsect0] = log_odds(synth_outcomes_unscaled["rej_prop_" + cfgsect0])
        synth_outcomes_unscaled[prefix + cfgsect1] = log_odds(synth_outcomes_unscaled["rej_prop_" + cfgsect1])
        synth_outcomes.replace([np.inf, -np.inf], np.nan, inplace=True)
        synth_outcomes_unscaled.replace([np.inf, -np.inf], np.nan, inplace=True)
        xlims_raw_scaled = (synth_outcomes[prefix + cfgsect0].min(), synth_outcomes[prefix + cfgsect0].max())
        ylims_raw_scaled = (synth_outcomes[prefix + cfgsect1].min(), synth_outcomes[prefix + cfgsect1].max())
        xlims_raw_unscaled = (synth_outcomes_unscaled[prefix + cfgsect0].min(), synth_outcomes_unscaled[prefix + cfgsect0].max())
        ylims_raw_unscaled = (synth_outcomes_unscaled[prefix + cfgsect1].min(), synth_outcomes_unscaled[prefix + cfgsect1].max())
        xlims_raw = (min(xlims_raw_scaled[0], xlims_raw_unscaled[0]), max(xlims_raw_scaled[1], xlims_raw_unscaled[1]))
        ylims_raw = (min(ylims_raw_scaled[0], ylims_raw_unscaled[0]), max(ylims_raw_scaled[1], ylims_raw_unscaled[1]))
        xlims = array(xlims_raw) + array([-0.1, 0.1]) * (xlims_raw[1] - xlims_raw[0])
        ylims = array(ylims_raw) + array([-0.1, 0.1]) * (ylims_raw[1] - ylims_raw[0])
        
    else:
        prefix = "rej_prop_"
        xlims = (-0.1, 1.1)
        ylims = (-0.1, 1.1)
        
    print(synth_outcomes.head())
        
    p = visualizations.bokeh_tool_scatter(source_rec=synth_outcomes, x_col=prefix + cfgsect0, 
                                                               y_col=prefix + cfgsect1, 
                          tooltips=[("({0} Rejection Rate, {1} rejection rate)".format(name0, name1), 
                                     "(@rej_prop_"+cfgsect0+"{1.11111}, @rej_prop_"+cfgsect1+"{1.11111})"), 
        ("H0", "@grund_truth")],
                          title="Poisson rejection rate comparison "+ str((m_null, m_alt)), color_col="color",
                          x_axis_label=name0, y_axis_label=name1,
                          x_range=Range1d(*xlims), y_range=Range1d(*ylims), glyph_size=glyph_size,
                          legend="FDR controlled")
    
    
    p = visualizations.bokeh_add_scatter(p, 
                                             source_rec=synth_outcomes_unscaled.replace([np.inf, -np.inf], np.nan), 
                                             x_col=prefix + cfgsect0, 
                                             y_col=prefix + cfgsect1,
                                             color_col="color",
                                             legend="Unscaled", glyph_size=glyph_size)
    if cmd_args.logit:
        p.xaxis.formatter = MyFormatter()
        p.yaxis.formatter = MyFormatter()

bokeh_show(p)