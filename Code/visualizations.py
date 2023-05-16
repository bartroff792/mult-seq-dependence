# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 23:08:56 2016

@author: mike
"""
from pylab import *
from nose.tools import *
import numpy
import pandas
import string, re
import pickle, os
import datetime, calendar
import seaborn as sns
from bokeh.plotting import figure as bokeh_figure
from bokeh.plotting import ColumnDataSource
from bokeh.models import HoverTool
import pickle

LINE_ALPHA = 0.1
FILL_ALPHA = 0.2
CENTROID_ALPHA = 0.5
LEGEND_BACKGROUND_ALPHA = .75

def bokeh_tool_scatter(source_rec, x_col, y_col, tooltips, title, x_axis_label, 
                       y_axis_label, x_range, y_range, color_col=None, legend=None,
                       glyph_size=9, show_color_centroid=True, line_alpha_override=None):

    if line_alpha_override is not None:
        line_alpha = line_alpha_override
    else:
        line_alpha = LINE_ALPHA
    source = ColumnDataSource(
        data=source_rec,
    )

    hover_ratio = HoverTool(
            tooltips=[
                ("index", "$index"),
                ("hypothesis", "@drug_name"),
            ] + tooltips
        )
    
    p = bokeh_figure(plot_width=800, plot_height=600, tools=[hover_ratio, 'resize', 'box_zoom', 'reset', 'save', 'wheel_zoom'],
               title=title, 
               x_axis_label=x_axis_label, y_axis_label=y_axis_label,
               x_range=x_range, y_range=y_range)
    p.name = str(title)
    if color_col is None:
        p.circle(x_col, y_col, size=glyph_size, source=source, fill_alpha=FILL_ALPHA, 
                 line_alpha=line_alpha, legend=legend)
    else:
        p.circle(x_col, y_col, size=glyph_size, source=source, fill_alpha=FILL_ALPHA, 
                 line_alpha=line_alpha, fill_color=color_col, line_color=color_col, legend=legend)
        if show_color_centroid:
            centroid_source = ColumnDataSource(
                    data=source_rec.groupby(color_col).agg("mean").reset_index())
            p.circle(x_col, y_col, size=2*glyph_size, source=centroid_source, 
                     alpha=CENTROID_ALPHA, 
                     fill_color=color_col, line_color=color_col)
            
    
        
    
    if legend:
        p.legend.location = "bottom_right"
        p.legend.label_text_font_size = '18pt'
        p.legend.background_fill_alpha = LEGEND_BACKGROUND_ALPHA
    
    p.title.text_font_size = '25pt'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.xaxis.major_label_text_font_size = '18pt'
    p.yaxis.axis_label_text_font_size = '20pt'
    p.yaxis.major_label_text_font_size = '18pt'
    return p

def bokeh_add_scatter(p, source_rec, x_col, y_col, color_col=None, legend=None,
                      glyph_size=9, show_color_centroid=True):
    source = ColumnDataSource(
        data=source_rec,
    )
    if color_col is None:
        p.triangle(x_col, y_col, size=glyph_size, source=source, fill_alpha=FILL_ALPHA, 
                   line_alpha=LINE_ALPHA, legend=legend)
    else:
        p.triangle(x_col, y_col, size=glyph_size, source=source, fill_alpha=FILL_ALPHA, 
                   line_alpha=LINE_ALPHA, fill_color=color_col, line_color=color_col, legend=legend)
        
    if show_color_centroid:
        if legend:
            legend = legend + " centroid"
        centroid_source = ColumnDataSource(
                data=source_rec.groupby(color_col).agg("mean").reset_index())
        p.triangle(x_col, y_col, size=2*glyph_size, source=centroid_source, 
                 alpha=CENTROID_ALPHA, 
                 fill_color=color_col, line_color=color_col)
        
    return p
    
    
def visualize_raw_drug_data_distribution():
    # Examine relationship between report length and number of reactions
    # Appears to be a mixture

    scl = cl.scales['9']['seq']['Blues']
    colorscale = [ [ float(i)/float(len(scl)-1), scl[i] ] for i in range(len(scl)) ]
    colorscale

    df = pandas.DataFrame({'daterange': date_range_col, 'reacts': log10(reacts_df[total_reacts_cname])})
    df.head()

    data = [
        go.Histogram2dContour(
            x=df['daterange'], # assign x as the dataframe column 'x'
            y=df['reacts'],
            colorscale=colorscale,
            #line=py.Line(width=0)
        )
    ]

    axis_template = dict(
        ticks='',
        showgrid=False,
        zeroline=False,
        showline=True,
        mirror=True,
        linewidth=2,
        linecolor='#444',
    )

    xaxis = axis_template.copy()
    xaxis['title'] = 'Years'
    yaxis = axis_template.copy()
    yaxis['title'] = 'Log 10 Total Reactions'
    layout=go.Layout(xaxis=xaxis,
                     yaxis=yaxis,
                     width=700,
                     height=750,
                     autosize=False,
                     hovermode='closest',
                     title='Log Total Reactions vs Report Length')

    fig = go.Figure(data=data, layout=layout)

    # IPython notebook
    py.iplot(fig, filename='num_reacts_vs_report_length_full', height=750)
    
    
def plot_multseq_llr(llr, A_vec, B_vec, ground_truth=None, title=None, verbose=True,
                     stepup=False, label_fontsize=18, title_fontsize=24,
                     add_metrics=False, do_annotation=True, ghost_lines=True,
                     jitter_mag=.1, stat_data_func=None, skip_main_plot=False):
    """Plot paths and terminations.
    """
    # TODO(mhankin): Either allow passage or remove
    p0, p1, FDR_LEVEL, FNR_LEVEL = (.1, .1, .1, .1)
    
    # TODO(mhankin): remove all non plotting code, stick wrapper func in demo or something
    import multseq
    from utils.simulation_funcs import compute_fdp

    # Rejective only
    rejective = B_vec is None
        
    fake_drug_outcomes, _ = multseq.modular_sprt_test(llr, A_vec, B_vec, rejective=rejective, verbose=verbose, stepup=stepup)
    if stat_data_func is not None:
        llr = stat_data_func()
    fig = figure(figsize=(12,8))

    # DF with row for each drug, specifying its outcome (ar0),
    # the step at which it was accepted or rejected (NaN if neither),
    # and the significance level at which it was terminated.
    drugTerminationData = fake_drug_outcomes['drugTerminationData']
    n_periods = len(llr)
    print(n_periods)
    n_drugs = len(llr.columns)
    if ground_truth is not None:
        color_iter = [iter(sns.color_palette("bright", n_colors=n_drugs)),
                      iter(sns.color_palette("pastel", n_colors=n_drugs))]
    else:   
        color_iter = iter(sns.color_palette(n_colors=n_drugs))

    if rejective:
        best_bounds = [0, max(A_vec)]
    else:
        best_bounds = [min(B_vec), max(A_vec)]    
    
    
    for drug_name, drug_term_data in drugTerminationData.iterrows(): 
        end_step = drug_term_data['step']

        if isnan(end_step):
            end_step = n_periods
        else:
            end_step = int(end_step) + 1

        if not isnan(drug_term_data['rejLevel']):
            drug_label = "{0} (rej at {1})".format(drug_name, end_step )
        elif not isnan(drug_term_data['accLevel']):
            drug_label = "{0} (acc at {1})".format(drug_name, end_step)
        else:
            drug_label = "{0}".format(drug_name)
            
        if ground_truth is not None:
            drug_label += " actually $H_{{{0}}}$".format(int(~ground_truth[drug_name]))
            color = next(color_iter[int(~ground_truth[drug_name])])
        else:
            color = next(color_iter)
        raw_jitter = jitter_mag * randn(end_step)
        raw_jitter[0] = 0
        if not skip_main_plot:
            plot(llr[drug_name][:end_step] + raw_jitter.cumsum(), label=drug_label,
                 color=color)
    
        # Show post termination data
        if ghost_lines:
            plot(llr[drug_name][(end_step-1):], linestyle='dotted', alpha=.8,
                 color=color)
        y_final = llr[drug_name][end_step-1]
        shift_length = max([.5, n_periods/10])
        if do_annotation:
            annotate(drug_label, (end_step-1, y_final), 
                 xytext=(end_step + shift_length - 1, (.8 + .4 * rand()) * y_final), arrowprops=dict(facecolor='black', shrink=0.2,), 
                 fontsize=label_fontsize)
        # update guess at relevant y bounds
        if min(llr[drug_name][:end_step]) < best_bounds[0]:
            best_bounds[0] = min(llr[drug_name][:end_step])
        if max(llr[drug_name][:end_step]) > best_bounds[1]:
            best_bounds[1] = max(llr[drug_name][:end_step])
            
        
    if rejective:
        best_bounds[1] = 1.5 * best_bounds[1]
        best_bounds[0] = -best_bounds[1]
        
    for j, A_val in enumerate(A_vec):
        max_step = fake_drug_outcomes['levelTripData']['rej'][j]
        hlines(A_val, xmin=0, xmax=max_step, linestyles='dashed')
        annotate("A" + str(j+1), (max_step, A_val), fontsize=16)
        
    if not rejective:        
        for j, B_val in enumerate(B_vec):
            max_step = fake_drug_outcomes['levelTripData']['acc'][j]
            hlines(B_val, xmin=0, xmax=max_step, linestyles='dashed')
            annotate("B" + str(j+1), (max_step, B_val), fontsize=16)

    ylim(1.2 * best_bounds[0], 1.2 * best_bounds[1])            
    # legend(loc='upper center', bbox_to_anchor=(0.5, 1.25),
    #           ncol=4, fancybox=True, shadow=True)
    if ground_truth is not None and add_metrics:
        fdp_level, fnp_level, _, _ = compute_fdp(drugTerminationData, ground_truth)
        
        fig.text(.5, 0.025, 
             '$H_0: p={0}   H_a: p={1}$ \n pFDR$\leq${2}, pFNR$\leq${3}\nFDP={4}, FNP={5}'.format(p0, p1, FDR_LEVEL, FNR_LEVEL, fdp_level, fnp_level))
    elif title is not None:
        fig.suptitle(title, fontsize=title_fontsize)
    fig.axes[0].tick_params(labelsize=label_fontsize)
    return fig, drugTerminationData
    # fig.savefig('/media/mike/joint/Dropbox/Research/MultSeq/Images/SyntheticMultSeqpFDR.png')
    #cutoffs.T.to_csv('/media/mike/joint/Dropbox/Research/MultSeq/Data/SyntheticpFDRCutoffs.csv', float_format="%.4e")
    
def demo_paths(ugly=True, stepup=False, label_fontsize=18, title_fontsize=24):
    llr = pandas.DataFrame({'Hyp1':array([0.0, 2.5, 3.5, 17.0, -30, 12]), 
                    'Hyp2':array([0.0, 2.5, 1.5, -1.5, 1.5, 2.5]),
                    'Hyp3':array([0.0, 0.5, -0.5, -1.5, -2.5, -1.5]),
                    'Hyp4':array([0.0, 0.2, -0.2, 0.0, 0.2, -0.2])})
    if not ugly:
        llr["Hyp3"][5] = -4.0
        llr["Hyp2"][1] = .85
        llr["Hyp1"][3:6] = array([3.2, 2.2, 3.7])
    A_vec = array([2, 1, .5, .25])
    B_vec = -array([3.0, 2.0, 1.0, .5])
    return plot_multseq_llr(llr, A_vec, B_vec, title="Sequential Stepdown Toy Demo", stepup=stepup,
                            label_fontsize=label_fontsize, title_fontsize=title_fontsize)
    
    
def grad_color_func(poisrate, lam0, lam1):
    if poisrate < lam0:
        return "blue"
    elif poisrate > lam1:
        return "red"
    else:
        return "green"
    
    
    
def build_comparison_plot_dataframe(general_df, rejective_df, synth_truth_mask, cmapf,
                                    gen_suffix="_gen", rej_suffix="_fh"):
        synth_gen_outcomes = general_df.apply(lambda u: u.value_counts(), reduce=True).T.fillna(0)
        synth_rej_outcomes = rejective_df.apply(lambda u: u.value_counts(), reduce=True).T.fillna(0)
        if not "rej" in synth_gen_outcomes.columns:
            synth_gen_outcomes["rej"] = False
        if not "acc" in synth_gen_outcomes.columns:
            synth_gen_outcomes["acc"] = False    
        if not "rej" in synth_rej_outcomes.columns:
            synth_rej_outcomes["rej"] = False
        if not "acc" in synth_rej_outcomes.columns:
            synth_rej_outcomes["acc"] = False    
        synth_gen_outcomes["rej_prop"] = synth_gen_outcomes["rej"] / (synth_gen_outcomes["rej"] + synth_gen_outcomes["acc"])
        
        synth_rej_outcomes["rej_prop"] = synth_rej_outcomes["rej"] / (synth_rej_outcomes["rej"] + synth_rej_outcomes["acc"])
        
        synth_outcomes = pandas.merge(synth_rej_outcomes, synth_gen_outcomes, suffixes=(rej_suffix, gen_suffix), 
                                      left_index=True, right_index=True)
        synth_outcomes["null"] = synth_truth_mask
    
        synth_outcomes["grund_truth"] = synth_truth_mask
        synth_outcomes["color"] = synth_truth_mask.apply(cmapf)
        synth_outcomes["drug_name"] = synth_outcomes.index
        return synth_outcomes