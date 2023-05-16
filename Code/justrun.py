#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:21:11 2018

@author: mhankin
"""
from pylab import *
import demo
import logging
logger = logging.getLogger()
logger.setLevel(20)
def doit(n_periods, alpha):
    xplot, y = (demo.finite_horizon_seq_stepdown_plot(n_periods=n_periods, alpha=alpha,
            skip_main_plot=True, stepup=True, do_scale=False, k_reps=600, do_parallel=False, sleep_time=30.0))
    y = y.drop("ar0", 1)
    relevant_steps = np.sort(y["step"].unique())[:-1]
    rej_list = []
    for step in relevant_steps:
        rej_list.append(np.sum(y["rejLevel"].apply(lambda u: not np.isnan(u)).astype(int) * (y["step"] <= step).astype(int)))
    return (relevant_steps, rej_list)
fig = figure(figsize=(12,12))
for n_periods in [1000, 2000]:
    plot(*doit(n_periods, 0.1), label=str(n_periods))
legend()
xlabel("Step number")
ylabel("Number of rejections")
title("Finite Horizon Yellowcard Rejections")
fig.savefig("{STORAGE_DIR}/Oct2018-rejective.png")

#x = demo.finite_horizon_seq_stepdown_plot(0.05, skip_main_plot=True, n_periods=1000, rho=None, stepup=True, do_scale=False)
#y=x[1].drop("ar0", 1)
#relevant_steps1000 = np.sort(y["step"].unique())[:-1]
#
#rej1000_list = []
#fig = figure(figsize=(16, 12))
#for step in relevant_steps1000:
#    rej1000_list.append(np.sum(y["rejLevel"].apply(lambda u: not np.isnan(u)).astype(int) * (y["step"] <= step).astype(int)))
#
#x = demo.finite_horizon_seq_stepdown_plot(0.05, skip_main_plot=True, n_periods=2000, rho=None, stepup=True, do_scale=False)
#y=x[1].drop("ar0", 1)
#relevant_steps2000 = np.sort(y["step"].unique())[:-1]
#
#rej2000_list = []
#fig = figure(figsize=(16, 12))
#for step in relevant_steps2000:
#    rej2000_list.append(np.sum(y["rejLevel"].apply(lambda u: not np.isnan(u)).astype(int) * (y["step"] <= step).astype(int)))
#
#    
#plot(relevant_steps1000, rej1000_list, label="T=1000")
#plot(relevant_steps2000, rej2000_list, label="T=2000")
#legend(fontsize=16)
#xticks(fontsize=16)
#yticks(fontsize=16)
#xlabel("Timestep", fontsize=19)
#ylabel("Number of Hypotheses", fontsize=19)
#title("Rejective", fontsize=24)
#fig.figsave("{STORAGE_DIR}/Sept2018-rejective.png")