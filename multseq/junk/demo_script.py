# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 06:59:01 2016

@author: mike
"""

from utils import cutoff_funcs
from numpy import arange, array
reload(cutoff_funcs)


#### Plot of scaling factor
t = logspace(0.0, log10(10000), num=250).astype(int)
x = array([cutoff_funcs.fdr_func(arange(1.0, 1.0 +n) / float(n)) for n in t])

####


alpha_fdr = .1
beta_fdr = .05
N = 10

alpha_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(alpha_fdr, 
                                                     arange(1.0, 1.0 + N))
beta_vec = cutoff_funcs.apply_fdr_control_to_alpha_vec(beta_fdr, 
                                                    arange(1.0, 1.0 + N))