# -*- coding: utf-8 -*-
"""
Created on Thu May 12 15:02:39 2016

@author: mike
"""

import unittest
import multseq #, fake_data
from numpy import arange, diff, array, percentile
import pandas
from pandas.util import testing as ptest
import pdb, logging
from utils import common_funcs, cutoff_funcs, data_funcs
from utils.common_funcs import log_odds, sigmoid
from utils.cutoff_funcs import finite_horizon_rejective_cutoffs
from utils.data_funcs import read_drug_data, simulate_reactions, assemble_llr

class TestMainTestProcedures(unittest.TestCase):
    def setUp(self):
        pass
        #self.addTypeEqualityFunc(pandas.DataFrame, ptest.assert
        
    def test_finite_horizon_rejective_cutoffs_ON_DRUGS_WITH_RATE(self):
        dar, dnar, _ = read_drug_data()
        drr_raw = dar + dnar
        # Screen rare drugs and drugs with no amnesia SE
        drug_mask = array( (drr_raw > 1) & (dar > 0) )
        drr = drr_raw[drug_mask]
        p0 = .002
        p1 = .004
        alpha_levels = array([.01, .05, .10])
        n_periods = 1000
        k_reps = 100
        cutoff_est, data_rec = finite_horizon_rejective_cutoffs(
            drr, p0, p1, alpha_levels, n_periods, k_reps, dbg=True)
        data_rec = array(data_rec)
        print sum(data_rec.var(0) == 0.0)

    # TODO: failing
    def test_modular_sprt_test_rejective(self):
        data_obj = fake_data.trickier()
        A1, B1, C1 = multseq.modular_sprt_test(data_obj.nllr, data_obj.A_vec,
                                               data_obj.B_vec, 
                                               record_interval=1, 
                                               rejective=True)
        print "-"*5, "A1", "-"*5
        for k, v in A1.iteritems():
            print "="*5, k,"="*5
            print v
        print "-"*5, "B1", "-"*5        
        print B1
        print "-"*5, "C1", "-"*5        
        print C1
 
    # TODO: failing       
    def test_modular_sprt_test(self):
        llr = pandas.DataFrame({'drug1':array([0.0, 2.5, 3.5, 17.0, -30, 12]), 
                                'drugA':array([0.0, 2.5, 1.5, -1.5, 1.5, 2.5]),
                                'drugZ':array([0.0, 0.5, -0.5, -1.5, -2.5, -1.5]),
                                'drugNull':array([0.0, 0.2, -0.2, 0.0, 0.2, -0.2])})
        #fine_grained_outcomes, termination_time_series, cutoff_output = multseq.modular_sprt_test(nllr)
        A_vec = array([2, 1, .5, .25])
        B_vec = -array([3.0, 2.0, 1.0, .5])
        A1, B1, C1 = multseq.modular_sprt_test(llr, A_vec, B_vec, 
                                               record_interval=1)
        print "-"*5, "A1", "-"*5
        for k, v in A1.iteritems():
            print "="*5, k,"="*5
            print v
        print "-"*5, "B1", "-"*5        
        print B1
        print "-"*5, "C1", "-"*5        
        print repr(C1)
        

    # TODO: failing
    def test_modular_sprt_test_wrapper(self):
        dar, dnar, meta_data = data_funcs.read_drug_data()
        am_reacts, nonam_reacts = data_funcs.simulate_reactions(dar, dnar, 500)
        p0 = data_funcs.whole_data_p0(*meta_data)
        p1 = data_funcs.whole_data_p1(*meta_data, p0=p0, n_se=2.0)
        llr_paths = data_funcs.assemble_llr(am_reacts, nonam_reacts, p0, p1)
        #fine_grained_outcomes, termination_time_series, cutoff_output = multseq.modular_sprt_test(nllr)
        A1, B1, C1 = multseq.modular_sprt_test_wrapper(llr_paths)
        A2, B2, C2 = multseq.sprt_test(llr_paths) 
#        pdb.set_trace()
        ptest.assert_almost_equal(A1, A2)
        ptest.assert_almost_equal(B1, B2)
        ptest.assert_almost_equal(C1, C2)
        
    def func(self):
        drr = pandas.Series({"ab":1.0, "cd":1.0, "ef":2.0, "gh":3.0, "ij":5.0})
        p0 = .055
        p1 = .05
        alpha= .1
        beta = .2
        dar = drr * pandas.Series({"ab":p0, "cd":p1, "ef":p0, "gh":p1, "ij":p0})
        dnar = drr - dar
        alpha_beta, A_B = cutoff_funcs.calc_bh_alpha_and_cuts(alpha, beta, 5)
        expect_len = cutoff_funcs.est_sample_size(alpha_beta[0], alpha_beta[1], drr, p0, p1)
        print expect_len
        for j in range(2):
            am_reacts, nonam_reacts = data_funcs.simulate_reactions(dar, dnar, 10 * expect_len)
            llr_paths = data_funcs.assemble_llr(am_reacts, nonam_reacts, p0, p1)
            retval = multseq.modular_sprt_test_wrapper(llr_paths, alpha, beta)
        
    def test_step_down_elimination(self):
        data_ser = pandas.Series({"abc": 1.0, "def": 2.0, "ghi":-1.0, "jkl":-2.0, "mno":-3.0, "pqr":.75})
        cutoff_vec = array([2.5, 1.5, .5, .1, .1, .1, .1])
        num_eliminated = 1
        yes_elim = multseq.step_down_elimination(data_ser, cutoff_vec, num_eliminated, highlow="high")
        logging.info("-"*5 + "\nStepdown: {0}\n{1}\n".format(yes_elim, 
                     multseq.step_illustration(data_ser, cutoff_vec, num_eliminated, highlow="high")))
        self.assertIn("abc", yes_elim[0])
        self.assertIn("def", yes_elim[0])
        self.assertEqual(3, yes_elim[1])
        self.assertEqual(len(yes_elim[0]), yes_elim[1])
        
        # No elim
        num_eliminated = 0
        no_elim = multseq.step_down_elimination(data_ser, cutoff_vec, num_eliminated, highlow="high")
        logging.info("-"*5 + "\nStepdown: {0}\n{1}\n".format(no_elim, 
                     multseq.step_illustration(data_ser, cutoff_vec, num_eliminated, highlow="high")))
        self.assertEqual(0, no_elim[1])
        self.assertEqual(len(no_elim[0]), no_elim[1])
        
        # skip
        data_ser = pandas.Series({"abc": 1.0, "def": 2.5, "ghi":-1.0, "jkl":-2.0, "mno":-3.0, "pqr":.75})
        cutoff_vec = array([2.0, 1.75, .5, .1, .1, .1])
        num_eliminated = 0
        skip_elim = multseq.step_down_elimination(data_ser, cutoff_vec, num_eliminated, highlow="high")
        logging.info("-"*5 + "\nStepdown: {0}\n{1}\n".format(skip_elim, 
                     multseq.step_illustration(data_ser, cutoff_vec, num_eliminated, highlow="high")))
        self.assertIn("def", skip_elim[0])
        self.assertEqual(1, skip_elim[1])
        self.assertEqual(len(skip_elim[0]), skip_elim[1])
        
    def test_step_up_elimination(self):
        data_ser = pandas.Series({"abc": 1.0, "def": 2.0, "ghi":-1.0, "jkl":-2.0, "mno":-3.0, "pqr":.75})
        cutoff_vec = array([2.5, 1.5, .5, .1, .1, .1, .1])
        num_eliminated = 1
        yes_elim = multseq.step_up_elimination(data_ser, cutoff_vec, num_eliminated, highlow="high")
        logging.info("-"*5 + "\nStepup: {0}\n{1}\n".format(yes_elim, 
                     multseq.step_illustration(data_ser, cutoff_vec, num_eliminated, highlow="high")))
        self.assertIn("abc", yes_elim[0])
        self.assertIn("def", yes_elim[0])
        self.assertEqual(3, yes_elim[1])
        self.assertEqual(len(yes_elim[0]), yes_elim[1])
        
        # No elim 
        # TODO: Fix so that it actually fails to eliminate!
        num_eliminated = 3
        no_elim = multseq.step_up_elimination(data_ser, cutoff_vec, num_eliminated, highlow="high")
        logging.info("-"*5 + "\nStepup: {0}\n{1}\n".format(no_elim, 
                     multseq.step_illustration(data_ser, cutoff_vec, num_eliminated, highlow="high")))
        self.assertEqual(3, no_elim[1])
        self.assertEqual(len(no_elim[0]), no_elim[1])
        
        # skip
        data_ser = pandas.Series({"abc": 1.0, "def": 2.5, "ghi":-1.0, "jkl":-2.0, "mno":-3.0, "pqr":.75})
        cutoff_vec = array([2.0, 1.75, .5, .1, .1, .1])
        num_eliminated = 3
        skip_elim = multseq.step_up_elimination(data_ser, cutoff_vec, num_eliminated, highlow="high")
        logging.info("-"*5 + "\nStepup: {0}\n{1}\n".format(skip_elim, 
                     multseq.step_illustration(data_ser, cutoff_vec, num_eliminated, highlow="high")))
        self.assertIn("def", skip_elim[0])
        self.assertEqual(3, skip_elim[1])
        self.assertEqual(len(skip_elim[0]), skip_elim[1])


if __name__ == '__main__':
    unittest.main()