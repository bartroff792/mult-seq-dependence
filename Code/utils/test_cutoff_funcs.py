from . import cutoff_funcs
import numpy as np
import pandas as pd
import pytest
# DONE simulation_orchestration.mc_sim_and_analyze_synth_data -> simulation_orchestration.calc_sim_cutoffs
# DONE simulation_orchestration.calc_sim_cutoffs -> cutoff_funcs.est_sample_size
# simulation_orchestration.calc_sim_cutoffs -> cutoff_funcs.finite_horizon_rejective_cutoffs
# cutoff_funcs.finite_horizon_rejective_cutoffs -> cutoff_funcs.finite_horizon_cutoff_simulation_wrapper
# cutoff_funcs.finite_horizon_cutoff_simulation_wrapper -> cutoff_funcs.finite_horizon_cutoff_simulation
# DONE cutoff_funcs.est_sample_size -> cutoff_funcs.llr_term_moments_general
EasyA_vec = np.array([3, 2, 1])
EasyB_vec = np.array([-3, -2, -1])

class TestLLRTermMomentsGeneral:
    def test_binom(self):
        n = 5
        p0 = 0.25
        p1 = 0.66
        params0 = {"n":pd.Series([n,n,n]),
                  "p":pd.Series([p0,p0,p0])}
        params1 = {"n":pd.Series([n,n,n]),
                    "p":pd.Series([p1,p1,p1])}
        term_mean = cutoff_funcs.llr_term_mean_general(params0,params1, "binom")
        assert (term_mean<0).all()
        alt_term_mean = cutoff_funcs.llr_term_mean_general(params1,params0, "binom")
        assert (alt_term_mean<0).all()

    def test_poisson(self):
        lam0 = 5
        lam1 = 10
        params0 = {"mu":pd.Series([lam0,lam0,lam0])}
        params1 = {"mu":pd.Series([lam1,lam1,lam1])}
        term_mean = cutoff_funcs.llr_term_mean_general(params0,params1, "pois")
        assert (term_mean<0).all()
        alt_term_mean = cutoff_funcs.llr_term_mean_general(params1,params0, "pois")
        assert (alt_term_mean<0).all()

class TestEstSampleSize:
    def test_binom(self):
        n = 5
        p0 = 0.25
        p1 = 0.26
        params0 = {"n":pd.Series([n,n,n]),
                  "p":pd.Series([p0,p0,p0])}
        params1 = {"n":pd.Series([n,n,n]),
                    "p":pd.Series([p1,p1,p1])}
        A_vec = np.array([1.0, 2.0, 3.0])
        B_vec = np.array([0.33, 0.66, 1.0])
        sample_size = cutoff_funcs.est_sample_size_general(EasyA_vec, EasyB_vec, params0, params1, "binom")
        assert sample_size>5

    def test_poisson(self):
        lam0 = 5
        lam1 = 5.1
        params0 = {"mu":pd.Series([lam0,lam0,lam0])}
        params1 = {"mu":pd.Series([lam1,lam1,lam1])}
        
        sample_size = cutoff_funcs.est_sample_size_general(EasyA_vec, EasyB_vec, params0, params1, "pois")
        assert sample_size>5

class TestCutoffVerifier:
    def test_kosher(self):
        cutoff_funcs.cutoff_verifier(EasyA_vec, EasyB_vec)

    def test_fail_len(self):
        with pytest.raises(AssertionError):
            cutoff_funcs.cutoff_verifier(EasyA_vec, EasyB_vec[:-1])

    def test_fail_order(self):
        with pytest.raises(AssertionError):
            cutoff_funcs.cutoff_verifier(EasyB_vec, EasyA_vec)
        # TODO: test other order fails
            

class TestFiniteHorizonCutoffSimulationGeneral:
    def test_binom(self):
        n = 5
        p0 = 0.25
        p1 = 0.26
        params0 = {"n":pd.Series([n,n,n]),
                  "p":pd.Series([p0,p0,p0])}
        params1 = {"n":pd.Series([n,n,n]),
                    "p":pd.Series([p1,p1,p1])}
        # The weights are wrong... they need be the weight of the full path
        # watch out for underruns
        max_llr, weights = cutoff_funcs.finite_horizon_cutoff_simulation_general(
            params0=params0, 
            params1=params1, 
            hyp_type="binom", 
            n_periods=50, 
            n_reps=10,
            )
        assert not np.isnan(max_llr).any(), f"max_llr has nans"
        assert not np.isnan(weights).any(), f"weights has nans"
        assert (weights>=0).all(), f"weights has negative values"
        assert len(weights)==len(max_llr), f"weights and max_llr have different lengths"
        mean_cut = np.sum(weights * max_llr) / np.sum(weights)
        var_cut = np.sum(weights * (max_llr - mean_cut)**2) / np.sum(weights)
        assert mean_cut > np.sqrt(var_cut), f"mean_cut={mean_cut} <= np.sqrt(var_cut)={np.sqrt(var_cut)}"

    def test_poisson(self):
        lam0 = 5
        lam1 = 5.1
        params0 = {"mu":pd.Series([lam0,lam0,lam0])}
        params1 = {"mu":pd.Series([lam1,lam1,lam1])}
        max_llr, weights = cutoff_funcs.finite_horizon_cutoff_simulation_general(
            params0=params0, 
            params1=params1, 
            hyp_type="pois", 
            n_periods=50, 
            n_reps=10,
            )
        assert not np.isnan(max_llr).any(), f"max_llr has nans"
        assert not np.isnan(weights).any(), f"weights has nans"
        assert (weights>=0).all(), f"weights has negative values"
        assert len(weights)==len(max_llr), f"weights and max_llr have different lengths"
        mean_cut = np.sum(weights * max_llr) / np.sum(weights)
        var_cut = np.sum(weights * (max_llr - mean_cut)**2) / np.sum(weights)
        assert mean_cut > np.sqrt(var_cut), f"mean_cut={mean_cut} <= np.sqrt(var_cut)={np.sqrt(var_cut)}"

class TestFiniteHorizon:
    pass

