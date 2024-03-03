from . import cutoff_funcs
import numpy as np
import pandas as pd
from scipy import stats
import pytest
# DONE simulation_orchestration.mc_sim_and_analyze_synth_data -> simulation_orchestration.calc_sim_cutoffs
# DONE simulation_orchestration.calc_sim_cutoffs -> cutoff_funcs.est_sample_size
# simulation_orchestration.calc_sim_cutoffs -> cutoff_funcs.finite_horizon_rejective_cutoffs
# cutoff_funcs.finite_horizon_rejective_cutoffs -> cutoff_funcs.finite_horizon_cutoff_simulation_wrapper
# cutoff_funcs.finite_horizon_cutoff_simulation_wrapper -> cutoff_funcs.finite_horizon_cutoff_simulation
# DONE cutoff_funcs.est_sample_size -> cutoff_funcs.llr_term_moments_general
EasyA_vec = np.array([3, 2, 1])
EasyB_vec = np.array([-3, -2, -1])

def build_simple_drug_params(m_hyps):
    mu = pd.Series(np.linspace(3,5, m_hyps))
    p0 = pd.Series(np.linspace(0.1, 0.5, m_hyps))
    p1 = p0+0.1
    params0 = {"mu":mu, "p":p0}
    params1 = {"mu":mu, "p":p1}
    return params0, params1
    

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

    def test_drug(self):
        m_hyps = 5
        params0, params1 = build_simple_drug_params(m_hyps)
        term_mean = cutoff_funcs.llr_term_mean_general(params0,params1, "drug")
        assert (term_mean<0).all()
        alt_term_mean = cutoff_funcs.llr_term_mean_general(params1,params0, "drug")
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
        sample_size = cutoff_funcs.est_sample_size(EasyA_vec, EasyB_vec, params0, params1, "binom")
        assert sample_size>5

    def test_poisson(self):
        lam0 = 5
        lam1 = 5.1
        params0 = {"mu":pd.Series([lam0,lam0,lam0])}
        params1 = {"mu":pd.Series([lam1,lam1,lam1])}
        
        sample_size = cutoff_funcs.est_sample_size(EasyA_vec, EasyB_vec, params0, params1, "pois")
        assert sample_size>5

    def test_drug(self):
        m_hyps = 5
        params0, params1 = build_simple_drug_params(m_hyps)
        sample_size = cutoff_funcs.est_sample_size(EasyA_vec, EasyB_vec, params0, params1, "drug")
        assert sample_size>5

class TestCutoffVerifier:
    def test_kosher(self):
        cutoff_funcs.llr_cutoff_verifier(EasyA_vec, EasyB_vec)

    def test_fail_len(self):
        with pytest.raises(AssertionError):
            cutoff_funcs.llr_cutoff_verifier(EasyA_vec, EasyB_vec[:-1])

    def test_fail_order(self):
        with pytest.raises(AssertionError):
            cutoff_funcs.llr_cutoff_verifier(EasyB_vec, EasyA_vec)
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
        max_llr, weights = cutoff_funcs.finite_horizon_cutoff_simulation(
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
        np.random.seed(42)
        max_llr, weights = cutoff_funcs.finite_horizon_cutoff_simulation(
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
        TOL = 0.5
        assert mean_cut > TOL * np.sqrt(var_cut), f"mean_cut={mean_cut} <= np.sqrt(var_cut)={np.sqrt(var_cut)}"

    def test_drug(self):
        
        m_hyps = 5
        params0, params1 = build_simple_drug_params(m_hyps)
        np.random.seed(42)
        max_llr, weights = cutoff_funcs.finite_horizon_cutoff_simulation(
            params0=params0,
            params1=params1,
            hyp_type="drug",
            n_periods=50,
            n_reps=10,
            )
        assert not np.isnan(max_llr).any(), f"max_llr has nans"
        assert not np.isnan(weights).any(), f"weights has nans"
        assert (weights>=0).all(), f"weights has negative values"
        assert len(weights)==len(max_llr), f"weights and max_llr have different lengths"
        # TODO: document what this is testing and why it should be expected to be true, and figure out why such a generous tolerance is required to pass
        mean_cut = np.sum(weights * max_llr) / np.sum(weights)
        var_cut = np.sum(weights * (max_llr - mean_cut)**2) / np.sum(weights)
        TOL = 0.25
        assert mean_cut > TOL * np.sqrt(var_cut), f"mean_cut={mean_cut} <= np.sqrt(var_cut)={np.sqrt(var_cut)}"
        



class TestInfiniteHorizonPfdrPfnrPvalueCutoffs:
    def test_no_m0_guess(self):
        m_hyps = 5
        m0 = None
        alpha_pfdr = 0.05
        beta_pfnr = 0.2
        alpha_raw = cutoff_funcs.create_fdr_controlled_bh_alpha_indpt(alpha_pfdr, m_hyps)
        beta_raw = cutoff_funcs.create_fdr_controlled_bh_alpha_indpt(beta_pfnr, m_hyps, )
        alpha_vec, beta_vec = cutoff_funcs.pfdr_pfnr_infinite_horizon_pvalue_cutoffs(
            alpha_raw, 
            beta_raw,
            alpha_pfdr,
            beta_pfnr,
            m0,
        )

    
    def test_kosher_m0_guess(self):
        m_hyps = 5
        m0 = 2
        alpha_pfdr = 0.05
        beta_pfnr = 0.2
        alpha_raw = cutoff_funcs.create_fdr_controlled_bh_alpha_indpt(alpha_pfdr, m_hyps)
        beta_raw = cutoff_funcs.create_fdr_controlled_bh_alpha_indpt(beta_pfnr, m_hyps, )
        alpha_vec, beta_vec = cutoff_funcs.pfdr_pfnr_infinite_horizon_pvalue_cutoffs(
            alpha_raw, 
            beta_raw,
            alpha_pfdr,
            beta_pfnr,
            m0,
        )
        cutoff_funcs.pvalue_cutoff_verifier(alpha_vec, beta_vec)

    
    @pytest.mark.parametrize("m,m0", [(5, 0),  (5, 5)])
    def test_bad_m0_guess(self, m, m0):
        m_hyps = m
        alpha_pfdr = 0.05
        beta_pfnr = 0.2
        alpha_raw = cutoff_funcs.create_fdr_controlled_bh_alpha_indpt(alpha_pfdr, m_hyps)
        beta_raw = cutoff_funcs.create_fdr_controlled_bh_alpha_indpt(beta_pfnr, m_hyps, )
        with pytest.raises(AssertionError):
            alpha_vec, beta_vec = cutoff_funcs.pfdr_pfnr_infinite_horizon_pvalue_cutoffs(
                alpha_raw, 
                beta_raw,
                alpha_pfdr,
                beta_pfnr,
                m0,
            )

    
class TestFiniteHorizonPfdrPvalueCutoffs:
    def test_no_m0_guess(self):
        m_hyps = 5
        m0 = None
        alpha_pfdr = 0.05
        alpha_raw = cutoff_funcs.create_fdr_controlled_bh_alpha_indpt(alpha_pfdr, m_hyps)
        alpha_vec = cutoff_funcs.pfdr_finite_horizon_pvalue_cutoffs(
            alpha_raw, 
            alpha_pfdr,
            m0,
        )
        cutoff_funcs.pvalue_cutoff_verifier(alpha_vec)

    
    def test_kosher_m0_guess(self):
        m_hyps = 5
        m0 = 2
        alpha_pfdr = 0.05
        alpha_raw = cutoff_funcs.create_fdr_controlled_bh_alpha_indpt(alpha_pfdr, m_hyps)
        alpha_vec = cutoff_funcs.pfdr_finite_horizon_pvalue_cutoffs(
            alpha_raw, 
            alpha_pfdr,
            m0,
        )
        cutoff_funcs.pvalue_cutoff_verifier(alpha_vec)

    
    @pytest.mark.parametrize("m,m0", [(5, 0),  (5, 5)])
    def test_bad_m0_guess(self, m, m0):
        m_hyps = m
        alpha_pfdr = 0.05
        alpha_raw = cutoff_funcs.create_fdr_controlled_bh_alpha_indpt(alpha_pfdr, m_hyps)
        with pytest.raises(AssertionError):
            alpha_vec = cutoff_funcs.pfdr_finite_horizon_pvalue_cutoffs(
            alpha_raw, 
            alpha_pfdr,
            m0,
        )

    
class TestGumbelFit:

    def test_flat_fit(self):
        np.random.seed(42)
        mu0 = 2.0
        beta0 = 1.0
        x = stats.gumbel_r(loc=mu0, scale=beta0).rvs(1000)
        weights = np.ones_like(x)
        mu, beta = cutoff_funcs.gumbel_r_param_fit(x, weights)
        tol = 0.1
        assert np.abs(mu-mu0)<tol
        assert np.abs(beta-beta0)<tol



class TestFiniteHorizonCutoffs:
    
    def test_poisson(self):

        lam0 = 5
        lam1 = 5.1
        m_hyps = 5
        params0 = {"mu":pd.Series(np.linspace(lam0, lam1, m_hyps))}
        params1 = {"mu":pd.Series(np.linspace(lam1, lam0 + lam1, m_hyps))}
        alpha_levels = cutoff_funcs.create_fdr_controlled_bl_alpha_indpt(0.1, m_hyps=m_hyps)
        np.random.seed(42)
        cutoffs = cutoff_funcs.estimate_finite_horizon_rejective_llr_cutoffs(
            params0=params0,
            params1=params1,
            alpha_levels=alpha_levels,
            n_periods=25,
            k_reps=100,
            hyp_type="pois",
        )
        cutoff_funcs.llr_cutoff_verifier(cutoffs)

    
    def test_binom(self):
        m_hyps = 5
        n = 5
        p0 = 0.25
        p1 = 0.66
        params0 = {"n":pd.Series(np.repeat([n], m_hyps)),
                  "p":pd.Series(np.linspace(p0, p1, m_hyps)),}
        params1 = {"n":pd.Series(np.repeat([n], m_hyps)),
                    "p":pd.Series(np.linspace(p1, p0 + p1, m_hyps)),}
        alpha_levels = cutoff_funcs.create_fdr_controlled_bl_alpha_indpt(0.1, m_hyps=m_hyps)
        np.random.seed(42)
        cutoffs = cutoff_funcs.estimate_finite_horizon_rejective_llr_cutoffs(
            params0=params0,
            params1=params1,
            alpha_levels=alpha_levels,
            n_periods=25,
            k_reps=100,
            hyp_type="binom",
        )
        cutoff_funcs.llr_cutoff_verifier(cutoffs)


    def test_drug(self):
        m_hyps = 5
        params0, params1 = build_simple_drug_params(m_hyps)
        alpha_levels = cutoff_funcs.create_fdr_controlled_bl_alpha_indpt(0.1, m_hyps=m_hyps)
        np.random.seed(42)
        cutoffs = cutoff_funcs.estimate_finite_horizon_rejective_llr_cutoffs(
            params0=params0,
            params1=params1,
            alpha_levels=alpha_levels,
            n_periods=25,
            k_reps=100,
            hyp_type="drug",
        )
        cutoff_funcs.llr_cutoff_verifier(cutoffs)