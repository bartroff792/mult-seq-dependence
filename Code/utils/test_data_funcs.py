import numpy as np
import pandas as pd
from . import data_funcs
import pytest
import copy



def build_simple_drug_params(m_hyps):
    mu = pd.Series(np.linspace(3,5, m_hyps))
    p0 = pd.Series(np.linspace(0.1, 0.5, m_hyps))
    p1 = p0+0.1
    params0 = {"mu":mu, "p":p0}
    params1 = {"mu":mu, "p":p1}
    return params0, params1

class TestComputeLLR:
    def test_binom_stepwise(self):
        n = 10
        hyp_idx = pd.Index(["a", "b", "c"], name="hyp_name")
        params = {
            "p": pd.Series([0.1, 0.2, 0.3], index=hyp_idx),
            "n": pd.Series(n, index=hyp_idx),
        }
        params0 = copy.deepcopy(params)
        params0["p"][:] = 0.1
        params1 = copy.deepcopy(params)
        params1["p"][:] = 0.3
        np.random.seed(42)
        obs = data_funcs.simulate_correlated_observations(
            params=params,
            n_periods=50,
            rho=0.5,
            hyp_type="binom",
        )
        llr = data_funcs.compute_llr(observed_data=obs,
                               hyp_type="binom",
                               params0=params0,
                               params1=params1,
                               cumulative=False)
        snr = llr.mean() / llr.std()
        # Ensure that the relative direction of llr for each
        # hypothesis is increasing, ie that the final one is more
        # likely to be rejected than the first
        assert np.diff(snr).min() > 0.5
    

    def test_binom_cumulative(self):
        n = 10
        hyp_idx = pd.Index(["a", "b", "c"], name="hyp_name")
        params = {
            "p": pd.Series([0.1, 0.2, 0.3], index=hyp_idx),
            "n": pd.Series(n, index=hyp_idx),
        }
        params0 = copy.deepcopy(params)
        params0["p"][:] = 0.1
        params1 = copy.deepcopy(params)
        params1["p"][:] = 0.3

        obs = data_funcs.simulate_correlated_observations(
            params=params,
            n_periods=50,
            rho=0.5,
            hyp_type="binom",
        )
        llr = data_funcs.compute_llr(observed_data=obs,
                               hyp_type="binom",
                               params0=params0,
                               params1=params1,
                               cumulative=True)
        trend = llr.diff().mean()
        # 
        assert np.diff(trend).min() > 0.1
    


class TestConstructDGP:
    def test_binom(self):
        dgp, gt = data_funcs.construct_dgp(
            m_null=5,
            m_alt=5,
            theta0=0.1,
            theta1=0.2,
            interleaved=False,
            hyp_type="binom",
            extra_params={"n": 10},
        )
        assert len(dgp["p"]) == 10
        assert len(gt) == 10

    def test_pois(self):
        dgp, gt = data_funcs.construct_dgp(
            m_null=5,
            m_alt=5,
            theta0=1.1,
            theta1=3.2,
            interleaved=False,
            hyp_type="pois",
        )
        assert len(dgp["mu"]) == 10
        assert len(gt) == 10

    def test_drug(self):
        dgp, gt = data_funcs.construct_dgp(
            m_null=5,
            m_alt=5,
            theta0=0.1,
            theta1=0.2,
            interleaved=False,
            hyp_type="drug",
            extra_params={"mu": 10},
        )
        assert len(dgp["p"]) == 10
        assert len(gt) == 10

    def test_m_alt_null_and_interleaved(self):
        dgp, gt = data_funcs.construct_dgp(
            m_null=5,
            m_alt=None,
            theta0=0.1,
            theta1=0.2,
            interleaved=True,
            hyp_type="binom",
            extra_params={"n": 10},
        )
        assert len(dgp["p"]) == 10
        assert len(gt) == 10

    def test_extraneous_extra_parameter_binom(self):
        with pytest.warns(UserWarning):
            dgp, gt = data_funcs.construct_dgp(
                m_null=5,
                m_alt=5,
                theta0=0.1,
                theta1=0.2,
                interleaved=False,
                hyp_type="binom",
                extra_params={"n": 10, "extra": 5},
            )

    def test_extraneous_extra_parameter_pois(self):
        with pytest.warns():
            dgp, gt = data_funcs.construct_dgp(
                m_null=5,
                m_alt=5,
                theta0=0.1,
                theta1=0.2,
                interleaved=False,
                hyp_type="pois",
                extra_params={
                    "n": 10,
                },
            )

    def test_missing_extra_parameter(self):
        with pytest.raises(KeyError):
            dgp, gt = data_funcs.construct_dgp(
                m_null=5,
                m_alt=5,
                theta0=0.1,
                theta1=0.2,
                interleaved=False,
                hyp_type="binom",
                extra_params={},
            )


def test_simple_toeplitz_corr_mat():
    corr_mat = data_funcs.simple_toeplitz_corr_mat(0.5, 5)
    assert corr_mat.shape == (5, 5)


class TestSimulateCorrelatedObservations:
    def test_basic_functionality_pois(self):
        hyp_idx = pd.Index(["a", "b", "c"], name="hyp_name")
        params = {"mu": pd.Series([1.0, 2.0, 3.0], index=hyp_idx)}
        obs = data_funcs.simulate_correlated_observations(
            params=params,
            n_periods=100,
            rho=0.5,
            hyp_type="pois",
        )
        assert len(obs)==1
        assert "obs" in obs
        obs_df = obs["obs"]
        assert obs_df.shape == (100, 3), f"obs_df.shape={obs_df.shape} expected (100, 3)"
        assert (
            obs_df.index.name == "period"
        ), f"obs_df.index.name={obs_df.index.name} expected 'period'"
        assert (
            obs_df.columns.name == "hyp_name"
        ), f"obs_df.columns.name={obs_df.columns.name} expected 'hyp_name'"
        assert (
            obs_df.index.dtype == np.int64
        ), f"obs_df.index.dtype={obs_df.index.dtype} expected np.int64"
        # assert obs_df.columns.dtype == np.object, f"obs_df.columns.dtype={obs_df.columns.dtype} expected np.object"
        assert (
            obs_df.dtypes.iloc[0] == np.float64
        ), f"obs_df.dtypes[0]={obs_df.dtypes[0]} expected np.float64"
        assert (
            obs_df.dtypes.iloc[1] == np.float64
        ), f"obs_df.dtypes[1]={obs_df.dtypes[1]} expected np.float64"
        assert (
            obs_df.dtypes.iloc[2] == np.float64
        ), f"obs_df.dtypes[2]={obs_df.dtypes[2]} expected np.float64"
        assert obs_df.min().min() >= 0, f"obs_df.min().min()={obs_df.min().min()} expected >= 0"
        assert (
            obs_df.max().max() <= 25
        ), f"obs_df.max().max()={obs_df.max().max()} expected <= 10"

    def test_basic_functionality_binom(self):

        hyp_idx = pd.Index(["a", "b", "c"], name="hyp_name")
        params = {
            "p": pd.Series([0.1, 0.2, 0.3], index=hyp_idx),
            "n": pd.Series([10, 10, 10], index=hyp_idx),
        }
        obs = data_funcs.simulate_correlated_observations(
            params=params,
            n_periods=100,
            rho=0.5,
            hyp_type="binom",
        )
        assert len(obs)==1
        assert "obs" in obs
        obs_df = obs["obs"]
        assert obs_df.shape == (100, 3)
        assert obs_df.index.name == "period"
        assert obs_df.columns.name == "hyp_name"
        assert obs_df.index.dtype == np.int64
        # assert obs.columns.dtype == np.object
        assert obs_df.dtypes.iloc[0] == np.float64
        assert obs_df.dtypes.iloc[1] == np.float64
        assert obs_df.dtypes.iloc[2] == np.float64
        assert obs_df.min().min() >= 0
        assert obs_df.max().max() <= 10

    def test_basic_functionality_drug(self):

        hyp_idx = pd.Index(["a", "b", "c"], name="hyp_name")
        params = {
            "p": pd.Series([0.1, 0.2, 0.3], index=hyp_idx),
            "mu": pd.Series([10, 10, 10], index=hyp_idx),
        }
        obs = data_funcs.simulate_correlated_observations(
            params=params,
            n_periods=100,
            rho=0.5,
            hyp_type="drug",
        )
        assert len(obs)==2
        assert "relevant_events" in obs
        assert "non_relevant_events" in obs
        for kk, obs_df in obs.items():
            assert obs_df.shape == (100, 3)
            assert obs_df.index.name == "period"
            assert obs_df.columns.name == "hyp_name"
            assert obs_df.index.dtype == np.int64
            # assert obs.columns.dtype == np.object
            assert obs_df.dtypes.iloc[0] == np.float64
            assert obs_df.dtypes.iloc[1] == np.float64
            assert obs_df.dtypes.iloc[2] == np.float64
            assert obs_df.min().min() >= 0

    def test_copula_draw_contstant(self):
        n_periods = 1000
        hyp_idx = pd.Index(["a", "b", "c"], name="hyp_name")
        rho = 0.3
        np.random.seed(42)
        unif_draw = data_funcs.copula_draw(hyp_idx=hyp_idx,
                               n_periods=n_periods,
                               rho=rho,
                               rand_order=False)
        cov_mat = unif_draw.cov()
        assert cov_mat.shape == (3, 3)
        assert cov_mat.loc["a", "b"] > cov_mat.loc["a", "c"]

    
    def test_copula_draw_block_diag(self):
        n_periods = 1000
        hyp_idx = pd.Index(["a", "b", "c", "d"], name="hyp_name")
        rho = data_funcs.CorrelationContainer(
            rho=pd.Series([-0.98, 0.98], index=[0,1]),
            group_ser=pd.Series([1, 0, 0, 1], index=hyp_idx)
        )

        np.random.seed(42)
        unif_draw = data_funcs.copula_draw(hyp_idx=hyp_idx,
                               n_periods=n_periods,
                               rho=rho,
                               rand_order=False)
        assert unif_draw.shape == (1000, 4)
        cov_mat = unif_draw.corr()
        TOL = 0.1
        assert cov_mat.loc["a", "d"] >  1.0 - TOL
        assert np.abs(cov_mat.loc["a", "c"]) < TOL
        assert cov_mat.loc["b", "c"] < -1 + TOL

class TestGeneralLLR:
    def test_binom(self):
        n = 10
        hyp_idx = pd.Index(["a", "b", "c"], name="hyp_name")
        params = {
            "p": pd.Series([0.1, 0.2, 0.3], index=hyp_idx),
            "n": pd.Series(n, index=hyp_idx),
        }
        params0 = copy.deepcopy(params)
        params0["p"][:] = 0.1
        params1 = copy.deepcopy(params)
        params1["p"][:] = 0.3
        llr, obs = data_funcs.generate_llr(
            params=params,
            n_periods=10,
            rho=0.6,
            hyp_type="binom",
            params0=params0,
            params1=params1,
            rand_order=False,
        )
        assert llr.shape == (10, 3), f"llr.shape={llr.shape} expected (10, 3)"
        assert (
            llr.index.name == "period"
        ), f"llr.index.name={llr.index.name} expected 'period'"
        assert (
            llr.columns.name == "hyp_name"
        ), f"llr.columns.name={llr.columns.name} expected 'hyp_name'"

    def test_pois(self):
        hyp_idx = pd.Index(["a", "b", "c"], name="hyp_name")
        params = {"mu": pd.Series([1.0, 2.0, 3.0], index=hyp_idx)}
        params0 = copy.deepcopy(params)
        params0["mu"][:] = 1.0
        params1 = copy.deepcopy(params)
        params1["mu"][:] = 3.0
        llr, obs = data_funcs.generate_llr(
            params=params,
            n_periods=10,
            rho=0.6,
            hyp_type="pois",
            params0=params0,
            params1=params1,
            rand_order=False,
        )
        assert llr.shape == (10, 3)
        assert llr.index.name == "period"
        assert llr.columns.name == "hyp_name"


    def test_drug(self):
        mu = 10
        hyp_idx = pd.Index(["a", "b", "c"], name="hyp_name")
        params = {
            "p": pd.Series([0.1, 0.2, 0.3], index=hyp_idx),
            "mu": pd.Series(mu, index=hyp_idx),
        }
        params0 = copy.deepcopy(params)
        params0["p"][:] = 0.1
        params1 = copy.deepcopy(params)
        params1["p"][:] = 0.3
        llr, obs = data_funcs.generate_llr(
            params=params,
            n_periods=10,
            rho=0.6,
            hyp_type="drug",
            params0=params0,
            params1=params1,
            rand_order=False,
        )
        assert llr.shape == (10, 3), f"llr.shape={llr.shape} expected (10, 3)"
        assert (
            llr.index.name == "period"
        ), f"llr.index.name={llr.index.name} expected 'period'"
        assert (
            llr.columns.name == "hyp_name"
        ), f"llr.columns.name={llr.columns.name} expected 'hyp_name'"
