import numpy as np
import pandas as pd
from . import data_funcs
import pytest
import copy



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
        assert obs.shape == (100, 3), f"obs.shape={obs.shape} expected (100, 3)"
        assert (
            obs.index.name == "period"
        ), f"obs.index.name={obs.index.name} expected 'period'"
        assert (
            obs.columns.name == "hyp_name"
        ), f"obs.columns.name={obs.columns.name} expected 'hyp_name'"
        assert (
            obs.index.dtype == np.int64
        ), f"obs.index.dtype={obs.index.dtype} expected np.int64"
        # assert obs.columns.dtype == np.object, f"obs.columns.dtype={obs.columns.dtype} expected np.object"
        assert (
            obs.dtypes.iloc[0] == np.float64
        ), f"obs.dtypes[0]={obs.dtypes[0]} expected np.float64"
        assert (
            obs.dtypes.iloc[1] == np.float64
        ), f"obs.dtypes[1]={obs.dtypes[1]} expected np.float64"
        assert (
            obs.dtypes.iloc[2] == np.float64
        ), f"obs.dtypes[2]={obs.dtypes[2]} expected np.float64"
        assert obs.min().min() >= 0, f"obs.min().min()={obs.min().min()} expected >= 0"
        assert (
            obs.max().max() <= 25
        ), f"obs.max().max()={obs.max().max()} expected <= 10"

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
        assert obs.shape == (100, 3)
        assert obs.index.name == "period"
        assert obs.columns.name == "hyp_name"
        assert obs.index.dtype == np.int64
        # assert obs.columns.dtype == np.object
        assert obs.dtypes.iloc[0] == np.float64
        assert obs.dtypes.iloc[1] == np.float64
        assert obs.dtypes.iloc[2] == np.float64
        assert obs.min().min() >= 0
        assert obs.max().max() <= 10


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
        llr, obs = data_funcs.generate_llr_general(
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
        llr, obs = data_funcs.generate_llr_general(
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

