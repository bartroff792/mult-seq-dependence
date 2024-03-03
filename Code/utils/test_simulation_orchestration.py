from . import simulation_orchestration
import pytest
import numpy as np
import itertools
# def test_compute_fdp():
#     fdp = simulation_orchestration.compute_fdp(
#         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#     )
#     assert fdp == 0.5

@pytest.mark.parametrize("error_control",[None, "fdr", "pfdr"])
class TestMCSimAndAnalyzeSynthData:
    def test_basic_functionality(self, error_control):
        fdp_data = simulation_orchestration.mc_sim_and_analyze_synth_data(
            alpha=0.1,
            beta=0.1,
            cut_type="BL",
            theta0=0.05,
            theta1=0.045,
            hyp_type="binom",
            extra_params={"n": 10},
            n_periods=None,
            m_null=3,
            m_alt=7,
            sim_reps=4,
            m0_known=False,
            error_control=error_control,
            rho=-0.5,
            interleaved=False,
            undershoot_prob=0.2,
            fin_par=True,
            fh_sleep_time=60,
            do_iterative_cutoff_MC_calc=False,
            stepup=False,
            analysis_func=simulation_orchestration.compute_fdp,
            rand_order=False,
        )

    def test_finite_horizon(self, error_control):
        fdp_data = simulation_orchestration.mc_sim_and_analyze_synth_data(
            alpha=0.1,
            beta=None,
            cut_type="BL",
            theta0=0.05,
            theta1=0.045,
            hyp_type="binom",
            extra_params={"n": 10},
            n_periods=25,
            m_null=3,
            m_alt=7,
            sim_reps=4,
            m0_known=False,
            error_control=error_control,
            rho=-0.5,
            interleaved=False,
            undershoot_prob=0.98,
            fin_par=True,
            fh_sleep_time=60,
            do_iterative_cutoff_MC_calc=False,
            stepup=False,
            analysis_func=simulation_orchestration.compute_fdp,
            rand_order=False,
        )


def generate_sim_cutoff_params():
    cut_types = ["BL", "BY", "HOLM"]
    error_controls = ["fdr", "pdfr", None]
    betas = [0.2, None]
    return list(itertools.product(cut_types, error_controls, betas))

class TestSimCutoffs:
    @pytest.mark.parametrize(["cut_type"], [("BL",), ("BY",), ("HOLM",)])
    def test_base_pvalue_cuts(self, cut_type):
        m_hyps = 5
        alpha_vec = simulation_orchestration.construct_base_pvalue_cutoffs(
            cut_type=cut_type,
            m_total=m_hyps,
            alpha=0.1,
        )
        assert len(alpha_vec)==m_hyps
        assert (np.diff(alpha_vec)>=0).all()

    @pytest.mark.parametrize(["cut_type", "error_control", "beta"], generate_sim_cutoff_params())
    def test_construct_sim_pvalue_cutoffs(self, cut_type, error_control, beta):
        m_hyps = 5
        alpha = 0.1
        alpha_vec, beta_vec = simulation_orchestration.construct_sim_pvalue_cutoffs(
            m_total=m_hyps,
            alpha=alpha,
            beta=beta,
            error_control=error_control,
            cut_type=cut_type,
        )
        if beta is None:
            print("hello")
            print("world")
        assert len(alpha_vec) == m_hyps
        assert (np.diff(alpha_vec)>=0).all()
        if beta is None:
            assert beta_vec is None
        else:
            assert len(beta_vec) == m_hyps
            assert (np.diff(beta_vec)>=0).all()


    def test_llr_cuts(self):
        simulation_orchestration.calc_llr_cutoffs
        raise NotImplementedError()