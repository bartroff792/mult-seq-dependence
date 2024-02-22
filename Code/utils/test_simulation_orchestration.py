from . import simulation_orchestration


# def test_compute_fdp():
#     fdp = simulation_orchestration.compute_fdp(
#         [0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
#         [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
#     )
#     assert fdp == 0.5


class TestMCSimAndAnalyzeSynthData:
    def test_basic_functionality(self):
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
            scale_fdr=True,
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

    def test_finite_horizon(self):
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
            scale_fdr=True,
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