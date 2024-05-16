import numpy as np
import pandas as pd
from . import multseq
import pytest
import copy


class TestNaiveBarrierTrips:
    def test_new_low(self):
        num_eliminated = 0
        cutoffs = np.array([-0.3, -0.2, -0.1])
        stats = np.array([-0.05, -0.25, -0.275])
        trip_vector = multseq.naive_barrier_trips(stats, cutoffs, num_eliminated, "low")
        np.testing.assert_array_equal(trip_vector, [0.0, 1.0, 0.0])

    
    def test_new_high_bad(self):
        num_eliminated = 0
        cutoffs = np.array([0.1, 0.2, 0.3])
        stats = np.array([0.05, 0.25, 0.275])
        with pytest.raises(AssertionError):
            trip_vector = multseq.naive_barrier_trips(stats, cutoffs, num_eliminated, "high")
        
    
    def test_new_high(self):
        num_eliminated = 0
        cutoffs = np.array([0.3, 0.2, 0.1])
        stats = np.array([0.05, 0.25, 0.275])
        trip_vector = multseq.naive_barrier_trips(stats, cutoffs, num_eliminated, "high")
        np.testing.assert_array_equal(trip_vector, [0.0, 1.0, 0.0])

    
    def test_new_high_flip(self):
        num_eliminated = 0
        cutoffs = np.array([0.3, 0.2, 0.1])
        stats = np.array([0.05, 0.25, 0.35])
        trip_vector = multseq.naive_barrier_trips(stats, cutoffs, num_eliminated, "high")
        np.testing.assert_array_equal(trip_vector, [1.0, 1.0, 0.0])

    
    def test_new_high_reversed(self):
        num_eliminated = 0
        cutoffs = np.array([0.3, 0.2, 0.1])
        stats = np.array([0.275, 0.25, 0.05])
        trip_vector = multseq.naive_barrier_trips(stats, cutoffs, num_eliminated, "high")
        np.testing.assert_array_equal(trip_vector, [0.0, 1.0, 0.0])

    
    def test_new_low_bad(self):
        num_eliminated = 0
        cutoffs = np.array([-0.1, -0.2, -0.3])
        stats = np.array([-0.05, -0.25, -0.275])
        with pytest.raises(AssertionError):
            trip_vector = multseq.naive_barrier_trips(stats, cutoffs, num_eliminated, "low")
        

    
    def test_mid_low(self):
        num_eliminated = 1
        cutoffs = np.array([-0.3, -0.2, -0.1])
        stats = np.array([-0.25, -0.15])
        trip_vector = multseq.naive_barrier_trips(stats, cutoffs, num_eliminated, "low")
        np.testing.assert_array_equal(trip_vector, [1.0, 1.0])
        stats_low_sig = np.array([-0.25, -0.05])
        trip_vector = multseq.naive_barrier_trips(stats_low_sig, cutoffs, num_eliminated, "low")
        np.testing.assert_array_equal(trip_vector, [1.0, 0.0])

    
    
    def test_mid_weird_low(self):
        num_eliminated = 1
        cutoffs = np.array([-0.4, -0.3, -0.2, -0.1])
        stats = np.array([-0.25, -0.275])
        trip_vector = multseq.naive_barrier_trips(stats, cutoffs, num_eliminated, "low")
        np.testing.assert_array_equal(trip_vector, [0.0, 1.0])

    
    def test_one_left_low(self):
        num_eliminated = 2
        cutoffs = np.array([-0.3, -0.2, -0.1])
        stats = np.array([-0.25,])
        trip_vector = multseq.naive_barrier_trips(stats, cutoffs, num_eliminated, "low")
        np.testing.assert_array_equal(trip_vector, [1.0,])


class TestStepDownElimination:
    def test_new_no_rejections_low(self):
        num_eliminated = 0
        cutoffs = np.array([-0.3, -0.2, -0.1])
        stats = np.array([-0.05, -0.275, -0.225])
        rej_hyps, num_rej_hyps = multseq.step_down_elimination(stats, cutoffs, num_eliminated, "low")
        assert num_rej_hyps == 0
        assert rej_hyps == []

    
    def test_new_no_rejections_high(self):
        num_eliminated = 0
        cutoffs = np.array([0.3, 0.2, 0.1])
        stats = np.array([0.05, 0.275, 0.225])
        rej_hyps, num_rej_hyps = multseq.step_down_elimination(stats, cutoffs, num_eliminated, "high")
        assert num_rej_hyps == 0
        assert rej_hyps == []
        

    def test_mid_no_rejections_high(self):
        num_eliminated = 1
        cutoffs = np.array([0.3, 0.2, 0.1])
        stats = np.array([0.15, 0.075])
        rej_hyps, num_rej_hyps = multseq.step_down_elimination(stats, cutoffs, num_eliminated, "high")
        assert num_rej_hyps == 0
        assert rej_hyps == []


    def test_new_one_rej_high(self):
        num_eliminated = 0
        cutoffs = np.array([0.3, 0.2, 0.1])
        stats = pd.Series(np.array([0.05, 0.375, 0.125]), index=["a", "b", "c"])
        rej_hyps, num_rej_hyps = multseq.step_down_elimination(stats, cutoffs, num_eliminated, "high")
        assert num_rej_hyps == 1
        assert set(rej_hyps) == set(["b"])
        
    def test_mid_one_rej_high(self):
        num_eliminated = 1
        cutoffs = np.array([0.3, 0.2, 0.1])
        stats = pd.Series(np.array([0.05, 0.275]), index=["b", "c"])
        rej_hyps, num_rej_hyps = multseq.step_down_elimination(stats, cutoffs, num_eliminated, "high")
        assert num_rej_hyps == 1
        assert set(rej_hyps) == set(["c"])

    def test_mid_all_rej_high(self):
        num_eliminated = 1
        cutoffs = np.array([0.3, 0.2, 0.1])
        stats = pd.Series(np.array([0.15, 0.275]), index=["b", "c"])
        rej_hyps, num_rej_hyps = multseq.step_down_elimination(stats, cutoffs, num_eliminated, "high")
        assert num_rej_hyps == 2
        assert set(rej_hyps) == set(["b", "c"])



class TestMSPRT:
    def test_two_step_rejective(self):
        stats = pd.DataFrame({"x":[1.5, 3.5], "y":[1.5, 1.5]})
        cutoffs = pd.DataFrame({"A":[3.0, 2.0], })
        msprt_out = multseq.msprt(stats, cutoffs, rejective=True)
        assert msprt_out.fine_grained.rejected == {1:["x"]}
        assert msprt_out.fine_grained.remaining == ["y"]

    def test_terminating_infinite_horizon(self):
        stats = pd.DataFrame({"x":[1.5, 3.5, 2.5], "y":[1.5, -1.5, -3.5]})
        cutoffs = pd.DataFrame({"A":[3.0, 2.0], "B":[-3.0, -2.0]})
        msprt_out = multseq.msprt(stats, cutoffs, rejective=False)
        assert msprt_out.fine_grained.rejected == {1:["x"]}
        assert msprt_out.fine_grained.accepted == {2:["y"]}
        assert msprt_out.fine_grained.remaining == []

    
    def test_non_terminating_infinite_horizon(self):
        stats = pd.DataFrame({"x":[1.5, 3.5, 2.5], "y":[1.5, -1.5, -2.5]})
        cutoffs = pd.DataFrame({"A":[3.0, 2.0], "B":[-3.0, -2.0]})
        msprt_out = multseq.msprt(stats, cutoffs, rejective=False)
        assert msprt_out.fine_grained.rejected == {1:["x"]}
        assert msprt_out.fine_grained.accepted == {}
        assert msprt_out.fine_grained.remaining == ["y"]

    
    def test_bad_cutoffs_infinite_horizon(self):
        stats = pd.DataFrame({"x":[1.5, 3.5, 2.5], "y":[1.5, -1.5, -2.5]})
        cutoffs = pd.DataFrame({"A":[3.0, 2.0], "B":[-2.0, -3.0]})
        with pytest.raises(AssertionError):
            msprt_out = multseq.msprt(stats, cutoffs, rejective=False)

    def test_streaming_with_pareable(self):
        raise NotImplementedError("Test not implemented")

