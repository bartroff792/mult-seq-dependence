# -*- coding: utf-8 -*-

import numpy as np


def chunk_mc(k_reps: int, num_jobs: int) -> list[int]:
    """Divide tasks into chunks for parallel computing.

    Args:
        k_reps (int): number of total tasks.
        num_jobs (int): number of parallel jobs.

    Returns:
        list[int]: list of number of tasks for each job.
    """
    if k_reps <  2 * num_jobs:
        n_rep_list = [2,] * int(np.floor(k_reps/2))
        if k_reps % 2==1:
            n_rep_list = n_rep_list  + [1,]
    else:
        chunksize = int(np.floor(k_reps / num_jobs))
        n_rep_list = [chunksize, ] * num_jobs
        if chunksize * num_jobs < k_reps:
            n_rep_list = n_rep_list  + [k_reps - chunksize * num_jobs,]
    return n_rep_list