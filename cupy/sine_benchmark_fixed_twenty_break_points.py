import cupy as cp
import numpy as np
import pwlf
from time import time
import os
breaks = np.linspace(0.0, 10.0, num=21)

n = np.logspace(3, 6.8, num=15, dtype=np.int)
n_repeats = 10
run_times = np.zeros((4, n.size, n_repeats))

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


def generate_matrix(n_data):
    # set random seed
    np.random.seed(256)
    # generate sin wave data
    x = np.linspace(0, 10, num=n_data)
    y = np.sin(x * np.pi / 2)
    # add noise to the data
    y = np.random.normal(0, 0.05, size=n_data) + y
    my_pwlf = pwlf.PiecewiseLinFit(x, y)
    A = my_pwlf.assemble_regression_matrix(breaks, my_pwlf.x_data)
    return A, y


for i, n_data in enumerate(n):
    A, y = generate_matrix(n_data)
    for j in range(n_repeats):
        Acp = cp.asarray(A)
        ycp = cp.asarray(y)
        # numpy.linalg.lstsq
        t0 = time()
        beta_np, _, _, _ = np.linalg.lstsq(A, y, rcond=1e-15)
        t1 = time()
        # cupy.linalg.lstsq
        t2 = time()
        beta_cp = cp.linalg.lstsq(Acp, ycp)
        t3 = time()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        run_times[0, i, j] = t1 - t0
        run_times[1, i, j] = t3 - t2

np.save('20_break_times.npy', run_times)
np.save('n.npy', n)
