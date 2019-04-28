import numpy as np
import pwlf
from time import time
import os
# force TF to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''
breaks = np.linspace(0.0, 10.0, num=21)

n = np.logspace(3, 7, num=15, dtype=np.int)
n_repeats = 10
run_times = np.zeros((2, n.size, n_repeats))

for i, n_data in enumerate(n):
    # set random seed
    np.random.seed(256)
    # generate sin wave data
    x = np.linspace(0, 10, num=n_data)
    y = np.sin(x * np.pi / 2)
    # add noise to the data
    y = np.random.normal(0, 0.05, size=n_data) + y
    for j in range(n_repeats):
        # normal PWLF fit
        t0 = time()
        my_pwlf = pwlf.PiecewiseLinFit(x, y)
        ssr = my_pwlf.fit_with_breaks(breaks)
        t1 = time()
        # PWLF TF fit
        t2 = time()
        my_pwlf = pwlf.PiecewiseLinFitTF(x, y)
        ssr = my_pwlf.fit_with_breaks(breaks)
        t3 = time()
        run_times[0, i, j] = t1 - t0
        run_times[1, i, j] = t3 - t2

np.save('bench_run_times/6_break_times.npy', run_times)
np.save('bench_run_times/n.npy', n)
