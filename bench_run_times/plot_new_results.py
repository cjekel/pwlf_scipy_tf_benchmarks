import numpy as np
import matplotlib.pyplot as plt

# factor for 90% coverage with 90% confidence using Normal distribution
# with 10 samples from table XII in [1]
# [1] Montgomery, D. C., & Runger, G. C. (2014). Applied statistics and
# probability for engineers. Sixth edition. John Wiley & Sons.
k = 2.535


run_strs = 'amd_ryzen_2700x_20/6_break_times.npy'
new_run_strs = 'amd_ryzen_2700x_20/new_20_break_times.npy'
n_strs = 'amd_ryzen_2700x/n.npy'
j = 1
run_times = np.load(run_strs)
new_runs = np.load(new_run_strs)
n = np.load(n_strs)

run_times_means = run_times.mean(axis=2)
run_times_stds = run_times.std(axis=2, ddof=1)
new_times_means = new_runs.mean(axis=2)
new_times_stds = new_runs.std(axis=2, ddof=1)

plt.figure()
plt.grid()
plt.errorbar(n, run_times_means[0], yerr=k*run_times_stds[0], capsize=2.0, label='Numpy')
# plt.errorbar(n, run_times_means[1], yerr=k*run_times_stds[1], capsize=2.0, label='TF GPU')
plt.errorbar(n, run_times_means[j], yerr=k*run_times_stds[j], capsize=2.0, label='TF CPU')
plt.errorbar(n, new_times_means[0], yerr=k*new_times_stds[0], capsize=2.0, label='New PWLF')

plt.xlabel('Number of data points')
plt.ylabel('Run time (seconds, Lower is better)')
plt.semilogx()
plt.semilogy()
plt.legend()


plt.show()
