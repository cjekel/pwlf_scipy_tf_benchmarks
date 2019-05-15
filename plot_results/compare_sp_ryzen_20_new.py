import numpy as np
import matplotlib.pyplot as plt

# factor for 90% coverage with 90% confidence using Normal distribution
# with 10 samples from table XII in [1]
# [1] Montgomery, D. C., & Runger, G. C. (2014). Applied statistics and
# probability for engineers. Sixth edition. John Wiley & Sons.
k = 2.535

run_times = np.load('../bench_run_times/amd_ryzen_2700x_20/6_break_times.npy')
n = np.load('../bench_run_times/amd_ryzen_2700x_20/n.npy')
run_times1 = np.load('../bench_run_times/amd_ryzen_2700x_20/new_20_break_times.npy')

run_times_means = run_times.mean(axis=2)
run_times_stds = run_times.std(axis=2, ddof=1)
run_times_means1 = run_times1.mean(axis=2)
run_times_stds1 = run_times1.std(axis=2, ddof=1)


plt.figure()
plt.title('AMD Ryzen 2700x: 20 line segments')
plt.grid()
plt.errorbar(n, run_times_means[0], yerr=k*run_times_stds[0], capsize=2.0, label='pwlf<=0.5.1')
plt.errorbar(n, run_times_means[1], yerr=k*run_times_stds[1], capsize=2.0, label='TF CPU float64')
plt.errorbar(n, run_times_means1[0], yerr=k*run_times_stds1[0], capsize=2.0, label='pwlf>=1.0.0')

plt.xlabel('Number of data points')
plt.ylabel('Run time (seconds, Lower is better)')
plt.semilogx()
plt.semilogy()
plt.legend()
plt.savefig('../figs/ryz_twenty_breaks_new.png', bbox_inches='tight')

print('TF float64 1e7 time faster', run_times_means[0][-1]/run_times_means1[0][-1])
# print('TF float32 1e7 time faster', run_times_means[0][-1]/run_times_means1[1][-1])

plt.show()