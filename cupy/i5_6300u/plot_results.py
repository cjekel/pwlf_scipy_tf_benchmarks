import numpy as np
import matplotlib.pyplot as plt

# factor for 90% coverage with 90% confidence using Normal distribution
# with 10 samples from table XII in [1]
# [1] Montgomery, D. C., & Runger, G. C. (2014). Applied statistics and
# probability for engineers. Sixth edition. John Wiley & Sons.
k = 2.535

run_times = np.load('6_break_times.npy')
n = np.load('n.npy')
run_times1 = np.load('20_break_times.npy')

run_times_means = run_times.mean(axis=2)
run_times_stds = run_times.std(axis=2, ddof=1)
run_times_means1 = run_times1.mean(axis=2)
run_times_stds1 = run_times1.std(axis=2, ddof=1)


plt.figure()
plt.title('i5-6300u: 6 line segments')
plt.grid()
plt.errorbar(n, run_times_means[0], yerr=k*run_times_stds[0], capsize=2.0,
             label='npumpy.linalg.lstsq')
# plt.errorbar(n, run_times_means[1], yerr=k*run_times_stds[1], capsize=2.0,
#              label='cupy.linalg.lstsq')
plt.errorbar(n, run_times_means[2], yerr=k*run_times_stds[2], capsize=2.0,
             label='tf.linalg.lstsq (CPU)')
# plt.errorbar(n, run_times_means[3], yerr=k*run_times_stds[3], capsize=2.0,
#              label='tf.linalg.lstsq fast=False (CPU)')


plt.xlabel('Number of data points')
plt.ylabel('Run time (seconds, Lower is better)')
plt.semilogx()
plt.semilogy()
plt.legend()
plt.savefig('i5_six_breaks.png', bbox_inches='tight')

# print('cupy 1e7 time faster', run_times_means[0][-1]/run_times_means[1][-1])

plt.figure()
plt.title('i5-6300u: 20 line segments')
plt.grid()
plt.errorbar(n, run_times_means1[0], yerr=k*run_times_stds1[0], capsize=2.0,\
             label='npumpy.linalg.lstsq')
# plt.errorbar(n, run_times_means1[1], yerr=k*run_times_stds1[1], capsize=2.0,
#              label='cupy.linalg.lstsq')
plt.errorbar(n, run_times_means1[2], yerr=k*run_times_stds1[2], capsize=2.0,
             label='tf.linalg.lstsq (CPU)')
# plt.errorbar(n, run_times_means1[3], yerr=k*run_times_stds1[3], capsize=2.0,
#              label='tf.linalg.lstsq fast=False (CPU)')


plt.xlabel('Number of data points')
plt.ylabel('Run time (seconds, Lower is better)')
plt.semilogx()
plt.semilogy()
plt.legend()
plt.savefig('i5_twenty_breaks.png', bbox_inches='tight')

# print('cupy 1e7 time faster', run_times_means1[0][-1]/run_times_means1[1][-1])

plt.show()

