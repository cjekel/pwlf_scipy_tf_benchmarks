import numpy as np
import tensorflow as tf
import pwlf
from time import time
import os

# force TF to use CPU
os.environ['CUDA_VISIBLE_DEVICES'] = ''

breaks = np.array((0.0, 0.94, 2.96, 4.93, 7.02, 9.04, 10.0))

n_repeats = 10
run_times = np.load('6_break_times.npy')
n = np.load('n.npy')


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
    Atf = tf.convert_to_tensor(A)
    ytf = tf.convert_to_tensor(y.reshape(-1, 1))
    for j in range(n_repeats):
        # tf.linalg.lstsq fast=True
        t4 = time()
        with tf.Session():
            beta_tf_fast = tf.linalg.lstsq(Atf, ytf, fast=True).eval()
        t5 = time()
        # tf.linalg.lstsq fast=False
        t6 = time()
        with tf.Session():
            beta_tf_fast = tf.linalg.lstsq(Atf, ytf, fast=False).eval()
        t7 = time()
        run_times[2, i, j] = t5 - t4
        run_times[3, i, j] = t7 - t6
    break
np.save('6_break_times.npy', run_times)
np.save('n.npy', n)
