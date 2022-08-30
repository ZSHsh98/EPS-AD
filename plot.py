#!/usr/bin/env python

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from scipy.stats import norm
import os
# path = 'exp_results/cifar10-robust_adv-100-eps0.031373-64x1-bm0-t0-end1e-5-cont/cifar10-wideresnet-28-10/sde_standard/seed1235/data0/score_npy.npy'
# path_adv = 'exp_results/cifar10-robust_adv-100-eps0.031373-64x1-bm0-t0-end1e-5-cont/cifar10-wideresnet-28-10/sde_standard/seed1235/data0/score_adv_npy.npy'
'''path = './image/scores_clean.npy'
path_adv = './image/scores_adv.npy'
log_dir = './image/scores_image/'
os.makedirs(log_dir, exist_ok=True)
x = np.load(path)
x_adv = np.load(path_adv)

tt = [0,1,2,3,10,30,50,70,98,99]
for t in tt:
    clean = x[t]
    adv = x_adv[t]

    num_bins = 75
    # the histogram of the data
    # n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='red', alpha=0.9)
    plt.hist(clean, num_bins, facecolor='red', density=True, alpha=0.9)
    plt.hist(adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
    plt.title(f'clean_adv_L2_denosing_iteration_{t}')
    # plt.savefig(f'clean_adv_L2_{t}.jpg')
    plt.savefig(f'{log_dir}/clean_adv_L2_t_size10_{t}.jpg')
    plt.close()'''
tt = [0.5, 5, 15, 20]
for t in tt:
    path = f'image/scores_t/scores_clean_t_{t}.npy'
    path_adv = f'image/scores_t/scores_adv_t_{t}.npy'
    log_dir = './image/scores_t/'
    os.makedirs(log_dir, exist_ok=True)
    x = np.load(path)
    x_adv = np.load(path_adv)


    clean = x[0]
    adv = x_adv[0]

    num_bins = 50
    # the histogram of the data
    # n, bins, patches = plt.hist(x, num_bins, normed=1, facecolor='red', alpha=0.9)
    plt.hist(clean, num_bins, facecolor='red', density=True, alpha=0.9)
    plt.hist(adv, num_bins,  facecolor='blue', density=True,alpha=0.9)
    plt.title(f'scores_clean_adv_t_{t}')
    # plt.savefig(f'clean_adv_L2_{t}.jpg')
    plt.savefig(f'{log_dir}/scores_clean_adv_t_{t}.jpg')
    plt.close()