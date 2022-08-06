import matplotlib.pyplot as plt
import numpy as np
import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
from newresnet import resnet101_blockn_split
import sys


_BASELINE_ACC1_CIFAR100 = 78.820
_BASELINE_ACC5_CIFAR100 = 94.560

_BASELINE_ACC1_IMAGENETTE = 90.242
_BASELINE_ACC5_IMAGENETTE = 99.083

_QBINS = np.arange(2, 31)

## split 1
a1s_s1_nonoise_cif = np.array([12.050, 56.550, 71.610, 74.510, 76.030, 77.050, 77.610, 77.980,
                               77.940, 77.990, 78.190, 78.310, 78.430, 78.390, 78.510, 78.710,
                               78.560, 78.750, 78.720, 78.720, 78.590, 78.650, 78.770, 78.590,
                               78.740, 78.570, 78.700, 78.810, 78.740])
a5s_s1_nonoise_cif = np.array([28.250, 78.620, 90.360, 92.760, 93.290, 93.890, 94.330, 94.510,
                               94.480, 94.570, 94.560, 94.550, 94.620, 94.650, 94.660, 94.570,
                               94.600, 94.580, 94.610, 94.550, 94.640, 94.520, 94.590, 94.700,
                               94.680, 94.600, 94.640, 94.580, 94.570])

a1s_s1_noise_cif = np.array([12.910, 65.050, 73.330, 74.390, 74.900, 75.620, 75.580, 75.720,
                             75.590, 75.800, 75.990, 76.080, 76.020, 75.770, 76.020, 75.960,
                             76.030, 75.910, 76.100, 76.040, 75.860, 75.900, 75.870, 75.910,
                             75.930, 75.960, 75.900, 75.910, 76.020])
a5s_s1_noise_cif = np.array([28.880, 87.610, 92.600, 93.210, 93.590, 93.770, 94.050, 94.010,
                             94.090, 93.980, 94.040, 94.120, 94.060, 94.060, 94.110, 94.090,
                             94.130, 94.020, 94.050, 94.130, 93.980, 94.050, 94.050, 94.100,
                             94.110, 94.050, 94.050, 94.070, 94.040])

a1s_s1_nonoise_imgnette = np.array([10.522, 34.548, 68.229, 82.777, 87.338, 88.561, 89.325,
                                    89.962, 89.656, 89.911, 89.936, 90.064, 90.191, 90.191,
                                    90.242, 90.344, 90.344, 90.191, 90.191, 90.191, 90.242,
                                    90.344, 90.191, 90.191, 90.242, 90.191, 90.217, 90.293,
                                    90.268])
a5s_s1_nonoise_imgnette = np.array([48.229, 74.573, 94.268, 97.783, 98.548, 98.854, 98.981,
                                    99.108, 99.083, 99.134, 99.083, 99.057, 99.057, 99.108,
                                    99.057, 99.057, 99.108, 99.032, 99.083, 99.108, 99.032,
                                    99.083, 99.083, 99.083, 99.032, 99.057, 99.032, 99.032,
                                    99.032])

a1s_s1_noise_imgnette = np.array([9.885, 58.955, 84.917, 87.822, 87.873, 87.847, 88.153,
                                  88.025, 87.847, 88.102, 88.229, 88.076, 88.153, 88.127,
                                  88.076, 88.127, 88.025, 88.025, 87.924, 88.025, 88.178,
                                  88.102, 88.204, 88.153, 88.051, 88.102, 88.102, 88.102,
                                  88.153])
a5s_s1_noise_imgnette = np.array([52.535, 87.465, 98.293, 98.904, 98.930, 98.904, 98.904,
                                  98.930, 98.930, 98.803, 98.828, 98.854, 98.879, 98.879,
                                  98.879, 98.854, 98.854, 98.828, 98.879, 98.904, 98.879,
                                  98.879, 98.904, 98.828, 98.854, 98.854, 98.879, 98.854,
                                  98.854])


## split 2
a1s_s2_nonoise_cif = np.array([12.310, 57.140, 70.480, 74.200, 75.670, 76.920, 77.430, 77.720,
                               77.950, 77.680, 78.010, 78.310, 78.380, 78.260, 78.540, 78.560,
                               78.670, 78.570, 78.670, 78.570, 78.570, 78.720, 78.810, 78.650,
                               78.760, 78.630, 78.760, 78.810, 78.700])
a5s_s2_nonoise_cif = np.array([27.870, 79.520, 89.950, 92.420, 93.160, 93.700, 94.190, 94.300,
                               94.240, 94.540, 94.620, 94.540, 94.660, 94.590, 94.560, 94.580,
                               94.610, 94.600, 94.600, 94.600, 94.640, 94.650, 94.620, 94.710,
                               94.530, 94.610, 94.670, 94.620, 94.610])

a1s_s2_noise_cif = np.array([12.550, 64.180, 72.420, 73.810, 74.610, 74.820, 75.120, 75.110,
                             75.340, 75.360, 75.500, 75.580, 75.420, 75.430, 75.660, 75.510,
                             75.560, 75.570, 75.460, 75.460, 75.540, 75.530, 75.440, 75.650,
                             75.460, 75.600, 75.390, 75.570, 75.650])
a5s_s2_noise_cif = np.array([28.820, 87.340, 92.420, 93.070, 93.430, 93.640, 93.820, 93.880,
                             93.860, 93.930, 93.930, 93.920, 93.840, 93.920, 93.980, 93.900,
                             93.980, 93.990, 93.960, 93.990, 93.950, 93.980, 93.930, 94.000,
                             93.920, 93.960, 93.980, 93.920, 94.020])

a1s_s2_noise_special_cif = np.array([10.680, 59.340, 71.930, 73.870, 74.700, 75.210, 75.230,
                                     75.360, 75.170, 75.390, 75.580, 75.520, 75.620, 75.550,
                                     75.550, 75.440, 75.650, 75.850, 75.700, 75.620, 75.600,
                                     75.600, 75.540, 75.640, 75.700, 75.720, 75.660, 75.760,
                                     75.650])
a5s_s2_noise_special_cif = np.array([24.600, 83.940, 91.680, 93.130, 93.420, 93.590, 93.710,
                                     93.680, 93.680, 93.670, 93.690, 93.720, 93.780, 93.750,
                                     93.710, 93.760, 93.810, 93.810, 93.730, 93.750, 93.800,
                                     93.790, 93.770, 93.810, 93.700, 93.710, 93.760, 93.830,
                                     93.770])

a1s_s2_nonoise_imgnette = np.array([10.166, 32.561, 66.471, 82.675, 87.312, 89.019, 89.401,
                                    89.732, 90.038, 89.885, 89.834, 89.885, 90.089, 90.318,
                                    90.115, 90.166, 90.013, 90.140, 90.089, 90.166, 90.217,
                                    90.293, 90.242, 90.318, 90.268, 90.140, 90.242, 90.369,
                                    90.318])
a5s_s2_nonoise_imgnette = np.array([48.102, 72.866, 93.834, 97.707, 98.624, 98.904, 99.032,
                                    99.083, 99.032, 99.032, 99.057, 99.057, 98.955, 99.083,
                                    99.134, 99.083, 99.006, 99.159, 99.108, 99.108, 99.032,
                                    99.057, 99.083, 99.057, 99.057, 99.032, 99.057, 99.057,
                                    99.108])

a1s_s2_noise_imgnette = np.array([10.166, 56.994, 82.879, 86.318, 86.395, 86.955, 87.210,
                                  87.287, 87.516, 87.389, 87.465, 87.745, 87.618, 87.541,
                                  87.363, 87.567, 87.567, 87.465, 87.465, 87.618, 87.618,
                                  87.490, 87.567, 87.414, 87.720, 87.592, 87.541, 87.490,
                                  87.567])
a5s_s2_noise_imgnette = np.array([52.408, 86.420, 97.758, 98.548, 98.675, 98.777, 98.777,
                                  98.777, 98.879, 98.879, 98.803, 98.879, 98.854, 98.879,
                                  98.879, 98.828, 98.803, 98.828, 98.803, 98.879, 98.803,
                                  98.828, 98.854, 98.803, 98.854, 98.854, 98.904, 98.752,
                                  98.803])


## split 3
a1s_s3_nonoise_cif = np.array([6.970, 49.720, 67.760, 73.100, 75.400, 76.600, 77.380, 77.300,
                               77.660, 77.740, 77.800, 78.150, 78.270, 78.490, 78.530, 78.560,
                               78.610, 78.560, 78.610, 78.590, 78.500, 78.660, 78.580, 78.550,
                               78.730, 78.660, 78.570, 78.590, 78.690])
a5s_s3_nonoise_cif = np.array([18.670, 75.080, 88.070, 91.410, 92.790, 93.500, 93.980, 94.220,
                               94.210, 94.340, 94.370, 94.520, 94.580, 94.550, 94.600, 94.500,
                               94.480, 94.570, 94.630, 94.550, 94.670, 94.710, 94.600, 94.660,
                               94.560, 94.610, 94.560, 94.610, 94.670])

a1s_s3_noise_cif = np.array([12.640, 63.610, 71.860, 72.880, 74.070, 74.020, 74.150, 74.270,
                             74.380, 74.360, 74.590, 74.700, 74.880, 74.520, 74.470, 74.360,
                             74.640, 74.670, 74.650, 74.460, 74.520, 74.460, 74.820, 74.460,
                             74.710, 74.560, 74.720, 74.430, 74.870])
a5s_s3_noise_cif = np.array([28.440, 86.890, 91.990, 92.590, 93.070, 93.270, 93.460, 93.630,
                             93.670, 93.550, 93.530, 93.520, 93.550, 93.570, 93.500, 93.630,
                             93.560, 93.560, 93.610, 93.560, 93.520, 93.400, 93.570, 93.510,
                             93.520, 93.500, 93.480, 93.580, 93.630])

a1s_s3_nonoise_imgnette = np.array([9.962, 30.013, 63.720, 81.682, 86.752, 88.484, 89.146,
                                    89.350, 89.631, 89.682, 89.631, 89.682, 89.783, 89.783,
                                    90.140, 90.064, 90.013, 89.911, 89.911, 89.962, 89.962,
                                    90.064, 90.038, 90.166, 90.191, 90.191, 90.140, 90.089,
                                    90.140])
a5s_s3_nonoise_imgnette = np.array([48.535, 69.554, 93.146, 97.478, 98.573, 98.955, 99.057,
                                    99.057, 99.032, 99.032, 99.032, 98.981, 98.981, 99.006,
                                    99.032, 98.955, 99.006, 98.981, 99.006, 99.006, 98.981,
                                    99.032, 99.032, 99.006, 99.083, 99.083, 99.057, 99.083,
                                    99.083])

a1s_s3_noise_imgnette = np.array([9.834, 56.204, 82.497, 86.140, 86.395, 86.955, 87.669,
                                  87.363, 87.363, 87.389, 87.312, 87.236, 87.924, 87.873,
                                  87.567, 87.720, 87.516, 87.236, 87.822, 87.592, 87.287,
                                  87.439, 87.567, 87.465, 87.822, 87.541, 88.051, 88.102,
                                  87.847])
a5s_s3_noise_imgnette = np.array([52.280, 84.790, 97.732, 98.522, 98.726, 98.752, 98.701,
                                  98.828, 98.828, 98.726, 98.726, 98.726, 98.752, 98.777,
                                  98.752, 98.803, 98.726, 98.752, 98.752, 98.803, 98.752,
                                  98.752, 98.752, 98.777, 98.854, 98.803, 98.701, 98.752,
                                  98.777])


a1s_s1_noise_companding_imgnette = np.array([21.580, 49.834, 58.599, 52.586, 45.885, 41.809,
                                             40.000, 38.471, 37.911, 37.223, 36.968, 36.688,
                                             36.459, 36.459, 36.459, 36.484, 36.459, 36.637,
                                             36.968, 36.943, 37.070, 36.994, 37.121, 37.121,
                                             37.223, 37.299, 37.401, 37.605, 37.605])
a5s_s1_noise_companding_imgnette = np.array([64.764, 85.732, 91.490, 86.191, 79.108, 74.293,
                                             70.573, 68.637, 66.879, 66.369, 65.834, 65.376,
                                             65.248, 65.070, 64.994, 65.096, 65.121, 65.096,
                                             65.197, 65.248, 65.427, 65.503, 65.427, 65.656,
                                             65.783, 65.911, 66.140, 66.293, 66.344])

a1s_s1_nonoise_companding_imgnette = np.array([])
a5s_s1_nonoise_companding_imgnette = np.array([])



if __name__ == "__main__":

    # Split comparison CIFAR100 with noise training
    _, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[0].set_yticks(np.arange(0, 101, 10))

    ax[0].plot(_QBINS, a1s_s1_noise_cif, color='royalblue', linewidth='1', label='split@1')
    ax[0].plot(_QBINS, a1s_s2_noise_cif, color='seagreen', linewidth='1', label='split@2')
    ax[0].plot(_QBINS, a1s_s3_noise_cif, color='firebrick', linewidth='1', label='split@3')
    ax[0].axhline(y=_BASELINE_ACC1_CIFAR100, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[0].legend(loc=4, fontsize=12)

    ax[0].set_ylim(top=100)
    ax[0].set_title("Acc@1", fontsize=14)

    ax[1].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[1].set_yticks(np.arange(0, 101, 10))

    ax[1].plot(_QBINS, a5s_s1_noise_cif, color='royalblue', linewidth='1', label='split@1')
    ax[1].plot(_QBINS, a5s_s2_noise_cif, color='seagreen', linewidth='1', label='split@2')
    ax[1].plot(_QBINS, a5s_s3_noise_cif, color='firebrick', linewidth='1', label='split@3')
    ax[1].axhline(y=_BASELINE_ACC5_CIFAR100, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[1].legend(loc=4, fontsize=12)

    ax[1].set_ylim(top=100)
    ax[1].set_title("Acc@5", fontsize=14)

    plt.tight_layout()
    plt.savefig("results/graphs/cif100_w_noise")
    plt.show()


    # Split comparison CIFAR100 w/o noise training
    _, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[0].set_yticks(np.arange(0, 101, 10))

    ax[0].plot(_QBINS, a1s_s1_nonoise_cif, color='royalblue', linewidth='1', label='split@1')
    ax[0].plot(_QBINS, a1s_s2_nonoise_cif, color='seagreen', linewidth='1', label='split@2')
    ax[0].plot(_QBINS, a1s_s3_nonoise_cif, color='firebrick', linewidth='1', label='split@3')
    ax[0].axhline(y=_BASELINE_ACC1_CIFAR100, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[0].legend(loc=4, fontsize=12)

    ax[0].set_ylim(top=100)
    ax[0].set_title("Acc@1", fontsize=14)

    ax[1].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[1].set_yticks(np.arange(0, 101, 10))

    ax[1].plot(_QBINS, a5s_s1_nonoise_cif, color='royalblue', linewidth='1', label='split@1')
    ax[1].plot(_QBINS, a5s_s2_nonoise_cif, color='seagreen', linewidth='1', label='split@2')
    ax[1].plot(_QBINS, a5s_s3_nonoise_cif, color='firebrick', linewidth='1', label='split@3')
    ax[1].axhline(y=_BASELINE_ACC5_CIFAR100, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[1].legend(loc=4, fontsize=12)

    ax[1].set_ylim(top=100)
    ax[1].set_title("Acc@5", fontsize=14)

    plt.tight_layout()
    plt.savefig("results/graphs/cif100_wo_noise")
    plt.show()


    # Split comparison Imagenette with noise training
    _, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[0].set_yticks(np.arange(0, 101, 10))

    ax[0].plot(_QBINS, a1s_s1_noise_imgnette, color='royalblue', linewidth='1', label='split@1')
    ax[0].plot(_QBINS, a1s_s2_noise_imgnette, color='seagreen', linewidth='1', label='split@2')
    ax[0].plot(_QBINS, a1s_s3_noise_imgnette, color='firebrick', linewidth='1', label='split@3')
    ax[0].axhline(y=_BASELINE_ACC1_IMAGENETTE, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[0].legend(loc=4, fontsize=12)

    ax[0].set_ylim(top=100)
    ax[0].set_title("Acc@1", fontsize=14)

    ax[1].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[1].set_yticks(np.arange(0, 101, 10))

    ax[1].plot(_QBINS, a5s_s1_noise_imgnette, color='royalblue', linewidth='1', label='split@1')
    ax[1].plot(_QBINS, a5s_s2_noise_imgnette, color='seagreen', linewidth='1', label='split@2')
    ax[1].plot(_QBINS, a5s_s3_noise_imgnette, color='firebrick', linewidth='1', label='split@3')
    ax[1].axhline(y=_BASELINE_ACC5_IMAGENETTE, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[1].legend(loc=4, fontsize=12)

    ax[1].set_ylim(top=100)
    ax[1].set_title("Acc@5", fontsize=14)

    plt.tight_layout()
    plt.savefig("results/graphs/imgnette_w_noise")
    plt.show()

    # Split comparison Imagenette w/o noise training
    _, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[0].set_yticks(np.arange(0, 101, 10))

    ax[0].plot(_QBINS, a1s_s1_nonoise_imgnette, color='royalblue', linewidth='1', label='split@1')
    ax[0].plot(_QBINS, a1s_s2_nonoise_imgnette, color='seagreen', linewidth='1', label='split@2')
    ax[0].plot(_QBINS, a1s_s3_nonoise_imgnette, color='firebrick', linewidth='1', label='split@3')
    ax[0].axhline(y=_BASELINE_ACC1_IMAGENETTE, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[0].legend(loc=4, fontsize=12)

    ax[0].set_ylim(top=100)
    ax[0].set_title("Acc@1", fontsize=14)

    ax[1].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[1].set_yticks(np.arange(0, 101, 10))

    ax[1].plot(_QBINS, a5s_s1_nonoise_imgnette, color='royalblue', linewidth='1', label='split@1')
    ax[1].plot(_QBINS, a5s_s2_nonoise_imgnette, color='seagreen', linewidth='1', label='split@2')
    ax[1].plot(_QBINS, a5s_s3_nonoise_imgnette, color='firebrick', linewidth='1', label='split@3')
    ax[1].axhline(y=_BASELINE_ACC5_IMAGENETTE, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[1].legend(loc=4, fontsize=12)

    ax[1].set_ylim(top=100)
    ax[1].set_title("Acc@5", fontsize=14)

    plt.tight_layout()
    plt.savefig("results/graphs/imgnette_wo_noise")
    plt.show()


    # Training for different splits comparison
    _, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[0].set_yticks(np.arange(0, 101, 10))

    ax[0].plot(_QBINS, a1s_s1_noise_cif, color='royalblue', linewidth='1', label='split@1')
    ax[0].plot(_QBINS, a1s_s2_noise_cif, color='seagreen', linewidth='1', label='split@2')
    ax[0].plot(_QBINS, a1s_s2_noise_special_cif, color='firebrick', linewidth='1', label='split@2"')
    ax[0].axhline(y=_BASELINE_ACC1_CIFAR100, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[0].legend(loc=4, fontsize=12)

    ax[0].set_ylim(top=100)
    ax[0].set_title("Acc@1", fontsize=14)

    ax[1].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[1].set_yticks(np.arange(0, 101, 10))

    ax[1].plot(_QBINS, a5s_s1_noise_cif, color='royalblue', linewidth='1', label='split@1')
    ax[1].plot(_QBINS, a5s_s2_noise_cif, color='seagreen', linewidth='1', label='split@2')
    ax[1].plot(_QBINS, a5s_s2_noise_special_cif, color='firebrick', linewidth='1', label='split@2"')
    ax[1].axhline(y=_BASELINE_ACC5_CIFAR100, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[1].legend(loc=4, fontsize=12)

    ax[1].set_ylim(top=100)
    ax[1].set_title("Acc@5", fontsize=14)

    plt.tight_layout()
    plt.savefig("results/graphs/training_regime_compare")
    plt.show()


    # Companding
    _, ax = plt.subplots(2, 1, figsize=(8, 8))

    ax[0].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[0].set_yticks(np.arange(0, 101, 10))

    ax[0].plot(_QBINS, a1s_s1_noise_imgnette, color='royalblue', linewidth='1', label='uniform')
    ax[0].plot(_QBINS, a1s_s1_noise_companding_imgnette, color='seagreen', linewidth='1', label='companding')
    ax[0].axhline(y=_BASELINE_ACC1_IMAGENETTE, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[0].legend(loc=4, fontsize=12)

    ax[0].set_ylim(top=100)
    ax[0].set_title("Acc@1", fontsize=14)

    ax[1].set_xticks(np.arange(min(_QBINS), max(_QBINS) + 1, 2))
    ax[1].set_yticks(np.arange(0, 101, 10))

    ax[1].plot(_QBINS, a5s_s1_noise_imgnette, color='royalblue', linewidth='1', label='uniform')
    ax[1].plot(_QBINS, a5s_s1_noise_companding_imgnette, color='seagreen', linewidth='1', label='companding')
    ax[1].axhline(y=_BASELINE_ACC5_IMAGENETTE, xmin=0.03, xmax=0.97, color='k', linestyle='--', linewidth='1',
                  label='baseline')

    ax[1].legend(loc=4, fontsize=12)

    ax[1].set_ylim(top=100)
    ax[1].set_title("Acc@5", fontsize=14)

    plt.tight_layout()
    plt.savefig("results/graphs/companding_compare")
    plt.show()

    # FLOPS

    original_stdout = sys.stdout  # Save a reference to the original standard output

    with open('results/FLOPS_count.txt', 'w') as f:
        sys.stdout = f  # Change the standard output to the file we created.
        with torch.cuda.device(0):
            net = models.resnet101()
            edge, cloud = resnet101_blockn_split(1, num_classes=10, pretrained=False, cifar=False)

            macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
                                                     print_per_layer_stat=False, verbose=False)

            print('ResNet101 (Full) - Computational complexity for Imagenet')
            print('-' * 50)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            print("\n\n")

            macs, params = get_model_complexity_info(edge, (3, 224, 224), as_strings=True,
                                                     print_per_layer_stat=False, verbose=False)

            print('ResNet101 (Edge) - Computational complexity for Imagenet')
            print('-' * 50)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            print("\n\n")

            macs, params = get_model_complexity_info(cloud, (256, 56, 56), as_strings=True,
                                                     print_per_layer_stat=False, verbose=False)

            print('ResNet101 (Cloud) - Computational complexity for Imagenet')
            print('-' * 50)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            print("\n\n")
            sys.stdout = original_stdout  # Reset the standard output to its original value
