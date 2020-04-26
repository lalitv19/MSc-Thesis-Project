""" This program is used for finding the peak of the first arrival path ( Time of arrival)
by using manual method"""

import numpy as np
import pandas as pd
import scipy.io as io
import matplotlib.pyplot as plt
import os
from copy import deepcopy

dir = os.getcwd()

# Load all the receiver data  - 
locations_data = io.loadmat(os.getcwd() +'/data/05-Feb-2019_meas_info')['meas_info'][0]
measurement_locs = np.zeros((15, 3))
for measurement_point in range(15):
    measurement_locs[measurement_point, :] = locations_data[measurement_point][0]
reference = deepcopy(measurement_locs[8])
measurement_locs -= reference


bee_location_data = io.loadmat(os.getcwd() +'/data/05-Feb-2019_ant_info')['ant_info'][0]
bee_locs = np.zeros((5, 3))
for bee_id in range(5):
    bee_locs[bee_id, :] = bee_location_data['loc'][bee_id][0][0][5].reshape(1, 3)[0]
bee_locs -= reference

# Convert the distance into time with added delays
distances = io.loadmat(os.getcwd()+'/data/distance_matrix')['distance_matrix']
TOA_GT_s = distances / 299792458
TOA_GT_ns = TOA_GT_s / 1e-9
delays = 699.616+13.364+1.22
TOA_GT_ns = TOA_GT_ns + delays
N = 250
Fs = 61440000
Ts = 1/Fs
Ts_ns = Ts/1e-9
seq_length = (Ts_ns*N)
freq = np.arange(N) / (float(N) / seq_length)

"""Algorithm used for finding the peak 
1) Find the maximum value of the sample in the signal range.
2) Find all the peaks that are 7 dB below this value.
    * If there is only one peak, then this is the 
rst arriving path.
    * If the fi
rst peak has a second peak that is within 3 samples and if the second
      peak is more than 4dB lower than the fi
rst, then take the second.
    * Or else, take the fi
rst peak."""

data_points = []
for loc in range(1, 16):
    all_cc_signals = io.loadmat(os.getcwd()+'/data/processed/Loc'+str(loc))
    cc_signals = all_cc_signals['NanoBee_'+str(1)]

    num_measurements = cc_signals.shape[0]

    for measurement in range(num_measurements):
        bee_1 = all_cc_signals['NanoBee_1'][measurement]
        bee_2 = all_cc_signals['NanoBee_2'][measurement]
        bee_3 = all_cc_signals['NanoBee_3'][measurement]
        bee_4 = all_cc_signals['NanoBee_4'][measurement]
        bee_5 = all_cc_signals['NanoBee_5'][measurement]

        TOA_GT_loc_bee_1 = TOA_GT_ns[loc-1, 1-1]
        TOA_GT_loc_bee_2 = TOA_GT_ns[loc-1, 2-1]
        TOA_GT_loc_bee_3 = TOA_GT_ns[loc-1, 3-1]
        TOA_GT_loc_bee_4 = TOA_GT_ns[loc-1, 4-1]
        TOA_GT_loc_bee_5 = TOA_GT_ns[loc-1, 5-1]

        peak_cc = bee_1[np.argmax(bee_1)]
        filtered_cc = deepcopy(bee_1)
        filtered_cc[filtered_cc < peak_cc-7] = 0
        pridicted_TOA_loc_bee_1 = freq[np.nonzero(filtered_cc)[0][0]]

        peak_cc = bee_2[np.argmax(bee_2)]
        filtered_cc = deepcopy(bee_2)
        filtered_cc[filtered_cc < peak_cc-7] = 0
        pridicted_TOA_loc_bee_2 = freq[np.nonzero(filtered_cc)[0][0]]

        peak_cc = bee_3[np.argmax(bee_3)]
        filtered_cc = deepcopy(bee_3)
        filtered_cc[filtered_cc < peak_cc-7] = 0
        pridicted_TOA_loc_bee_3 = freq[np.nonzero(filtered_cc)[0][0]]

        peak_cc = bee_4[np.argmax(bee_4)]
        filtered_cc = deepcopy(bee_4)
        filtered_cc[filtered_cc < peak_cc-7] = 0
        pridicted_TOA_loc_bee_4 = freq[np.nonzero(filtered_cc)[0][0]]

        peak_cc = bee_5[np.argmax(bee_5)]
        filtered_cc = deepcopy(bee_5)
        filtered_cc[filtered_cc < peak_cc-7] = 0
        pridicted_TOA_loc_bee_5 = freq[np.nonzero(filtered_cc)[0][0]]

        GT = np.array([TOA_GT_loc_bee_1, TOA_GT_loc_bee_2, TOA_GT_loc_bee_3, TOA_GT_loc_bee_4, TOA_GT_loc_bee_5])
        predicted = np.array([pridicted_TOA_loc_bee_1, pridicted_TOA_loc_bee_2, pridicted_TOA_loc_bee_3, pridicted_TOA_loc_bee_4, pridicted_TOA_loc_bee_5])
        location = measurement_locs[loc-1]

        data_point = np.concatenate((predicted, location))
        # data_point = np.concatenate((GT, location))
        data_points.append(data_point)


data = {}
data['test_points'] = data_points
io.savemat(os.getcwd()+'/data/test_points.mat', data)

