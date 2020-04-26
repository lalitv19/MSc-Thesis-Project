""" This program is used for predicting the Time of Arrival (TOA in nano seconds) of the signal
 using linear regression model"""

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
np.set_printoptions(suppress=True)

# CHange the path for the dataset
dir = 'D:/Lalit/Capstone/data'

distances = io.loadmat(dir +'/distance_matrix')['distance_matrix']
locations_data = io.loadmat(dir +'/05-Feb-2019_meas_info')['meas_info'][0]

# Convert the distance into time with added delays
TOA_GT_s = distances / 299792458
TOA_GT_ns = TOA_GT_s / 1e-9
delays = 699.616+13.364+1.22
TOA_GT_ns += delays
N = 250
Fs = 61440000
Ts = 1/Fs
Ts_ns = Ts/1e-9
seq_length = (Ts_ns*N)
freq = np.arange(N) / (float(N) / seq_length)

train_points = [1, 2, 3, 4, 5, 6, 7, 9, 11,12,13, 14, 15]
test_points = [8, 10]

#Create train and test dataset from  location points
train_set = []
for loc in train_points:
    for bee in range(1, 6):
        TOA_GT_loc_bee = TOA_GT_ns[loc-1, bee-1]
        all_cc_signals = io.loadmat(dir + '/processed/Loc'+str(loc))

        for measurement in range(all_cc_signals['NanoBee_'+str(bee)].shape[0]):
            cc_signals = all_cc_signals['NanoBee_'+str(bee)][measurement]
#            data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee/seq_length]))
            data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee]))
            train_set.append(data_point)

train_set = np.array(train_set)
np.random.shuffle(train_set)
# print(train_set.shape)



test_set = []
for loc in test_points:
    for bee in range(1, 6):
        TOA_GT_loc_bee = TOA_GT_ns[loc-1, bee-1]
        all_cc_signals = io.loadmat(dir + '/processed/Loc'+str(loc))

        for measurement in range(all_cc_signals['NanoBee_'+str(bee)].shape[0]):
            cc_signals = all_cc_signals['NanoBee_'+str(bee)][measurement]
            #data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee/seq_length]))
            data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee]))
            test_set.append(data_point)

test_set = np.array(test_set)
np.random.shuffle(test_set)
# print(test_set.shape)


X_train_unscaled = train_set[:, :250]
Y_train_unscaled = train_set[:, -1]

mu_X_train = np.mean(X_train_unscaled, axis=0)
std_X_train = np.std(X_train_unscaled, axis=0)


X_train_scaled = (X_train_unscaled - mu_X_train) / std_X_train
Y_train_scaled = Y_train_unscaled

# Scale test data (use mu and std from training data!!)
X_test_unscaled = test_set[:, :250]
Y_test_unscaled = test_set[:, -1]
X_test_scaled = (X_test_unscaled - mu_X_train) / std_X_train
Y_test_scaled = Y_test_unscaled


##############################
#   Linear Regression
######################
regressor = LinearRegression()  
regressor.fit(X_train_scaled, Y_train_scaled) #training the algorithm
y_pred = regressor.predict(X_test_scaled)

error = (Y_test_scaled*seq_length) - (y_pred*seq_length)

#Plot CDF Plot for errors
fig, ax1 = plt.subplots()
ax1.set_xlabel('Deviation from actual TOA in nano seconds', fontsize=12)
#ax1.set_ylabel('Percenatage of signals', fontsize=12)
sorted_errors = np.sort(error)
Y = np.arange(0, 1, 1/len(sorted_errors))
ax1.plot(sorted_errors, Y)
ax1.set_xlim([0, 100])
plt.savefig(dir +'/plots/Linear Reg TOA for loc 8_10.pdf')
plt.show()
