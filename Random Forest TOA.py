""" This program is used for predicting the Time of Arrival (TOA in nano seconds) of the signal
 using random forrest regressor model"""

import numpy as np
import pandas as pd
import scipy.io as io
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)


dir = 'F:/MS BA/Capstone/Nokia Bell Labs/data'
distances = io.loadmat(dir +'/distance_matrix')['distance_matrix']
locations_data = io.loadmat(dir +'/05-Feb-2019_meas_info')['meas_info'][0]

# Convert the distance into time with added delays

TOA_GT_s = distances / 299792458
TOA_GT_ns = TOA_GT_s / 1e-9
delays = 699.616+13.364+1.22
TOA_GT_ns += delays
TOA_GT_ns_Scaled = TOA_GT_ns - np.mean(TOA_GT_ns)


train_points = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # train with random 11 location data points
dev_points = [12, 13] # Use two location data points for model valodation

# test_points = [3, 4]
test_points = [14, 15] # Use two location data points for model test

################################
train_set = pd.DataFrame()

for loc in train_points:
    all_cc_signals =  io.loadmat(dir + '/Unprocessed/Loc'+str(loc))
    
    for i in range(5):
        cc_signals = pd.DataFrame(all_cc_signals['NanoBee_'+str(i+1)])  # nanbee start from 1
        toa = pd.DataFrame([pd.DataFrame(TOA_GT_ns)[i][loc-1]] * len(cc_signals)) # dataframe indexing starts form 0 for nanbee
        data_point = pd.concat([cc_signals, toa], axis=1)
        train_set = train_set.append(cc_signals, ignore_index = True)

np.random.shuffle(train_set.values)

######## dev set data preparation
dev_set = pd.DataFrame()

for loc in dev_points:
    all_cc_signals =  io.loadmat(dir + '/processed/Loc'+str(loc))
    
    for i in range(5):
        cc_signals = pd.DataFrame(all_cc_signals['NanoBee_'+str(i+1)])  # nanbee start from 1
        toa = pd.DataFrame([pd.DataFrame(TOA_GT_ns)[i][loc-1]] * len(cc_signals)) # dataframe indexing starts form 0 for nanbee
        data_point = pd.concat([cc_signals, toa], axis=1)
        dev_set = dev_set.append(data_point, ignore_index = True)

np.random.shuffle(dev_set.values)

####### test set data 
test_set = pd.DataFrame()

for loc in test_points:
    all_cc_signals =  io.loadmat(dir + '/processed/Loc'+str(loc))
    
    for i in range(5):
        cc_signals = pd.DataFrame(all_cc_signals['NanoBee_'+str(i+1)])  # nanbee start from 1
        toa = pd.DataFrame([pd.DataFrame(TOA_GT_ns)[i][loc-1]] * len(cc_signals)) # dataframe indexing starts form 0 for nanbee
        data_point = pd.concat([cc_signals, toa], axis=1)
        test_set = test_set.append(data_point, ignore_index = True)

np.random.shuffle(test_set.values)

#########################################################################
#           Randowm Forrest Regressor
########################################################################

# Data preprocessing
# Scale training data
train_x = train_set.iloc[:, :250]
train_y = train_set.iloc[:, 250:]


# Scale dev data (use mu and std from training data!!)
dev_x = dev_set.iloc[:, :250]
dev_y = dev_set.iloc[:, 250:]

# Scale test data (use mu and std from training data!!)
test_x = test_set.iloc[:, :250]
test_y = test_set.iloc[:, 250:]

print("Data loaded")

#######################################
# Import the model we are using

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# Instantiate model with 1000 decision trees
model = DecisionTreeRegressor(random_state = 0)
# Train the model on training data
model.fit(train_x, train_y)

y_pred = model.predict(test_x)

print("Actual :", test_y, " Predicted : ", y_pred)

# The mean squared error
print("Mean squared error: %.2f"% mean_squared_error(test_y, y_pred))


