""" This program predicts x, y, z location of transmitter by directly using cross coreelated signals """

import numpy as np
import pandas as pd
import scipy.io as io
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from copy import deepcopy
np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)


dir = 'D:/Lalit/Capstone/data'
distances = io.loadmat(dir +'/distance_matrix')['distance_matrix']
locations_data = io.loadmat(dir +'/05-Feb-2019_meas_info')['meas_info'][0]

measurement_locs = np.zeros((15, 3))
for measurement_point in range(15):
    measurement_locs[measurement_point, :] = locations_data[measurement_point][0]

reference = deepcopy(measurement_locs[8])
measurement_locs -= reference

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


train_points = [1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 15]
dev_points = [12, 13]
test_points = [8, 10]

train_set = []
for loc in train_points:
    all_cc_signals = io.loadmat(dir + '/processed/Loc'+str(loc))

    location = measurement_locs[loc-1]

    num_measurements = all_cc_signals['NanoBee_1'].shape[0]
    for measurement in range(num_measurements):
        bee_1 = all_cc_signals['NanoBee_1'][measurement]
        bee_2 = all_cc_signals['NanoBee_2'][measurement]
        bee_3 = all_cc_signals['NanoBee_3'][measurement]
        bee_4 = all_cc_signals['NanoBee_4'][measurement]
        bee_5 = all_cc_signals['NanoBee_5'][measurement]

        data_point = np.concatenate((bee_1, bee_2, bee_3, bee_4, bee_5, location))
        train_set.append(data_point)
train_set = np.array(train_set)
np.random.shuffle(train_set)


dev_set = []
for loc in dev_points:
    all_cc_signals = io.loadmat(dir + '/processed/Loc'+str(loc))

    location = measurement_locs[loc-1]

    num_measurements = all_cc_signals['NanoBee_1'].shape[0]
    for measurement in range(num_measurements):
        bee_1 = all_cc_signals['NanoBee_1'][measurement]
        bee_2 = all_cc_signals['NanoBee_2'][measurement]
        bee_3 = all_cc_signals['NanoBee_3'][measurement]
        bee_4 = all_cc_signals['NanoBee_4'][measurement]
        bee_5 = all_cc_signals['NanoBee_5'][measurement]

        data_point = np.concatenate((bee_1, bee_2, bee_3, bee_4, bee_5, location))
        dev_set.append(data_point)
dev_set = np.array(dev_set)
np.random.shuffle(dev_set)


test_set = []
for loc in test_points:
    all_cc_signals = io.loadmat(dir + '/processed/Loc'+str(loc))

    location = measurement_locs[loc-1]

    num_measurements = all_cc_signals['NanoBee_1'].shape[0]
    for measurement in range(num_measurements):
        bee_1 = all_cc_signals['NanoBee_1'][measurement]
        bee_2 = all_cc_signals['NanoBee_2'][measurement]
        bee_3 = all_cc_signals['NanoBee_3'][measurement]
        bee_4 = all_cc_signals['NanoBee_4'][measurement]
        bee_5 = all_cc_signals['NanoBee_5'][measurement]

        data_point = np.concatenate((bee_1, bee_2, bee_3, bee_4, bee_5, location))
        test_set.append(data_point)
test_set = np.array(test_set)
np.random.shuffle(test_set)


def distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2)**2))


#########################################################################
#           MLP
########################################################################

import torch
import torch.nn as nn
import torch.utils.data as data_utils

class MLP(nn.Module):
    # Create a class which we'll call 'MLP'
    def __init__(self):
        # Extend the class (see tutorial for explanation)
        super(MLP, self).__init__()

        # Specify the architecture of the MLP
        self.fc1 = nn.Linear(num_inputs, num_L2)
        self.fc2 = nn.Linear(num_L2, num_L3)
        self.fc3 = nn.Linear(num_L3, num_outputs)

    def forward(self, x):
        # We only have to specify the foward pass; Pytorch will automatically compute gradients. We're using ReLU non-linearities
        a2 = torch.relu(self.fc1(x))
        a3 = torch.relu(self.fc2(a2))

        # The output layer uses a softmax to give a probability distribution over all 10 possible classes (e.g. shoe, sneaker etc.)
        y_hat = self.fc3(a3)
        return y_hat

def loss_func(y, y_hat):
    # Mean square error loss
    se = ((y-y_hat)**2)
    return torch.mean(se)

# Neural networks are stochastic. Different random seeds will result in different weights in the trained network
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters which must be tuned (see slides for how to tune the hyperparameters)
num_inputs = 1250
num_L2 = 100
num_L3 = 100
num_outputs = 3
num_epochs = 100
learning_rate_alpha = 0.0001
batch_size = 16
lambd_for_regularization = 5.0

model = MLP()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_alpha, weight_decay=lambd_for_regularization)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_alpha)

# Data preprocessing
# Scale training  
X_train_unscaled = train_set[:, :1250]
Y_train_unscaled = train_set[:, 1250:]

mu_X_train = np.mean(X_train_unscaled, axis=0)
std_X_train = np.std(X_train_unscaled, axis=0)

mu_Y_train = np.mean(Y_train_unscaled, axis=0)
std_Y_train = np.std(Y_train_unscaled, axis=0)

max_Y_train = np.max(Y_train_unscaled, axis=0)
min_Y_train = np.min(Y_train_unscaled, axis=0)

X_train_scaled = (X_train_unscaled - mu_X_train) / std_X_train
Y_train_scaled = Y_train_unscaled

# Scale dev data (use mu and std from training data!!)
X_dev_unscaled = dev_set[:, :1250]
Y_dev_unscaled = dev_set[:, 1250:]

X_dev_scaled = (X_dev_unscaled - mu_X_train) / std_X_train
Y_dev_scaled = Y_dev_unscaled

# Scale test data (use mu and std from training data!!)
X_test_unscaled = test_set[:, :1250]
Y_test_unscaled = test_set[:, 1250:]

X_test_scaled = (X_test_unscaled - mu_X_train) / std_X_train
Y_test_scaled = Y_test_unscaled

# Convert to Pytorch tensors. Hence, Pytorch can backpropagate gradients.
train_x = torch.tensor(X_train_scaled, dtype=torch.float32)
train_y = torch.tensor(Y_train_scaled, dtype=torch.float32)
train_set = data_utils.TensorDataset(train_x, train_y)

# Place data into an object that allows us to iterate over training cases
train_loader = data_utils.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)

dev_x = torch.tensor(X_dev_scaled, dtype=torch.float32)
dev_y = torch.tensor(Y_dev_scaled, dtype=torch.float32)
dev_set = data_utils.TensorDataset(dev_x, dev_y)
dev_loader = data_utils.DataLoader(dataset=dev_set, batch_size=1, shuffle=True)

test_x = torch.tensor(X_test_scaled, dtype=torch.float32)
test_y = torch.tensor(Y_test_scaled, dtype=torch.float32)
test_set = data_utils.TensorDataset(test_x, test_y)
test_loader = data_utils.DataLoader(dataset=test_set, batch_size=1, shuffle=True)

train_loss_over_epoch = []
dev_loss_over_epoch = []
best_dev_error = np.inf
for epoch in range(num_epochs):
    print('epoch', epoch)
    train_losses = []
    for i, (x, y) in enumerate(train_loader):
        # Forward pass
        y_hat = model(x)
        train_loss = loss_func(y, y_hat)
#         # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
#         # save errors on training set
        train_losses.append(train_loss.item())
    train_loss_over_epoch.append(np.mean(train_losses))

    with torch.no_grad():
        # save errors on development set
        dev_losses = []
        for i, (x, y) in enumerate(dev_loader):
            y_hat = model(x)
            dev_loss = loss_func(y, y_hat)
            dev_losses.append(dev_loss.item())
        dev_loss_over_epoch.append(np.mean(dev_losses))
#         # Model Selection based on development set loss: save the best model found over all epochs
        if np.mean(dev_losses) < best_dev_error:
            best_dev_error = np.mean(dev_losses)
            torch.save(model, dir +'/mlp_loc_weights')

fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch', fontsize=20)
ax1.set_ylabel('Loss', fontsize=20)
ax1.plot(train_loss_over_epoch, color='k', linewidth=3, label='Train Loss')
ax1.plot(dev_loss_over_epoch, color='r', linewidth=3, label='Dev Loss')
ax1.set_ylim([0, 30])
h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h1, l1, loc='upper left', ncol=2, prop={'size': 12})
plt.savefig(dir +'/Mlp_loc_learning_curves.pdf')
plt.show()

model = torch.load(dir+'/mlp_loc_weights')

with torch.no_grad():
    # save errors on development set
    dev_losses = []
    errors_locs = []
    for i, (x, y) in enumerate(test_loader):
        y_hat = model(x)
       # print('Ground Truth: ' + str(np.array(y[0])), 'Predicted: '+ str(np.array(y_hat[0])))
        errors_locs.append(distance(np.array(y), np.array(y_hat)))
        
    print('errors_loc', np.average(errors_locs))
        
# Generate CUmmulicative Distribution Function - CDF of errors
    
fig, ax1 = plt.subplots()
ax1.set_xlabel('Deviation from actual points in meters', fontsize=12)
#ax1.set_ylabel('Percenatage of signals', fontsize=12)
sorted_errors = np.sort(errors_locs)
Y = np.arange(0, 1, 1/len(sorted_errors))
ax1.plot(sorted_errors, Y)
ax1.set_xlim([0, 15])
plt.savefig(dir +'/mlp_loc_updated.pdf')
plt.show()
