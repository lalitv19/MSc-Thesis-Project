""" This program is used for predicting the Time of Arrival (TOA in nano seconds) of the signal
 using multi layer peceptron model"""

import numpy as np
import scipy.io as io
import matplotlib.pyplot as plt
import os
np.set_printoptions(suppress=True)

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

train_points = [1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 15]
dev_points = [12, 13]
test_points = [8, 10]

train_set = []
for loc in train_points:
    for bee in range(1, 6):
        TOA_GT_loc_bee = TOA_GT_ns[loc-1, bee-1]
        all_cc_signals = io.loadmat(dir + '/processed/Loc'+str(loc))

        for measurement in range(all_cc_signals['NanoBee_'+str(bee)].shape[0]):
            cc_signals = all_cc_signals['NanoBee_'+str(bee)][measurement]

            # data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee]))
            data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee/seq_length]))

            train_set.append(data_point)
            # plt.plot(freq, data_point[:250])
            # plt.plot([TOA_GT_loc_bee, TOA_GT_loc_bee], [0, 100], color='r')
            # plt.ylim([0, 70])
            # plt.show()
train_set = np.array(train_set)
np.random.shuffle(train_set)
# print(train_set.shape)


dev_set = []
for loc in dev_points:
    for bee in range(1, 6):
        TOA_GT_loc_bee = TOA_GT_ns[loc-1, bee-1]
        all_cc_signals = io.loadmat(dir + '/processed/Loc'+str(loc))

        for measurement in range(all_cc_signals['NanoBee_'+str(bee)].shape[0]):
            cc_signals = all_cc_signals['NanoBee_'+str(bee)][measurement]

            # data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee]))
            data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee/seq_length]))

            dev_set.append(data_point)
            plt.plot(freq, data_point[:250])
            plt.plot([TOA_GT_loc_bee, TOA_GT_loc_bee], [0, 100], color='r')
            plt.ylim([0, 70])
            plt.show()
dev_set = np.array(dev_set)
np.random.shuffle(dev_set)
# print(dev_set.shape)


test_set = []
for loc in test_points:
    for bee in range(1, 6):
        TOA_GT_loc_bee = TOA_GT_ns[loc-1, bee-1]
        all_cc_signals = io.loadmat(dir + '/processed/Loc'+str(loc))

        for measurement in range(all_cc_signals['NanoBee_'+str(bee)].shape[0]):
            cc_signals = all_cc_signals['NanoBee_'+str(bee)][measurement]

            # data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee]))
            data_point = np.concatenate((cc_signals, [TOA_GT_loc_bee/seq_length]))
            test_set.append(data_point)
            plt.plot(freq, data_point[:250])
            plt.plot([TOA_GT_loc_bee, TOA_GT_loc_bee], [0, 100], color='r')
            plt.ylim([0, 70])
            plt.show()
test_set = np.array(test_set)
np.random.shuffle(test_set)
# print(test_set.shape)

#########################################################################
#           MLP
########################################################################

import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.nn.functional as F

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
        a2 = torch.tanh(self.fc1(x))
        a3 = torch.tanh(self.fc2(a2))

        # The output layer uses a softmax to give a probability distribution over all 10 possible classes (e.g. shoe, sneaker etc.)
        y_hat = torch.sigmoid(self.fc3(a3))
        # y_hat = self.fc3(a3)

        return y_hat


def loss_func(y, y_hat):
    # Mean square error loss
    y_hat = y_hat.reshape(-1)
    se = (y-y_hat)**2
    return torch.mean(se)


# Neural networks are stochastic. Different random seeds will result in different weights in the trained network
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

# Hyperparameters which must be tuned (see slides for how to tune the hyperparameters)
num_inputs = 250
num_L2 = 50
num_L3 = 50
num_outputs = 1
num_epochs = 100
learning_rate_alpha = 0.0001
batch_size = 64
lambd_for_regularization = 0.001

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_alpha, weight_decay=lambd_for_regularization)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_alpha)


# Data preprocessing
# Scale training data

X_train_unscaled = train_set[:, :250]
Y_train_unscaled = train_set[:, -1]

mu_X_train = np.mean(X_train_unscaled, axis=0)
std_X_train = np.std(X_train_unscaled, axis=0)

mu_Y_train = np.mean(Y_train_unscaled, axis=0)
std_Y_train = np.std(Y_train_unscaled, axis=0)

max_X_train = np.max(X_train_unscaled, axis=0)
min_X_train = np.min(X_train_unscaled, axis=0)

max_Y_train = np.max(Y_train_unscaled, axis=0)
min_Y_train = np.min(Y_train_unscaled, axis=0)

X_train_scaled = (X_train_unscaled - mu_X_train) / std_X_train

Y_train_scaled = Y_train_unscaled

# Scale dev data (use mu and std from training data!!)
X_dev_unscaled = dev_set[:, :250]
Y_dev_unscaled = dev_set[:, -1]
X_dev_scaled = (X_dev_unscaled - mu_X_train) / std_X_train
Y_dev_scaled = Y_dev_unscaled

# Scale test data (use mu and std from training data!!)
X_test_unscaled = test_set[:, :250]
Y_test_unscaled = test_set[:, -1]
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

        # Backward and optimize
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        # save errors on training set
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

        # Model Selection based on development set loss: save the best model found over all epochs
        if np.mean(dev_losses) < best_dev_error:
            best_dev_error = np.mean(dev_losses)
            torch.save(model, dir +'/models/mlp_TOA_weights')


fig, ax1 = plt.subplots()
ax1.set_xlabel('Epoch', fontsize=20)
ax1.set_ylabel('Loss', fontsize=20)
ax1.plot(train_loss_over_epoch, color='k', linewidth=3, label='Train Loss')
ax1.plot(dev_loss_over_epoch, color='r', linewidth=3, label='Dev Loss')
# ax1.set_ylim([0, 1000])
h1, l1 = ax1.get_legend_handles_labels()
ax1.legend(h1, l1, loc='upper left', ncol=2, prop={'size': 12})
plt.show()
plt.close()
plt.savefig(os.getcwd()+'/MLP TOA learning_curves.pdf')



model = torch.load(dir +'/models/mlp_TOA_weights')

# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)


N = 250
Fs = 61440000
Ts = 1/Fs
Ts_ns = Ts/1e-9
seq_length = (Ts_ns*N)
freq = np.arange(N) / (float(N) / seq_length)
error = []
with torch.no_grad():
    # save errors on development set
    dev_losses = []
    i = 0
    for i, (x, y) in enumerate(test_loader):

        y_hat = model(x)
        error.append(np.abs((y*seq_length) - (y_hat*seq_length)))
        

# CDF plot for errors genereating in predicted TOAs from Actual TOAs
fig, ax1 = plt.subplots()
ax1.set_xlabel('Deviation from actual TOA in nano seconds', fontsize=12)
#ax1.set_ylabel('Percenatage of signals', fontsize=12)
sorted_errors = np.sort(error)
Y = np.arange(0, 1, 1/len(sorted_errors))
ax1.plot(sorted_errors, Y)
ax1.set_xlim([0, 50])
plt.savefig(dir +'/plots/MLP TOA for loc 8_10.pdf')
plt.show()
