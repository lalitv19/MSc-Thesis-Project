''' This program is used to extract the only the cross correlated signals 
 from all the 15 matlab files ( for 15 different transmitter locations) 
 and saving them in matlab files again.
 Each nanobee is like a signal reciever'''

import numpy as np
import pandas as pd
import scipy.io as io
import matplotlib.pyplot as plt
import os

dir = os.getcwd()

for location_ID in range(1, 16):

    print(location_ID)
    loc = io.loadmat(os.getcwd()+'/data/unprocessed/Loc'+str(location_ID)+'.mat')

    cc_dict = {}

    # correlated data
    c_data = loc['Loc'+str(location_ID)][0][0]['bee']['c']
    for nanoBee_ID in range(5):
        values = 10*np.log10(abs(c_data[0][nanoBee_ID])) # convert the signal into frequency DB

        key = 'NanoBee_' + str(nanoBee_ID+1)

        cc_dict[key] = values[:, :250] # extract relevant part of cc time series

    io.savemat(os.getcwd()+'/data/processed/Loc'+str(location_ID), cc_dict)


