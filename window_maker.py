"""
File Overview
--------------
Takes samples and aggregates them into windows based on different stratgies and window sizes
"""

import numpy as np
from dataset_defintions import *

def check_data_size(window_size, data_shape):
    """
    desc
        common check for different windowing strategies
    params
        window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
        data_shape  -- shape of the sample data
    """
    if window_size > data_shape:
        raise ValueError('Window larger than data given')

def leaping_window(df, window_size, coi=False):
    """
    desc: 
        Places behaviors into windows.
    params: 
        df -- dataframe of cleaned input data, likely from a csv
        window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
        coi - classes of interest. If specified other behaviors will be thrown out and not placed into windows.
    return:
        windows -- list of lists of accel data (EX:[x,y,z,...,x,y,z,class_label])
        allclasses -- list of the behavior classes that are present in the windows
    """
    classes = []
    windows = []

    data_shape = df.shape[0]
    check_data_size(window_size, data_shape)

    number_of_rows_minus_window = data_shape - window_size + 1
    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i: i+window_size]

        if len(window[BEHAVIOR].unique()) != 1:
            continue
        if len(set(np.ediff1d(window.index))) != 1:
            continue

        if coi:
            coi_classes = list(coi)
            current_behavior = window[BEHAVIOR].iloc[0]
            if current_behavior not in list(coi_classes):
                continue

        windows.append(window)
        classes.append(window[BEHAVIOR].iloc[0])

    return windows, set(classes)