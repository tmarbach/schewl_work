import numpy as np
from clean_acceleration_data import remove_case_sensitivity
from dataset_defintions import *


def check_data_size(window_size, data_shape):
    if window_size > data_shape:
        raise ValueError('Window larger than data given')

def singleclass_leaping_window(df, window_size, coi=False):
    """
    desc: 

    params: 
        df -- dataframe of cleaned input data, likely from a csv
        window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)

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

def pull_window(df, window_size, class_list):
    """
    Input: 
        df -- dataframe of cleaned input data, likely from a csv
        window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
    Output:
        windows -- list of lists of accel data (EX:[x,y,z,...,x,y,z,class_label])
        allclasses -- list of the behavior classes that are present in the windows
    
    """
    classes = []
    windows = []

    data_shape = df.shape[0]
    check_data_size(window_size, data_shape)

    number_of_rows_minus_window = df.shape[0] - window_size + 1
    df = remove_case_sensitivity(df)

    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i:i+window_size]

        if len(set(window[BEHAVIOR])) != 1:
            continue
        if len(set(np.ediff1d(window.index))) != 1:
            continue
        if window.iloc[0][BEHAVIOR] not in class_list:
            continue

        windows.append(window)
        classes.append(window.iloc[0][BEHAVIOR])

    allclasses = set(classes)

    return windows, list(allclasses)
