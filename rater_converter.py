from distutils.command.clean import clean
import pandas as pd
import numpy as np
import csv
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN



def pull_window(df, window_size):
    """
    Input: 
    df -- dataframe of cleaned input data, likely from a csv
    window_size -- number of rows of data to convert to 1 row for AcceleRater (25 = 1sec)
    Output:
    windows -- list of lists of accel data (EX:[x,y,z,...,x,y,z,class_label])
    """
    if window_size > df.shape[0]:
        raise ValueError('Window larger than data given')
    windows = []
    number_of_rows_minus_window = df.shape[0] - window_size + 1
    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i:i+window_size]
        if len(set(window.behavior)) != 1:
            continue
        if len(set(np.ediff1d(window.input_index))) != 1:
             continue
        windows.append(window)
    return windows


def accel_singlelabel_xy(windows):
    """
    Input:
        windows -- list of dataframes of all one class 
    Output:
        Xdata -- arrays of xyz data of each window stacked together
        ydata -- integer class labels for each window
    """
    positions = ['accX', 'accY', 'accZ']
    strikes = ['h', 'm']
    Xdata, ydata = [], []
    for window in windows:
        behavior = window['behavior'].iloc[0]
        if behavior in strikes:
            behavior  = 't'
        Xdata.append(window[positions].to_numpy())
        ydata.append(behavior)
    return np.stack(Xdata), np.asarray(ydata)

# why not, create windows normally, oversample, then flatten and append?


def accel_oversampler(Xdata,ydata, sample_flag = False):
    """
    Reduces the dimensions of inout data, splits into train/test set,
    then randomly oversamples the minority classes to match the majority class"""
    nsamples, nx, ny = Xdata.shape
    Xdata2d = Xdata.reshape((nsamples,nx*ny))
    # stshsp = StratifiedShuffleSplit(n_splits= 1, test_size =0.2, random_state=42)
    # train_index, test_index = next(stshsp.split(Xdata2d,ydata))
    # x_train, x_test = Xdata2d[train_index], Xdata2d[test_index]
    # y_train, y_test = ydata[train_index], ydata[test_index]
    if sample_flag == 'o':
        ros = RandomOverSampler(random_state=0)
        X_resampled, y_resampled = ros.fit_resample(Xdata2d, ydata)
    elif sample_flag == 's':
        X_resampled, y_resampled = SMOTE(k_neighbors=2).fit_resample(Xdata2d, ydata)
    elif sample_flag == 'a':
        X_resampled, y_resampled = ADASYN(n_neighbors=2).fit_resample(Xdata2d, ydata)
    else:
        X_resampled, y_resampled = Xdata2d, ydata

    tupdata = zip(X_resampled,y_resampled)
    total_oversample_data = [list(elem) for elem in tupdata]
    clean_data = [np.append(sample[0], sample[1]) for sample in total_oversample_data]

    print("train/test sets stratified and split")
    return clean_data


def rater_construct_train_test(windows):
    """
    Input:
        windows -- list of dataframes of all one class 
    Output:
        total_data -- list of lists of flattened accel datawith class label a the end
    """
    positions = ['accX', 'accY', 'accZ']
    total_data = [] 
    strikes = ['h', 'm']
    for window in windows:
        windowdata = window[positions].to_numpy()
        xlist = windowdata.tolist()
        flat_list = [item for sublist in xlist for item in sublist]
        behavior = window['behavior'].iloc[0]
        if behavior in strikes:
            behavior  = 't'
        flat_list.append(behavior)
        total_data.append(flat_list)
        
    return total_data


def main():
    df = pd.read_csv("~/CNNworkspace/raterdata/dec21_cleanPennf1.csv")
    df = df.loc[df['behavior'] != 'l']
    windows = pull_window(df, 25)
    all_data = rater_construct_train_test(windows)
    with open("CNNworkspace/raterdata/nostill_Pennf1flat_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(all_data)


#TODO:
    # write script to clean data, option for training and test/raw data

if __name__ == "__main__":
    main()