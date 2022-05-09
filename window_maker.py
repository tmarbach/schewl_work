import numpy as np

def pull_window(df, window_size):
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

    if window_size > df.shape[0]:
        raise ValueError('Window larger than data given')

    number_of_rows_minus_window = df.shape[0] - window_size + 1

    for i in range(0, number_of_rows_minus_window, window_size):
        window = df[i:i+window_size]
        if len(set(window.behavior)) != 1:
            continue
        if len(set(np.ediff1d(window.input_index))) != 1:
             continue
            
        windows.append(window)
        classes.append(window.iloc[0]['behavior'])
    
    allclasses = set(classes)

    return windows, list(allclasses)