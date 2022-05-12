#!/usr/bin/env python3

#Performs Data Cleaning

import os
import pandas as pd

# Column Names
BEHAVIOR = 'BEHAVIOR'
ACCELERATION_X = 'ACCX'
ACCELERATION_Y = 'ACCY'
ACCELERATION_Z = 'ACCZ'
WARNING_COLUMN_MISSING = " column is missing"
NEW_INDEX = 'INPUT_INDEX'
COLUMN_NAMES = [ BEHAVIOR,
            ACCELERATION_X, 
            ACCELERATION_Y, 
            ACCELERATION_Z ]

# Preppared Suffix
PREPPED = "prepped_"

def prepare_dataset(raw_dataset_path):
    """
    
    """
    df = None
    if os.path.isdir(raw_dataset_path):
        df = directory_cleaner(raw_dataset_path)
        output_prepped_data(raw_dataset_path + "meh", df)
    elif os.path.isfile(raw_dataset_path):
        # Dataset has already been preppared read the file
        if PREPPED in os.path.basename(raw_dataset_path):
            df = pd.read_csv(raw_dataset_path)
        else:
            df = csv_cleaner(raw_dataset_path)
            output_prepped_data(raw_dataset_path, df)
    else:
        print("ERROR: YOUR PATH SUX")

    return df

def accumulate_csv_files(data_set_path):
    """
    """
    files = []
    for file in os.listdir(data_set_path):
        if file.endswith('.csv'):
            files.append(data_set_path + file)
            
    return files

def column_warnings(columns):
    """
    Inform the user if the data is not as expected
    """
    if BEHAVIOR not in columns:
        raise ValueError(f"{BEHAVIOR}' {WARNING_COLUMN_MISSING}")
    if ACCELERATION_X not in columns:
        raise ValueError(f"{ACCELERATION_X}' {WARNING_COLUMN_MISSING}")
    if ACCELERATION_Y not in columns:
        raise ValueError(f"{ACCELERATION_Y}' {WARNING_COLUMN_MISSING}")
    if ACCELERATION_Z not in columns:
        raise ValueError(f"{ACCELERATION_Z}' {WARNING_COLUMN_MISSING}")

def csv_cleaner(csv_filepath):
    """

    """
    df = pd.read_csv(csv_filepath, low_memory=False)

    # Remove case sensitivity from column names
    upper_case_column_names = [columns_name.upper() for columns_name in df.columns]
    df.columns = upper_case_column_names

    column_warnings(df.columns)

    df[NEW_INDEX] = df.index
    df = df[[c for c in COLUMN_NAMES if c in df] + [c for c in df if c not in COLUMN_NAMES]]

    df = df.dropna(subset=[BEHAVIOR])
    df = df.loc[df[BEHAVIOR] != 'n']

    drop_column_names = set(df.columns) - set(COLUMN_NAMES)
    df = df.drop(drop_column_names, axis=1)

    return df

def directory_cleaner(data_set_path):
    """
    """
    collected_dfs = []
    files = accumulate_csv_files(data_set_path)

    for csv in files:
        accelerometry_df = csv_cleaner(csv)
        collected_dfs.append(accelerometry_df)

    combined_df = pd.concat(collected_dfs)
    return combined_df

def output_prepped_data(original_data, clean_csv_df):
    """
    
    """
    filename = os.path.basename(original_data)
    clean_csv_df.to_csv(PREPPED + filename + ".csv", index=False)
