"""
File Overview
--------------
Functionaility for composing and optionally saving metrics
"""

import os
import pandas as pd
from dataset_defintions import *

# Prepared Suffix
PREPPED = "prepped_"

# Warning message when preparing dataset
WARNING_COLUMN_MISSING = " column is missing"

def clean_dataset(dataset_path):
    """
    desc
        Reads and potentially cleans an acceleration dataset
    param
        dataset_path - path to the dataset directory or file
    return
        df - dataframe of the parsed data if any is found
    """
    df = None
    if os.path.isdir(dataset_path):
        df = directory_cleaner(dataset_path)
        output_prepped_data(dataset_path + PREPPED + "meh", df)
    elif os.path.isfile(dataset_path):
        # Dataset has already been prepared read the file
        if PREPPED in os.path.basename(dataset_path):
            df = pd.read_csv(dataset_path)
        else:
            df = csv_cleaner(dataset_path)
            output_prepped_data(dataset_path, df)
    else:
        raise ValueError("The path given was invalid. No data was parsed")

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
    desc
        Raises an error to the user if the data is not as expected
    params
        columns - the names of the columns of the dataframe to be evaluated
    """
    if BEHAVIOR not in columns:
        raise ValueError(f"{BEHAVIOR}' {WARNING_COLUMN_MISSING}")
    if ACCELERATION_X not in columns:
        raise ValueError(f"{ACCELERATION_X}' {WARNING_COLUMN_MISSING}")
    if ACCELERATION_Y not in columns:
        raise ValueError(f"{ACCELERATION_Y}' {WARNING_COLUMN_MISSING}")
    if ACCELERATION_Z not in columns:
        raise ValueError(f"{ACCELERATION_Z}' {WARNING_COLUMN_MISSING}")

def remove_case_sensitivity(df):
    """
    desc
        Removes case sensitivity of column names by making all column names upper case
    params
        df - the dataframe to have it's columns revised
    return
        column names of given dataframe rewritten to uppercase
    """
    upper_case_column_names = [columns_name.upper() for columns_name in df.columns]
    df.columns = upper_case_column_names

    return df

def csv_cleaner(csv_filepath):
    """
    desc
        Cleans the acceleration data csv
    params
        csv_filepath - filepath + name to the csv
    return
        df - dataframe with the cleaned version of the csv data
    """
    df = pd.read_csv(csv_filepath, low_memory=False)
    df = remove_case_sensitivity(df)

    column_warnings(df.columns)

    # Reassigning the index to the one
    # given explicitly as a column in the dataset
    df[NEW_INDEX] = df.index

    # For any row in the column subset that has a N/A value drop it
    df = df.dropna(subset=[BEHAVIOR, ACCELERATION_X, ACCELERATION_Y, ACCELERATION_Z])

    # No video indicates that the sample had no video to discern 
    # it's behavior and should be removed 
    df = df.loc[df[BEHAVIOR] != NO_VIDEO]

    # Create a set of column names we would like to drop by
    # removing the ones we would like to keep from the overall set
    drop_column_names = set(df.columns) - set(COLUMN_NAMES)
    df = df.drop(drop_column_names, axis=1)

    return df

def directory_cleaner(dataset_path):
    """
    desc
        Cleans a directory containing acceleration data csv's
    params
        dataset_path - path to a directory of acceleration data csv's
    return
        combined_df - a dataframe with all csv's cleaned and combined into one dataframe
    """
    collected_dfs = []
    files = accumulate_csv_files(dataset_path)

    for csv in files:
        accelerometry_df = csv_cleaner(csv)
        collected_dfs.append(accelerometry_df)

    combined_df = pd.concat(collected_dfs)
    return combined_df

def output_prepped_data(output_path, clean_csv_df):
    """
    desc
        Writes the dataframe contained cleaned data to a new csv file
    params
        output_path - where to write the output
        clean_csv_df - cleaned csv as a dataframe
    """
    # If the path given was a filename get the basepath
    filename = os.path.basename(output_path)
    clean_csv_df.to_csv(PREPPED + filename + ".csv", index=False)
