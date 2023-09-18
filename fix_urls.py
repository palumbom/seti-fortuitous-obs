import numpy as np
import pandas as pd
import os, sys, pdb, pytz
import matplotlib.pyplot as plt

# fix string that should be array
def string_to_array(string_data):
    cleaned_data = string_data.strip('[]\n').split()
    return np.array([float(value) for value in cleaned_data])

# read in function for old csv
def read_csv(datfile):
    # set names of columns to fix
    cols_to_fix = ['ra_trans', 'dec_trans', 'sep_trans', \
               'maxdrift_trans', 'ra_rec', 'dec_rec', \
               'sep_rec', 'maxdrift_rec']

    # read the data
    df = pd.read_csv(datfile)

    # turn long strings back into arrays
    for col_name in cols_to_fix:
        arr = []
        for i in range(len(df)):
            arr.append(string_to_array(df.loc[i,col_name]))
        df[col_name] = arr

    return df

# read in file with original urls
df1 = read_csv("data/fortuitous.csv")

# read in the new csv
df_new = pd.read_csv("data/add_missing_urls.csv")

# get urls that were actually correct
idx = df_new.newurl.isnull().values
df_new.loc[idx, "newurl"] = df1.url[idx]

# now replace old urls
df1.url = df_new.newurl

# now write to csv
df1.to_csv("data/fortuitous.csv", mode="w", index=False)
