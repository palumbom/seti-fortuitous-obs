# imports
import numpy as np
import pandas as pd
import os, sys, pdb, pytz
import matplotlib.pyplot as plt
import astropy.units as u

# function to fix the arrays that were written as strings
def string_to_array(string_data):
    cleaned_data = string_data.strip('[]\n').split()
    return np.array([float(value) for value in cleaned_data])

# file for reading csv
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

# get the file
datdir = os.path.realpath(__file__ + "/../data/") + "/"
datfile = datdir + "fortuitous.csv"
texfile = datdir + "table.tex"
assert os.path.isfile(datfile)

# set distances
dists = [*range(550,1000,50), *range(1000,27600, 500)] * u.au

# read it in
df = read_csv(datfile)
df = df.drop_duplicates("gaia_source_id")
df_new = pd.DataFrame(columns=df.columns)

# subset the rows and columns to write
for i in range(np.max([15, len(df_new)])):
    # do intense table surgery
    num_rows = 2
    k = [0, len(dists)-1]
    df_temp = pd.concat([df.iloc[i:i+1, :]]*num_rows, ignore_index=True)
    cols_to_alter = ["ra_trans", "dec_trans", "maxdrift_trans", "ra_rec", "dec_rec", "maxdrift_rec"]
    for i in range(len(df_temp)):
        if i == 0:
            for c in cols_to_alter:
                df_temp.loc[i, c] = df_temp.loc[i, c][k[i]]
        else:
            for c in df_temp.columns:
                if c in cols_to_alter:
                    df_temp.loc[i, c] = df_temp.loc[i, c][k[i]]
                else:
                    df_temp.loc[i, c] = " "
    if i == 0:
        df_new = pd.concat([df_temp, df_new], ignore_index=True)
    else:
        df_new = pd.concat([df_new, df_temp], ignore_index=True)

# get columns to write and write it
cols_to_write = ["gaia_source_id", "btl_index", "target", "ra_obs",
                 "dec_obs", "obs_band", "ra_trans",
                 "dec_trans", "maxdrift_trans"]

df_new.to_latex(buf=texfile, columns=cols_to_write, na_rep="-", index=False)
