# imports
import numpy as np
import pandas as pd
import os, sys, pdb, pytz
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import Table, Column, MaskedColumn
import cdspyreadme

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
cols_to_alter = ["ra_trans", "dec_trans", "maxdrift_trans", "ra_rec", "dec_rec", "maxdrift_rec"]
for i in range(np.max([15, len(df_new)])):
    # do intense table surgery
    num_rows = 2
    k = [0, len(dists)-1]
    df_temp = pd.concat([df.iloc[i:i+1, :]]*num_rows, ignore_index=True)
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


# now get the FULL machine-readable table
df = read_csv(datfile)
df_new = pd.DataFrame(columns=df.columns)

# subset the rows and columns to write
for i in range(len(df)):
    # do intense table surgery
    num_rows = len(dists)
    k = np.arange(len(dists))
    df_temp = pd.concat([df.iloc[i:i+1, :]]*num_rows, ignore_index=True)
    cols_to_alter = ["ra_trans", "dec_trans", "sep_trans", "maxdrift_trans", "ra_rec", "dec_rec", "sep_rec", "maxdrift_rec"]
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

# get it as an astropy table
t1 = Table.from_pandas(df_new)
t2 = Table(t1, masked=True)

# set the masking
mask_idx = np.full(len(t1), True)
mask_idx[0:len(t1)-1:len(dists)] = False

# change dataypes
t2["gaia_source_id"][mask_idx] = 0
t2["gaia_source_id"].fill_value = 0
t2["gaia_source_id"] = t2["gaia_source_id"].astype('i8')
t2["gaia_source_id"].mask = mask_idx

t2["gaia_dist"][mask_idx] = 0
t2["gaia_dist"].fill_value = 0
t2["gaia_dist"] = t2["gaia_dist"].astype('f8')
t2["gaia_dist"].mask = mask_idx

t2["btl_index"][mask_idx] = 0
t2["btl_index"].fill_value = 0
t2["btl_index"] = t2["btl_index"].astype('i8')
t2["btl_index"].mask = mask_idx

t2["ra_obs"][mask_idx] = 0
t2["ra_obs"].fill_value = 0
t2["ra_obs"] = t2["ra_obs"].astype('f8')
t2["ra_obs"].mask = mask_idx

t2["dec_obs"][mask_idx] = 0
t2["dec_obs"].fill_value = 0
t2["dec_obs"] = t2["dec_obs"].astype('f8')
t2["dec_obs"].mask = mask_idx

t2["ra_trans"].fill_value = 0
t2["ra_trans"] = t2["ra_trans"].astype('f8')

t2["dec_trans"].fill_value = 0
t2["dec_trans"] = t2["dec_trans"].astype('f8')

t2["sep_trans"].fill_value = 0
t2["sep_trans"] = t2["sep_trans"].astype('f8')

t2["maxdrift_trans"].fill_value = 0
t2["maxdrift_trans"] = t2["maxdrift_trans"].astype('f8')

t2["ra_rec"].fill_value = 0
t2["ra_rec"] = t2["ra_rec"].astype('f8')

t2["dec_rec"].fill_value = 0
t2["dec_rec"] = t2["dec_rec"].astype('f8')

t2["sep_rec"].fill_value = 0
t2["sep_rec"] = t2["sep_rec"].astype('f8')

t2["maxdrift_rec"].fill_value = 0
t2["maxdrift_rec"] = t2["maxdrift_rec"].astype('f8')

# make probe dist column
t2["probe_dist"] = np.repeat(dists.value, int(len(t2)/len(dists)))

# set units
t2["gaia_source_id"].unit = None
t2["gaia_dist"].unit = u.pc
t2["btl_index"].unit = None
t2["target"].unit = None
t2["url"].unit = None
t2["ra_obs"].unit = u.deg
t2["dec_obs"].unit = u.deg
t2["obs_time"].unit = None
t2["obs_band"].unit = None
t2["ra_trans"].unit = u.deg
t2["dec_trans"].unit = u.deg
t2["sep_trans"].unit = u.deg
t2["maxdrift_trans"].unit = u.Hz / u.s
t2["ra_rec"].unit = u.deg
t2["dec_rec"].unit = u.deg
t2["sep_rec"].unit = u.deg
t2["maxdrift_rec"].unit = u.Hz / u.s
t2["probe_dist"].unit = u.AU

# add to tablemaker
tablemaker = cdspyreadme.CDSTablesMaker()

# set the readme
tablemaker.title = "Fortuitous Observations of Potential Stellar Relay Probe Positions with GBT"
tablemaker.author = 'M.L. Palumbo'
tablemaker.authors = 'J.T. Wright, M.H. Huston'
tablemaker.date = "2023"

# add the data
table = tablemaker.addTable(t2, name="tab1.txt")

# set column descriptions
column = table.get_column("gaia_source_id")
column.description="Source ID from Gaia DR3 Catalog"

column = table.get_column("gaia_dist")
column.description="Gaia gspphot distance"

column = table.get_column("btl_index")
column.description="Databse index of Breakthrough Listen observation"

column = table.get_column("target")
column.description="Target of Breakthrough Listen observation"

column = table.get_column("url")
column.description="URL for Breakthrough Listen data download"

column = table.get_column("ra_obs")
column.description="RA in ICRS of the Breakthrough Listen observation"

column = table.get_column("dec_obs")
column.description="Dec in ICRS of the Breakthrough Listen observation"

column = table.get_column("obs_time")
column.description="ISO 8601 compliant date format of Breakthrough Listen observation; timezone is UTC"

column = table.get_column("obs_band")
column.description="Band of the GBT observation"

column = table.get_column("ra_trans")
column.description="List of RAs along transmitting probe focal line in ICRS"

column = table.get_column("dec_trans")
column.description="List of Decs along transmitting probe focal line in ICRS"

column = table.get_column("sep_trans")
column.description="List of angular separations of transmitting probes from observation pointing"

column = table.get_column("maxdrift_trans")
column.description="List of maximum drift rates of transmitting probes"

column = table.get_column("ra_rec")
column.description="List of RAs along receiving probe focal line in ICRS"

column = table.get_column("dec_rec")
column.description="List of Decs along receiving probe focal line in ICRS"

column = table.get_column("sep_rec")
column.description="List of angular separations of receiving probes from observation pointing"

column = table.get_column("maxdrift_rec")
column.description="List of maximum drift rates of receiving probes"

column = table.get_column("probe_dist")
column.description="List of probe distances that coordinates and drift rates were calculated for"

# write it out
tablemaker.toMRT()

# pdb.set_trace()

# from astropy.io import ascii
# data = ascii.read("tab1.txt", format="mrt")
