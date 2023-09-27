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
        for c in cols_to_alter:
            df_temp.loc[i, c] = df_temp.loc[i, c][k[i]]
    if i == 0:
        df_new = pd.concat([df_temp, df_new], ignore_index=True)
    else:
        df_new = pd.concat([df_new, df_temp], ignore_index=True)

# get it as an astropy table
t1 = Table.from_pandas(df_new)
t2 = Table()

# change dataypes
t2["Gaia"] = t1["gaia_source_id"].astype('i8')
t2["Distance"] = t1["gaia_dist"].astype('f8')
t2["BTLIndex"] = t1["btl_index"].astype('i8')
t2["Target"] = t1["target"]
t2["url"] = t1["url"]
t2["RAObs"] = t1["ra_obs"].astype('f8')
t2["DEObs"] = t1["dec_obs"].astype('f8')
t2["ObsTime"] = t1["obs_time"]
t2["ObsBand"] = t1["obs_band"]
t2["RATrans"] = t1["ra_trans"].astype('f8')
t2["DETrans"] = t1["dec_trans"].astype('f8')
t2["SepTrans"] = t1["sep_trans"].astype('f8')
t2["MaxDriftTr"] = t1["maxdrift_trans"].astype('f8')
t2["RARec"] = t1["ra_rec"].astype('f8')
t2["DERec"] = t1["dec_rec"].astype('f8')
t2["SepRec"] = t1["sep_rec"].astype('f8')
t2["MaxDriftRec"] = t1["maxdrift_rec"].astype('f8')
t2["ProbeDist"] = np.tile(dists.value, int(len(t2)/len(dists)))

# set units
t2["Gaia"].unit = None
t2["Distance"].unit = u.pc
t2["BTLIndex"].unit = None
t2["Target"].unit = None
t2["url"].unit = None
t2["RAObs"].unit = u.deg
t2["DEObs"].unit = u.deg
t2["ObsTime"].unit = None
t2["ObsBand"].unit = None
t2["RATrans"].unit = u.deg
t2["DETrans"].unit = u.deg
t2["SepTrans"].unit = u.deg
t2["MaxDriftTr"].unit = u.Hz / u.s
t2["RARec"].unit = u.deg
t2["DERec"].unit = u.deg
t2["SepRec"].unit = u.deg
t2["MaxDriftRec"].unit = u.Hz / u.s
t2["ProbeDist"].unit = u.AU

# add to tablemaker
tablemaker = cdspyreadme.CDSTablesMaker()

# set the readme
tablemaker.title = "Fortuitous Observations of Potential Stellar Relay Probe Positions with GBT"
tablemaker.author = 'Palumbo M.:'
tablemaker.authors = 'Wright J.T., Huston M.H.'
tablemaker.date = "2023"
tablemaker.table = "Table of GBT observaations that fall near the antipodes of stars within 100 pc"

# add the data
table = tablemaker.addTable(t2, name="datafile1.txt")

# set column descriptions
column = table.get_column("Gaia")
column.description="Source ID from Gaia DR3 Catalog"

column = table.get_column("Distance")
column.description="Gaia gspphot distance"

column = table.get_column("BTLIndex")
column.description="Databse index of Breakthrough Listen observation"

column = table.get_column("Target")
column.description="Target of Breakthrough Listen observation"

column = table.get_column("url")
column.description="URL for Breakthrough Listen data download"

column = table.get_column("RAObs")
column.description="RA in ICRS of the Breakthrough Listen observation"

column = table.get_column("DEObs")
column.description="Dec in ICRS of the Breakthrough Listen observation"

column = table.get_column("ObsTime")
column.description="ISO 8601 compliant date format of Breakthrough Listen observation; timezone is UTC"

column = table.get_column("ObsBand")
column.description="Band of the GBT observation"

column = table.get_column("RATrans")
column.description="List of RAs along transmitting probe focal line in ICRS"

column = table.get_column("DETrans")
column.description="List of Decs along transmitting probe focal line in ICRS"

column = table.get_column("SepTrans")
column.description="List of angular separations of transmitting probes from observation pointing"

column = table.get_column("MaxDriftTr")
column.description="List of drift rates of transmitting probes"

column = table.get_column("RARec")
column.description="List of RAs along receiving probe focal line in ICRS"

column = table.get_column("DERec")
column.description="List of Decs along receiving probe focal line in ICRS"

column = table.get_column("SepRec")
column.description="List of angular separations of receiving probes from observation pointing"

column = table.get_column("MaxDriftRec")
column.description="List of drift rates of receiving probes"

column = table.get_column("ProbeDist")
column.description="List of probe distances that coordinates and drift rates were calculated for"

# write it out
tablemaker.toMRT()

# pdb.set_trace()

# from astropy.io import ascii
# data = ascii.read("tab1.txt", format="mrt")
