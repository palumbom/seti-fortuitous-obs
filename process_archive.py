# imports
import numpy as np
import pandas as pd
import os, sys, pdb, pytz
import matplotlib.pyplot as plt

from pytz import timezone
from IPython import embed
from tzlocal import get_localzone
from datetime import datetime, timedelta
from astroquery.gaia import Gaia

from skyfield.api import load, Star, T0, Loader
from skyfield.positionlib import ICRF, position_of_radec

import astropy.units as u
from astropy.time import Time
from astropy.table import QTable
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord

from barycorrpy import utc_tdb
from barycorrpy import get_BC_vel

# data directory
datdir = os.path.realpath(__file__ + "/../data/") + "/"
if not os.path.isdir(datdir):
    os.mkdir(datdir)
print("Data directory: " + datdir)

# beam info
beamLabel = ["L","S","C","X"]
upp = [1.73, 2.60, 7.8, 11.6]
low = [1.15, 1.73, 3.95, 7.8]
beams = [12.6/x for x in upp]
beamdeg =[x/60.0 for x in beams]
beamDict = dict(zip(beamLabel, beamdeg))
freqDict = dict(zip(beamLabel, upp))

# distances along focal line
dists = [*range(550,1000,50), *range(1000,27600, 500)] * u.au

# timezone stuff
utcTZ = timezone("UTC")
ts = load.timescale()

# set load directory
load = Loader(datdir)

# load planets
planets = load("de421.bsp")
sun = planets["sun"]
earth = planets["earth"]
d = 4.132e18
catt = d/2.99792458e10

# calc antipode of observation, assuming probe at 550 AU and star at 25pc
def calc_obs_antipode(ra_obs, dec_obs, obstime, probe_dist=550 * u.au):
    # get time of observation
    t = ts.from_datetime(obstime.datetime.replace(tzinfo=utcTZ))

    # get parallax at 550 AU
    dist_pc = probe_dist.to(u.pc)
    parallax = dist_pc.to(u.mas, equivalencies=u.parallax())

    # construct "star" object for probe
    coord = SkyCoord(ra_obs, dec_obs)
    ra_hour = coord.ra.hourangle * u.hourangle
    the_probe = Star(ra_hours=ra_hour.value, dec_degrees=dec_obs.value, parallax_mas=parallax.value)

    # calculate vectors
    sun_probe = sun.at(t).observe(the_probe)
    sun_probe_unit_vec = sun_probe.position.au / np.linalg.norm(sun_probe.position.au)
    sun_star_vec = (25.0 * u.pc).to(u.au).value * -sun_probe_unit_vec
    earth_star_vec = earth.at(t).observe(sun).position.au + sun_star_vec

    # get ra and dec in ICRF frame
    ra, dec, _ = ICRF(earth_star_vec).radec()
    ra = ((ra._degrees) % 360.0) * (24.0/360.0)
    dec = dec._degrees
    star = position_of_radec(ra, dec, (25.0 * u.pc).to(u.au).value, t=t)
    starCoord = star.position.au
    starDir = ICRF(starCoord).radec()
    ra_out = starDir[0]._degrees * u.deg
    dec_out = starDir[1]._degrees * u.deg
    return ra_out, dec_out

def calc_transmitter_pos(the_star, obstime, probe_dist):
    # distance in AU of probe
    z = probe_dist

    # get time of observation
    dt = obstime.datetime.replace(tzinfo=utcTZ)

    # compute ltt and retarded times
    c_cms = 2.99792458e10 * u.cm / u.s
    ltt = (z.to(u.cm))/c_cms #light travel time, seconds
    t = ts.from_datetime(dt)
    t_r = ts.from_datetime(dt - timedelta(seconds=ltt.value))
    t_rr = ts.from_datetime(dt - 2*timedelta(seconds=ltt.value))
    t_trans = ts.from_datetime(dt + 2*timedelta(seconds=catt) - 2*timedelta(seconds=ltt.value))

    # get vectors for probe
    S = sun.at(t)
    E = earth.at(t)
    x_ac = sun.at(t_trans).observe(the_star) - sun.at(t_trans)
    xvec = x_ac.position
    x = xvec.au/xvec.length().au
    P = S - ICRF(z.value*x)

    # get probe pos observed from Earth
    PE = P-E
    PE_dir = PE.radec()
    ra_out = PE_dir[0]._degrees * u.deg
    dec_out = PE_dir[1]._degrees * u.deg
    return ra_out, dec_out

def calc_receiver_pos(the_star, obstime, probe_dist):
    # distance in AU of probe
    z = probe_dist

    # get time of observation
    dt = obstime.datetime.replace(tzinfo=utcTZ)

    # compute ltt and retarded times
    c_cms = 2.99792458e10 * u.cm / u.s
    ltt = (z.to(u.cm))/c_cms #light travel time, seconds
    t = ts.from_datetime(dt)
    t_r = ts.from_datetime(dt - timedelta(seconds=ltt.value))
    t_rr = ts.from_datetime(dt - 2*timedelta(seconds=ltt.value))

    # get vectors for probe
    S = sun.at(t)
    E = earth.at(t)
    x_ac = sun.at(t_rr).observe(the_star) - sun.at(t_rr)
    xvec = x_ac.position
    x = xvec.au/xvec.length().au
    P = S - ICRF(z.value*x)

    # get probe pos observed from Earth
    PE = P-E
    PE_dir = PE.radec()
    ra_out = PE_dir[0]._degrees * u.deg
    dec_out = PE_dir[1]._degrees * u.deg
    return ra_out, dec_out

def get_gaia_stars(max_dist=(100.0 * u.pc)):
    # construct the query
    query = f"SELECT source_id, ra, dec, parallax, distance_gspphot, pmra, pmdec, radial_velocity \
             FROM gaiadr3.gaia_source \
             WHERE distance_gspphot <= {max_dist.value}\
             AND ruwe <1.4"

    # submit the query and get results
    job = Gaia.launch_job_async(query)
    return job.get_results()

def get_band(center_freq):
    freq = center_freq / 1e3
    if freq < upp[0]:
        return "L"
    elif freq < upp[1]:
        return "S"
    elif freq < upp[2]:
        return "C"
    elif freq < upp[3]:
        return "X"
    else:
        return "out"

def read_btl_csv():
    # import the breakthrough archival data CSV
    infile = datdir + "bldb_files.csv"
    names = ["telescope", "datetime", "target", \
             "ra", "dec", "center_freq", "datatype", \
             "datasize", "checksum", "url"]
    df = pd.read_csv(infile, names=names)

    # only select filterbank GBT observations
    df = df[(df.telescope == "GBT") & (df.datatype=="HDF5")]# | (df.datatype=="filterbank"))]

    # get bands of data
    bands = []
    for i in df["center_freq"].index:
        band = get_band(df["center_freq"][i])
        bands = np.append(bands, band)

    # add column and filter
    df["band"] = bands
    df = df[df.band != "out"]

    # move to astropy QTable
    t1 = QTable.from_pandas(df[["target", "ra", "dec", "band", "datetime", "url"]], index=True)
    t1["ra"] *= u.deg
    t1["dec"] *= u.deg
    t1["datetime"] = Time(t1["datetime"])
    return t1

def star_from_gaia(gaia_row):
    # get ra in hours
    coord = SkyCoord(gaia_row["ra"] * u.deg, gaia_row["dec"] * u.deg)
    ra_hour = coord.ra.hourangle

    gaia_epoch = ts.tt(2016.0)

    # create and return star object
    if (gaia_row["radial_velocity"] is np.ma.masked):
        the_star = Star(ra_hours=ra_hour,
                        dec_degrees=gaia_row["dec"],
                        ra_mas_per_year=gaia_row["pmra"],
                        dec_mas_per_year=gaia_row["pmdec"],
                        parallax_mas=gaia_row["parallax"],
                        epoch=gaia_epoch)
    else:
        the_star = Star(ra_hours=ra_hour,
                        dec_degrees=gaia_row["dec"],
                        ra_mas_per_year=gaia_row["pmra"],
                        dec_mas_per_year=gaia_row["pmdec"],
                        parallax_mas=gaia_row["parallax"],
                        radial_km_per_s=gaia_row["radial_velocity"],
                        epoch=gaia_epoch)
    return the_star

def get_drift_for_probe(ra_obs, dec_obs, obstime, band):
    # get the time of obs + make array 15 min into future
    # jdate = obs.header['tstart'] + 2400000.5
    jdate = obstime.jd
    JDUTC = np.linspace(jdate, jdate + (60.0 * 15.0/86400.), num=100)

    # get the pointing of the obs
    c = SkyCoord(ra_obs, dec_obs)
    s = c.to_string('decimal')
    ra_probe, dec_probe = [float(string) for string in s.split()]

    # other needed params
    obsname = 'GBT'
    epoch = 2451545.0
    rv = 0.0
    zmeas = 0.0
    ephemeris='https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/a_old_versions/de405.bsp'

    # get the BC vel
    baryvel = get_BC_vel(JDUTC=JDUTC, ra=ra_probe, dec=dec_probe, obsname="GBT",
                         rv=rv, zmeas=zmeas, epoch=epoch, ephemeris=ephemeris,
                         leap_update=True)

    # take the derivative of velocity to get acceleration (i.e., drift)
    dt = np.diff(JDUTC) * 86400.0
    dv = np.diff(baryvel[0])
    dv_dt = dv/dt

    # get the correct band
    upp_freq = freqDict[band]

    # calculate the drift rate
    drate = dv/dt * upp_freq * 1e9 / 3e8
    idx = np.argmax(np.abs(drate))
    maxdrift = drate[idx]
    return maxdrift

def format_dec_string(coord):
    fs = 1
    if coord < 0:
        fs = -1
        coord = abs(coord)
    qS = '"'
    return f"{int(fs*float(int(coord)))}$^\circ$ {int((coord - float(int(coord))) // (1/60))}' {np.around((coord - float(int(coord)) - ((coord - float(int(coord))) // (1/60))*(1/60))/(1/(60*60)),3)}{qS} "

def format_ra_string(coord):
    return f"{int(coord // 15)}h {int((coord - (coord // 15)*15) // 0.25)}m {np.around((coord - (coord // 15)*15 - ((coord - (coord // 15)*15) // 0.25)*0.25)/(360/(24*60*60)),3)}s"


def main():
    # set output filename
    filename = datdir + "fortuitous.csv"
    if os.path.isfile(filename):
        os.remove(filename)

    # get table of gaia data and btl data
    btl_tbl = read_btl_csv()
    gaia_tbl = get_gaia_stars()

    # get SkyCoord objects for stars in gaia catalog
    gaia_ra = gaia_tbl["ra"]
    gaia_dec = gaia_tbl["dec"]
    gaia_pmra = gaia_tbl["pmra"]
    gaia_pmdec = gaia_tbl["pmdec"]
    gaia_dist = gaia_tbl["distance_gspphot"]
    gaia_frame = "ICRS"
    gaia_obstime = Time("2016-01-01T00:00:00", format="isot", scale="utc")
    c_gaia = SkyCoord(gaia_ra, gaia_dec, pm_ra_cosdec=gaia_pmra, pm_dec=gaia_pmdec,
                      distance=gaia_dist, obstime=gaia_obstime, frame="icrs")

    # allocate memory for probe coords, drift rates, etc.
    ra_trans = np.zeros(len(dists))
    dec_trans = np.zeros(len(dists))
    sep_trans = np.zeros(len(dists))
    maxdrift_trans = np.zeros(len(dists))

    ra_rec = np.zeros(len(dists))
    dec_rec = np.zeros(len(dists))
    sep_rec = np.zeros(len(dists))
    maxdrift_rec = np.zeros(len(dists))

    # set tolerance for course search
    tol_angle = 0.5 * u.deg

    # loop over breakthrough listen obs
    for i in range(len(btl_tbl)):
        # print location
        print(">>> Doing BTL " + str(btl_tbl[i]["index"]))

        # get observation info
        ra_obs = btl_tbl[i]["ra"]
        dec_obs = btl_tbl[i]["dec"]
        obstime = btl_tbl[i]["datetime"]
        band = btl_tbl[i]["band"]

        # get skycoord of obs
        c_obs = SkyCoord(ra_obs, dec_obs)

        # get the simple antipode of observation
        ra_anti, dec_anti = calc_obs_antipode(ra_obs, dec_obs, obstime)
        c_anti = SkyCoord(ra_anti, dec_anti, frame="icrs")

        # apply pm correction to gaia coord for obstime
        c_gaia_temp = c_gaia.apply_space_motion(obstime)

        # calculate angular separation between btl antipode and gaia stars
        sep = c_anti.separation(c_gaia_temp)

        # get indices of gaia stars that are within tolerance
        idx = np.where(sep <= tol_angle)[0]
        close_stars = gaia_tbl[idx]

        # loop over close stars
        for j in range(len(close_stars)):
            # re-zero allocated memory
            ra_trans[:] = 0.0
            dec_trans[:] = 0.0
            sep_trans[:] = 0.0
            maxdrift_trans[:] = 0.0

            ra_rec[:] = 0.0
            dec_rec[:] = 0.0
            sep_rec[:] = 0.0
            maxdrift_rec[:] = 0.0

            # create star object
            gaia_star = star_from_gaia(close_stars[j])

            # loop over distances on focal line
            for k in range(len(dists)):
                # get transmitter coordinates
                ra_trans_k, dec_trans_k = calc_transmitter_pos(gaia_star, obstime, dists[k])
                c_trans_k = SkyCoord(ra_trans_k, dec_trans_k)

                # calculate the seperation
                sep_trans_k = c_obs.separation(c_trans_k).value

                # get the drift rate
                drate_trans_k = get_drift_for_probe(ra_trans_k, dec_trans_k, obstime, band)

                # copy to array
                ra_trans[k] = ra_trans_k.value
                dec_trans[k] = dec_trans_k.value
                sep_trans[k] = sep_trans_k
                maxdrift_trans[k] = drate_trans_k

                # get receiver coordinate
                ra_rec_k, dec_rec_k = calc_receiver_pos(gaia_star, obstime, dists[k])
                c_rec_k = SkyCoord(ra_rec_k, dec_rec_k)

                # calculate the sepeartion
                sep_rec_k = c_obs.separation(c_rec_k).value

                # get the drift rate
                drate_rec_k = get_drift_for_probe(ra_rec_k, dec_rec_k, obstime, band)

                # copy to array
                ra_rec[k] = ra_rec_k.value
                dec_rec[k] = dec_rec_k.value
                sep_rec[k] = sep_rec_k
                maxdrift_rec[k] = drate_rec_k

            # set angular distance tolerance
            ang_dist_tol = 1.0 * beamDict[band]

            # get ang distance conditions
            ang_dist_cond = (any(sep_rec <= ang_dist_tol) | any(sep_trans <= ang_dist_tol))

            # decide whether to write out row
            if ang_dist_cond:
                # make data row
                row = {"gaia_source_id":close_stars[j]["source_id"],
                       "gaia_dist":close_stars[j]["distance_gspphot"],

                       "btl_index":btl_tbl[i]["index"],
                       "target":btl_tbl[i]["target"], "url":btl_tbl[i]["url"],
                       "ra_obs":ra_obs.value, "dec_obs":dec_obs.value,
                       "obs_time":obstime.isot,
                       "obs_band":band,

                       "ra_trans":ra_trans, "dec_trans":dec_trans,
                       "sep_trans":sep_trans, "maxdrift_trans":maxdrift_trans,

                       "ra_rec":ra_rec, "dec_rec":dec_rec,
                       "sep_rec":sep_rec, "maxdrift_rec":maxdrift_rec}

                # make it a data frame
                df = pd.DataFrame.from_dict([row])

                # decide whether to write header
                if os.path.isfile(filename):
                    header = False
                else:
                    header = True

                # write it to disk
                df.to_csv(filename, mode="a", index=False, header=header)

    return None

if __name__ == "__main__":
    main()


# # Given string
# string_data = '[173.34744027 173.355064   173.36151421 ... 173.43705651 173.43709127]'

# # Remove the brackets and line breaks, then split the string into individual values
# cleaned_data = string_data.strip('[]\n').split()

# # Convert the cleaned string values to floats and create a NumPy array
# numpy_array = np.array([float(value) for value in cleaned_data])

# print(numpy_array)
