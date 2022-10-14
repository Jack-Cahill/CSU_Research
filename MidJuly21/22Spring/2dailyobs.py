#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:24:59 2022

@author: jcahill4
"""

import xarray as xr
import datetime
import pandas as pd
import numpy as np

vp = 'v200'  # ulwrf_2 (olr), tp (prcp), or gh_2 (pres)
Tests = 'False'  # True, False, mini1, mini2

if vp == 'olr':
    var = 'ulwrf_2'
elif vp == 'prcp':
    var = 'tp'
elif vp == 'h500':
    var = 'gh_2'
elif vp == 'rel_hum':
    var = '2r'
elif vp == 'sfc_temp':
    var = '2t'
elif vp == 'v850':
    var = 'v_3'
elif vp == 'v200':
    var = 'v_3'

if Tests == True:
    df_all = xr.open_mfdataset('/Users/jcahill4/DATA/{}/Obs/raw_gefs/Tests/*.nc'.format(vp), concat_dim='time',
                               combine='nested')
elif Tests == 'mini1':
    df_all = xr.open_mfdataset('/Users/jcahill4/DATA/{}/Obs/raw_gefs/Tests1/*.nc'.format(vp), concat_dim='time',
                               combine='nested')
elif Tests == 'mini2':
    df_all = xr.open_mfdataset('/Users/jcahill4/DATA/{}/Obs/raw_gefs/Tests2/*.nc'.format(vp), concat_dim='time',
                               combine='nested')
else:
    df_all = xr.open_mfdataset('/Users/jcahill4/DATA/{}/Obs/raw_gefs/*.nc'.format(vp), concat_dim='time',
                               combine='nested')

# %% Create list of dates

if Tests == 'mini1' or Tests == 'mini2':
    numdays = 2
else:
    numdays = 7305
if Tests == True:
    base = datetime.datetime(2016, 7, 19)
elif Tests == 'mini1':
    base = datetime.datetime(2017, 12, 31)
elif Tests == 'mini2':
    base = datetime.datetime(2004, 6, 23)
else:
    base = datetime.datetime(2000, 1, 1)
date_list = [base + datetime.timedelta(days=x) for x in range(numdays)]

# %%
# If pressure, only get 500hPa
if var == 'gh_2' or var == '2r' or var == '2t' or var == 'v_3':
    df = df_all[var]
    df = df[:, 0, :, :]
else:
    df = df_all[var]
print(df)
# %%
for i in range(int(len(df.time.values) / 4)):

    # Average daily
    if Tests == True:
        df1 = df[0 + i]
        df2 = df[1261 + i]
        df3 = df[1261 * 2 + i]
        df4 = df[1261 * 3 + i]
    elif Tests == 'mini1' or Tests == 'mini2':
        df1 = df[0 + i]
        df2 = df[2 + i]
        df3 = df[2 * 2 + i]
        df4 = df[2 * 3 + i]
    else:
        df1 = df[0 + i]
        df2 = df[7305 + i]
        df3 = df[7305 * 2 + i]
        df4 = df[7305 * 3 + i]
    df14 = xr.concat([df1, df2, df3, df4], dim="time")
    dfAvg = df14.mean(dim='time')

    # # # Reshape so it's like obs

    # lons
    df_shp_lons = dfAvg[:, ::2]

    # lats
    df_shp_lats1 = df_shp_lons[:360, :]
    df_shp_lats2 = df_shp_lons[360, :]
    df_shp_lats3 = df_shp_lons[362:, :]

    df_shp_lats1 = df_shp_lats1[::2, :]
    df_shp_lats3 = df_shp_lats3[::2, :]

    df_shp = xr.concat([df_shp_lats1, df_shp_lats2, df_shp_lats3], dim="lat")

    # Flip lats
    df_shp = df_shp.reindex(lat=list(reversed(df_shp.lat)))

    # Drop level if necessary
    if var == 'gh_2' or var == 'v_3':
        df_shp = df_shp.drop('plev')
    elif var == '2r' or var == '2t':
        df_shp = df_shp.drop('height')

    # Add extra dimension for time
    df_shp_v = df_shp.values
    df_out = df_shp_v.reshape((1, len(df_shp.lat.values), len(df_shp.lon.values)))

    # Select time
    tt = date_list[i]
    tlist = np.array([tt.date()])
    tlist_pd = pd.DataFrame(tlist, columns=['time'])
    tlist_pd['time'] = pd.to_datetime(tlist_pd['time'], format='%Y-%m-%d')

    # time for file output
    tout = tt.date().strftime('%Y%m%d')

    # Output dfAvg as xarray with same set-up as original
    if var == 'ulwrf_2':
        Xout = xr.DataArray(data=df_out, dims=["time", "lat", "lon"],
                            coords=dict(time=tlist_pd['time'], lon=df_shp.lon.values, lat=df_shp.lat.values, ),
                            attrs=dict(long_name="Upward long-wave radiation flux", units="W m**-2",
                                       param="193.5.0",
                                       level_type="toa"))
    elif var == 'tp':
        Xout = xr.DataArray(data=df_out, dims=["time", "lat", "lon"],
                            coords=dict(time=tlist_pd['time'], lon=df_shp.lon.values, lat=df_shp.lat.values, ),
                            attrs=dict(long_name="Total Precipitation", units="kg m**-2",
                                       param="8.1.0"))
    elif var == 'gh_2':
        Xout = xr.DataArray(data=df_out, dims=["time", "lat", "lon"],
                            coords=dict(time=tlist_pd['time'], lon=df_shp.lon.values, lat=df_shp.lat.values, ),
                            attrs=dict(standard_name="geopotential_height_500",
                                       long_name="Geopotential height at 500hPa",
                                       units="gpm", param="5.3.0"))
    elif var == '2r':
        Xout = xr.DataArray(data=df_out, dims=["time", "lat", "lon"],
                            coords=dict(time=tlist_pd['time'], lon=df_shp.lon.values, lat=df_shp.lat.values, ),
                            attrs=dict(standard_name="relative_humidity", long_name="2 metre relative humidity",
                                       units="%", param="1.1.0"))
    elif var == '2t':
        Xout = xr.DataArray(data=df_out, dims=["time", "lat", "lon"],
                            coords=dict(time=tlist_pd['time'], lon=df_shp.lon.values, lat=df_shp.lat.values, ),
                            attrs=dict(standard_name="air_temperature", long_name="2 metre temperature",
                                       units="K", param="0.0.0"))
    elif var == 'v_3':
        Xout = xr.DataArray(data=df_out, dims=["time", "lat", "lon"],
                            coords=dict(time=tlist_pd['time'], lon=df_shp.lon.values, lat=df_shp.lat.values, ),
                            attrs=dict(standard_name="northward_wind", long_name="V component of wind",
                                       units="m s**-1", param="3.2.0"))

    Xout = Xout.to_dataset(name='{}'.format(var))  # Name variable
    Xout.to_netcdf('{}_{}.nc'.format(vp, tout))
