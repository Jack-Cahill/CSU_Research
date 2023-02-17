#  This file provides all the pieces of my artifical neural network as of (2/16/23)

import geopandas as gp
import tensorflow as tf
import sys
import numpy as np
import xarray as xr
import pandas as pd
import time
import datetime
import matplotlib.pyplot as plt
import cmasher as cmr
from sklearn import preprocessing
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import matplotlib.colors as colors
from Functions import make_model, savitzky_golay

sys.path.append('')  # add path to network, metrics and plot
warnings.filterwarnings('ignore', "GeoSeries.isna", UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option("display.max_rows", None, "display.max_columns", None)
plt.style.use('dark_background')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['legend.title_fontsize'] = 'xx-small'
pd.options.mode.chained_assignment = None  # default='warn'

cmap = cmr.redshift  # CMasher
cmap = plt.get_cmap('cmr.redshift')  # MPL

# %% # # # # # # # # # # INPUTS # # # # # # # # # #

# Define a set of random seeds (or just one seed if you want to check)

# Define Input Map variable and dimensions
variable = 'olr'
lat_slice = slice(-25, 25)  # lat and lon region make up the Tropics
lon_slice = slice(60, 300)

# Define predictor
Pred = 'sfc_temp'

# Helps sort data
cdata = 'clima_gefs'
LT_tot = 35  # how many total lead times are there in the UFS forecast?

# Decrease input resolution?
dec = 0  # 0 if normal, 1 if decrease

# Which type of Neural Network are we running
Reg = '1E'  # Options found in Reg_dict

# Dict for region (lat, lon, class name, class number, map size, epoch size, seeds)
Reg_dict = {'full': [slice(23.9, 49.6), slice(235.4, 293.6), 'All Classes', [0, 1, 2], 6, 10000,
                     [92, 95, 100, 137, 141, 142]],
            '4': [slice(33.9, 37.6), slice(255.4, 260.6), 'All Classes', [0], 0, 10000, [92, 95, 100, 137, 141, 142]],
            'NW': [slice(23.9, 49.6), slice(235.4, 293.6), 'Underestimates', [0], 7, 10000,
                   [92, 95, 100, 137, 141, 142]],
            'SE': [slice(23.9, 49.6), slice(235.4, 293.6), 'Overestimates', [2], 6, 10000,
                   [92, 95, 100, 137, 141, 142]],
            'SW': [slice(23.9, 49.6), slice(235.4, 293.6), 'Underestimates', [0], 7, 10000,
                   [92, 95, 100, 137, 141, 142]],
            'all_ep': [slice(23.9, 49.6), slice(235.4, 293.6), 'All Classes', [0], 6, 1000,
                       [92, 95, 100, 137, 141, 142]],
            'test': [slice(33.9, 34.1), slice(255.4, 260.6), 'All Classes', [0, 1, 2], 0, 10000, [92, 95]],
            'WC1': [slice(39.9, 40.1), slice(235.4, 244.4), 'N/A', [0, 1, 2], 0, 10000, [92]],
            'FallUnd_T': [slice(23.9, 49.6), slice(235.4, 293.6), 'All Classes', [0, 1, 2], 6, 10000,
                          [92, 95, 100, 137, 141, 142]],
            '1E': [slice(45.9, 46.1), slice(237.4, 237.6), 'All Classes', [0, 1, 2], 6, 10000,
                   [92, 95, 100, 137, 141, 142]],
            '1W': [slice(39.9, 40.1), slice(269.4, 269.6), 'All Classes', [0, 1, 2], 6, 10000,
                   [92, 95, 100, 137, 141, 142]],
            'solo': [slice(39.9, 40.1), slice(269.4, 269.6), 'All Classes', [0, 1, 2], 6, 10000,
                     [92]]}

# # # # # # # # # #  NEURAL NETWORK INPUTS # # # # # # # # # #

lead_time1 = 10  # will be averaged from LT1-LT2
lead_time2 = 14
epochs = Reg_dict[Reg][5]
nodes = 20
batch_size = 32  # Not identified in plot naming scheme
LR = 0.01  # Learning Rate
Trash = 0  # If we wanted to throw out any seeds, this would count how many seeds we toss
Classes = 3  # 3 classes (-1: Under_est, 0: Acc_est, 1: Over_est)
TTLoco = 0  # Order of Training and Validation datasets (0 if TrainVal, 1 if ValTrain)
RIDGE1 = 0.0  # Regularization Techniques
DROPOUT = 0.3
PATIENCE = 20
MinMax = 0  # 0 if minimizing loss and 1 if maximizing accuracy

if MinMax == 0:
    monitor = 'val_loss'
    mode = 'min'
else:
    monitor = 'val_prediction_accuracy'
    mode = 'max'

# # # # # # # # # #  READ IN DATA # # # # # # # # # #

# OBS
# Read in ALL obs using xarray and parameters
ds_obs = xr.open_mfdataset('/Users/jcahill4/DATA/{}/Obs/{}/clima_{}.nc'.format(
    variable, cdata, variable), concat_dim='time', combine='nested')
ds_obs = ds_obs[variable].sel(lon=lon_slice, lat=lat_slice)

# Specify lat and lon for input
lats = ds_obs.lat.values  # lat is latitude name given by data
lons = ds_obs.lon.values
lons = lons + 180

# Turn obs into a pandas array
ds_obs_l = list(zip(ds_obs.time.values, ds_obs.values))
ds_obs_all = pd.DataFrame(ds_obs_l)
ds_obs_all.columns = ['time', '{} obs'.format(variable)]

# Get dates that match UFS dates (weekly forecasts)
ds_obs_sub = ds_obs_all[4:7295:7]  # 4: start at 1-5-00, 7295: end at 12-18-19, 7: weekly forecast
ds_obs_sub = ds_obs_sub.reset_index(drop=True)
ds = ds_obs_sub.sort_values(by='time')
timer = ds['time']

# Pred data base
ds_UFS1_base = xr.open_mfdataset('/Users/jcahill4/DATA/{}/UFS/{}/*.nc'.format(Pred, cdata), concat_dim='time',
                                 combine='nested')
if Pred == 'h500':
    ds_obs1_base = xr.open_dataset('/Users/jcahill4/DATA/{}/Obs/{}/clima_{}.nc'.format(Pred, cdata, Pred))
else:
    ds_obs1_base = xr.open_mfdataset('/Users/jcahill4/DATA/{}/Obs/{}/*.nc'.format(Pred, cdata), concat_dim='time',
                                     combine='nested')
ds_UFS_base = xr.open_mfdataset('/Users/jcahill4/DATA/{}/UFS/{}/*.nc'.format(Pred, cdata), concat_dim='time',
                                combine='nested')
CONUS = ds_obs1_base['{}'.format(Pred)].sel(lat=Reg_dict[Reg][0], lon=Reg_dict[Reg][1])

# Decrease resolution if looking at a region larger than a singular location
if Reg == '1E' or Reg == '1W' or Reg == 'solo':
    CONUS_lats = CONUS.lat.values
    CONUS_lons = CONUS.lon.values
else:
    CONUS_lats = CONUS.lat.values[::4]
    CONUS_lons = CONUS.lon.values[::4]
    CONUS_lons = CONUS_lons[:len(CONUS_lons) - 1]
print(CONUS_lats)
print(CONUS_lons)

# # # # # # # # # #  MJO + ENSO COMPARISON SET-UP # # # # # # # # # #

# Merge MJO and ENSO info
mdata = pd.read_csv('/Users/jcahill4/Downloads/mjo_phase2.csv')
mdata = mdata.drop_duplicates()
mdata_ymd = mdata.iloc[:, :3]  # Just grab YMD data
mdata_dt1 = pd.to_datetime(mdata_ymd[['year', 'month', 'day']])  # convert YMD columns to one dt column

# Add Date and Phase to YMD data
mdata_ymd['dates'] = mdata_dt1
mdata_ymd['M_Phase'] = mdata['Phase']
mdata_ymd['E_Phase'] = mdata['ENSO']
mdata_ymd['Amp'] = mdata['Amplitude']
mdata_ymd['Amp'] = pd.to_numeric(mdata_ymd['Amp'], downcast="float")

# RMM adjustment (if Amplitude (sqrt(RMM1^2 + RMM2^2)) < 1, then convert phase to 0 (phase 0 is just no MJO Phase)
mdata_ymd['M_Phase'] = np.where(mdata_ymd['Amp'] < 1.0, 0, mdata_ymd['M_Phase'])

# Create new column for season
mdata_ymd.loc[mdata_ymd['month'].between(5.9, 8.1), 'Season'] = 'Summer'
mdata_ymd.loc[mdata_ymd['month'].between(8.9, 11.1), 'Season'] = 'Fall'
mdata_ymd.loc[mdata_ymd['month'].between(11.9, 12.1), 'Season'] = 'Winter'
mdata_ymd.loc[mdata_ymd['month'].between(0.9, 2.1), 'Season'] = 'Winter'
mdata_ymd.loc[mdata_ymd['month'].between(2.9, 5.1), 'Season'] = 'Spring'

# Drop Y-M-D
mdata_ymd = mdata_ymd.drop(['year', 'month', 'day'], axis=1)

# # # # # # # # # # PANDAS PRED OBS (FOR OUT MAPS) # # # # # # # # # #

Pred_sliceLat = slice(-30, 65)
Pred_sliceLon = slice(60, 300)

# Turn preds into a pandas array
ds_obs1_base_Pred = ds_obs1_base[Pred].sel(lat=Pred_sliceLat, lon=Pred_sliceLon)
ds_obs1_base_l = list(zip(ds_obs1_base_Pred.time.values, ds_obs1_base_Pred.values))
pred_obs = pd.DataFrame(ds_obs1_base_l)
pred_obs.columns = ['dates', '{} obs'.format(Pred)]

# reset indices
pred_obs = pred_obs.reset_index(drop=True)

# # # # # # # # # #  ACCURACY MAP SET-UP # # # # # # # # # #

# 23 is for the 23 maps I'll make
# 1 all samples, 3 classes (predicted class), 3 classes (actual class), 4 all samples each season, 3*4 (class*season)
acc_map_tot = 23
acc_pcents = 3  # 20%, 30% and all
Acc_Map_Data = np.empty((acc_pcents, acc_map_tot, len(CONUS_lons), len(CONUS_lats)))
Acc_Map_Data[:] = np.nan
TV_tot = 5

# # # # # # # # # #  ACC SPLITS (20, 30, ALL) # # # # # # # # # #

# We want to save accuracies for multiple different top %s
# key: [fig title, % ignored for acc, idx list, corr_or_no list, class list]
Acc_main_dict = {'20': ['20% Most Confident and Correct Samples', 0.8, [], [], [], []],
                 '30': ['30% Most Confident and Correct Samples', 0.7, [], [], [], []],
                 'All': ['All Correct Samples', 0, [], [], [], []]}

# # # # # # # # # #  THRESHOLD SAVE SET-UP # # # # # # # # # #

hit_miss_vars = 5  # Correct_R, Conf, y_val, WConf, idx
hit_miss_len = 209  # 208 or 209 for length of indices per TV set-up
seedr = len(Reg_dict[Reg][6])  # number of seeds

# # # # # # # # # #  REGIONS  # # # # # # # # # #

# Convert lat and lons into grid pts
ptlats = np.tile(CONUS_lats, len(CONUS_lons))
ptlons = np.repeat(CONUS_lons, len(CONUS_lats))

# Make sure lons match shapefile lons
ptlons = ptlons - 360

# Read in shapefiles & set coords
states = gp.read_file('/Users/jcahill4/DATA/Udders/usa-states-census-2014.shp')
states.crs = 'EPSG:4326'

# Set regions
SE = states[states['STUSPS'].isin(['FL', 'GA', 'AL', 'SC', 'NC', 'VA'])]
S = states[states['STUSPS'].isin(['TX', 'OK', 'KS', 'AR', 'LA', 'MS'])]
SW = states[states['STUSPS'].isin(['UT', 'NM', 'CO', 'AZ'])]
W = states[states['STUSPS'].isin(['CA', 'NV'])]
NW = states[states['STUSPS'].isin(['WA', 'OR', 'ID'])]
WNC = states[states['STUSPS'].isin(['MT', 'WY', 'ND', 'SD', 'NE'])]
ENC = states[states['STUSPS'].isin(['MN', 'IA', 'WI', 'MI'])]
C = states[states['STUSPS'].isin(['MO', 'IL', 'IN', 'OH', 'KY', 'TN', 'WV'])]
NE = states[states['STUSPS'].isin(['PA', 'MD', 'DE', 'NJ', 'CT', 'NY', 'RI', 'MA', 'ME', 'NH', 'VT'])]
FU_T = states[states['STUSPS'].isin(['WA', 'OR', 'ID', 'NV', 'MT', 'WY', 'ND', 'SD', 'NE', 'CO', 'UT', 'OK', 'KS',
                                     'MN', 'IA', 'MO', 'IL'])]

# put lat and lons into array as gridpoints using geopandas
SF_grid = pd.DataFrame({'longitude': ptlons, 'latitude': ptlats})
gdf = gp.GeoDataFrame(SF_grid, geometry=gp.points_from_xy(SF_grid.longitude, SF_grid.latitude))

# Plot region
if Reg == 'NW':
    region = NW
elif Reg == 'SE':
    region = SE
elif Reg == 'SW':
    region = SW
elif Reg == 'FallUnd_T':
    region = FU_T
else:
    region = states
US_pts = gp.sjoin(gdf, region, op='within')
us_boundary_map = states.boundary.plot(color="white", linewidth=1)
US_pts.plot(ax=us_boundary_map, color='pink')
plt.show()

# put lat and lons into pandas df
reg_lon_lat = pd.DataFrame({'longitude': US_pts.longitude, 'latitude': US_pts.latitude})

# # # # # # # # # #  TRAIN VAL SPLIT # # # # # # # # # #

# val_start, val_end, train_start1, train_end1, train_start2, train_end2, bignum, smlnum, out_name, title
TV_dict = {'TT1': [0, 208, 208, 2000, 2000, 2001, 834, 208, 'TT1', 'Validation - 1st Quintile'],
           'TT2': [208, 417, 0, 208, 417, 2000, 833, 209, 'TT2', 'Validation - 2nd Quintile'],
           'TT3': [417, 625, 0, 417, 625, 2000, 834, 208, 'TT3', 'Validation - 3rd Quintile'],
           'TT4': [625, 834, 0, 625, 834, 2000, 833, 209, 'TT4', 'Validation - 4th Quintile'],
           'TT5': [834, 1042, 0, 834, 2000, 2001, 834, 208, 'TT5', 'Validation - 5th Quintile']}

# 3 is for different classes, 4 is for different types of hmaps (correct, class, etc), 3x9 is for hmap shape
TV_hmap_mean = np.empty((TV_tot, acc_pcents, 3, 4, 3, 9))
TV_hmap_mean[:] = np.nan

# # # # # # # # # #  IDX ALLLLL # # # # # # # # # #

# Create a pandas arrays that has index, dates, and obs maps of validation data
val_idx = np.arange(0, 1042)
obs_maps = ds['{} obs'.format(variable)][0:1042]
date_list = ds['time'][0:1042]
idx_date = pd.DataFrame({'Index': val_idx, 'dates': date_list, 'obs': obs_maps})

# Merge with time data
idx_all = pd.merge(idx_date, mdata_ymd, on='dates')
print(idx_all[:20])
# %% MEGA LOOP

# Set up a counter, so we can determine how many grid points we're running for
counter = 0

# Loop through locations
for c1, xxx in enumerate(CONUS_lons):
    for c2, xx in enumerate(CONUS_lats):

        # Check if the lats and lons fall within specified region - if so, run NN
        df_chk = reg_lon_lat.loc[reg_lon_lat.latitude == xx]
        if (xxx - 360) in df_chk.longitude.unique():

            # count many grid points we're running for and print grid point
            counter = counter + 1
            print(xx, xxx)

            # just empty lists - each list contains each seeds, idx, CorNo, Class, acc_list final (szn/class array),
            # acc_list general
            grid_dict = {'20': [[], [], [], [], [[] for _ in range(acc_map_tot)], []],
                         '30': [[], [], [], [], [[] for _ in range(acc_map_tot)], []],
                         'All': [[], [], [], [], [[] for _ in range(acc_map_tot)], []]}

            # Set up new hitmiss numpy array save
            HM_Data = np.empty((seedr, TV_tot, hit_miss_vars, hit_miss_len))
            HM_Data[:] = np.nan

            # Loop through TVc set-up
            for TVc, KEY in enumerate(TV_dict):

                # Loop through seeds
                for seedcount, x in enumerate(Reg_dict[Reg][6]):

                    # Select map point
                    lat_sliceP = slice(xx - .01, xx + .01)
                    lon_sliceP = slice(xxx - .01, xxx + .01)

                    # Seed
                    NP_SEED = x
                    np.random.seed(NP_SEED)
                    tf.random.set_seed(NP_SEED)

                    # Prediction data
                    # Read in data (UFS and Obs)
                    ds_UFS1 = ds_UFS1_base['{}'.format(Pred)].sel(lat=lat_sliceP, lon=lon_sliceP)
                    ds_obs1 = ds_obs1_base['{}'.format(Pred)].sel(lat=lat_sliceP, lon=lon_sliceP)
                    ds_UFS1_vals = ds_UFS1[:, 0, 0].values
                    ds_UFS1_tvals = ds_UFS1[:, 0, 0].time.values

                    # # # # # # # # # # READ IN DATA - TRIPLE CHECKED # # # # # # # # # #

                    # PART1: MERGE UFS AND OBS DATA SO THEY MATCH
                    z = 0
                    while z < len(ds_UFS1.time.values) / LT_tot:

                        # # NEW (four lines) # #
                        # Grab UFS data between lead_times
                        UFS_vals = ds_UFS1_vals[z * LT_tot:z * LT_tot + LT_tot]
                        UFS_vals = UFS_vals[lead_time1:lead_time2 + 1]
                        UFS_tvals = ds_UFS1_tvals[z * LT_tot:z * LT_tot + LT_tot]
                        UFS_tvals = UFS_tvals[lead_time1:lead_time2 + 1]

                        # # NEW (one line) # #
                        # Convert UFS data to pandas
                        UFS = list(zip(UFS_tvals, UFS_vals))
                        UFS = pd.DataFrame(UFS)
                        UFS.columns = ['time', '{} UFS'.format(Pred)]

                        if z == 0:
                            ds_UFS10 = UFS
                        else:
                            ds_UFS10 = pd.concat([ds_UFS10, UFS])
                        z = z + 1

                    # Convert Obs to pandas
                    ds_obsp_l = list(zip(ds_obs1.time.values, ds_obs1.values.flatten()))
                    ds_obsp = pd.DataFrame(ds_obsp_l)
                    ds_obsp.columns = ['time', '{} Obs'.format(Pred)]

                    # Remove time (non-date) aspect
                    ds_UFS10['time'] = pd.to_datetime(ds_UFS10['time']).dt.date
                    ds_obsp['time'] = pd.to_datetime(ds_obsp['time']).dt.date

                    # Merge UFS and obs
                    ds10x = pd.merge(ds_obsp, ds_UFS10, on='time')

                    # PART 2: AVERAGE EACH FILE BETWEEN LEAD_TIMES X-Y, SO WE HAVE TWO VECTORS (OBS AND UFS)
                    divisor = int(lead_time2 - lead_time1 + 1)
                    ds10x = ds10x.sort_values(by='time')
                    ds10x = ds10x.drop(['time'], axis=1)
                    ds10 = ds10x.groupby(np.arange(len(ds10x)) // divisor).mean()

                    # PART 3: CREATE TWO MORE COLUMNS/VECTORS FOR ERRORS AND CLASSES
                    # Create new column for errors
                    ds10['Error'] = ds10['{} UFS'.format(Pred)] - ds10['{} Obs'.format(Pred)]

                    if Classes == 3:
                        # Find ranges for classes (this uses 3 classes)
                        q1 = ds10['Error'].quantile(0.33)
                        q2 = ds10['Error'].quantile(0.67)

                        # Create Classes
                        ds10['Class'] = np.digitize(ds10['Error'], [q1, q2])  # convert to classes based on breaks in q
                        ds_UFSp = ds10

                    elif Classes == 5:
                        q = np.quantile(ds10['Error'], np.arange(1, Classes) / Classes)  # returns quantile breaks
                        ds10['Class'] = np.digitize(ds10['Error'], q)
                        ds_UFSp = ds10

                    # PART 4: SEPARATE DATA
                    # Create array of all input maps
                    x = 0
                    ALLx = []
                    while x < len(ds['{} obs'.format(variable)]):
                        ALLx += [ds['{} obs'.format(variable)][x]]
                        x = x + 1
                    ALLx = np.array(ALLx)

                    # Create array of all classes for input maps
                    y = 0
                    ALLy = []
                    while y < len(ds_UFSp['Class']):
                        ALLy += [ds_UFSp['Class'][y]]
                        y = y + 1
                    ALLy = np.array(ALLy)
                    ALLy = ALLy.astype(int)

                    # Split data in training and validation data
                    xt_stt_x = ALLx[TV_dict[KEY][2]:TV_dict[KEY][3]]
                    xt_end_x = ALLx[TV_dict[KEY][4]:TV_dict[KEY][5]]
                    xt_stt_y = ALLy[TV_dict[KEY][2]:TV_dict[KEY][3]]
                    xt_end_y = ALLy[TV_dict[KEY][4]:TV_dict[KEY][5]]

                    x_train = np.concatenate((xt_stt_x, xt_end_x), axis=None)
                    y_train = np.concatenate((xt_stt_y, xt_end_y), axis=None)
                    x_val = ALLx[TV_dict[KEY][0]:TV_dict[KEY][1]]
                    y_val = ALLy[TV_dict[KEY][0]:TV_dict[KEY][1]]

                    # Change x_train and x_val shape if we wanted to decrease input map resolution
                    if dec == 1:
                        x_train = x_train[:, ::4, ::4]
                        x_val = x_val[:, ::4, ::4]
                        x_train_shp = x_train.reshape(TV_dict[KEY][6], len(lats[::4]) * len(lons[::4]))
                        x_val_shp = x_val.reshape(TV_dict[KEY][7], len(lats[::4]) * len(lons[::4]))
                    else:
                        x_train_shp = x_train.reshape(TV_dict[KEY][6], len(lats) * len(lons))
                        x_val_shp = x_val.reshape(TV_dict[KEY][7], len(lats) * len(lons))

                    # # # # # # # # # # BUILD NEURAL NETWORK # # # # # # # # # #
                    es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=PATIENCE, mode=mode,
                                                                   restore_best_weights=True, verbose=0)
                    callbacks = [es_callback]
                    if Reg == 'all_ep':
                        callbacks = False

                    model, loss_function = make_model(nodes, x_train_shp, RIDGE1, DROPOUT, NP_SEED, LR, y_train)

                    # y_train one-hot labels
                    enc = preprocessing.OneHotEncoder()
                    onehotlabels = enc.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
                    hotlabels_train = onehotlabels[:, :model.output_shape[-1]]

                    # y_val one-hot labels
                    onehotlabels = enc.fit_transform(np.array(y_val).reshape(-1, 1)).toarray()
                    hotlabels_val = onehotlabels[:, :model.output_shape[-1]]

                    # # # # # # # # # # TRAIN NETWORK # # # # # # # # # #
                    start_time = time.time()
                    history = model.fit(x_train_shp, hotlabels_train,
                                        validation_data=(x_val_shp, hotlabels_val),
                                        batch_size=batch_size,
                                        epochs=epochs,
                                        shuffle=True,  # shuffle data before each epoch
                                        verbose=0,
                                        callbacks=callbacks)
                    stop_time = time.time()

                    # # # # # # # # # # # PREDICTION & OUTPUT INFO # # # # # # # # # #

                    # Grab loss and acc values from training
                    out = history.history
                    out_list = list(out.items())

                    # training loss
                    loss = out_list[0]
                    loss = loss[1]

                    # acc of validation
                    acc = out_list[5]
                    acc = np.array(acc[1]) * 100

                    # Confidences
                    Conf_all = model.predict(x_val_shp)  # Confidences for all classes
                    WConf = np.argmax(Conf_all, axis=1)  # index of winning confidence
                    Conf = np.amax(Conf_all, 1)  # array of just the winning confidences

                    # Make an array of all the times when our array is correct [1] or not [0]
                    Correct_R = []
                    k = 0
                    while k < len(WConf):
                        if WConf[k] == y_val[k]:
                            Correct_R += [1]
                        else:
                            Correct_R += [0]
                        k = k + 1
                    Correct_R = np.array(Correct_R)

                    # Organize confidences
                    idx = np.arange(TV_dict[KEY][0], TV_dict[KEY][1], 1)  # array of idx of map (0 is first map in val)
                    hit_miss1 = np.stack((Correct_R, Conf, y_val, WConf, idx), axis=-1)
                    hit_miss = hit_miss1[np.argsort(hit_miss1[:, 1])]  # sort from least to most confident

                    # Add values to HM array
                    for HMV in range(hit_miss_vars):
                        for HML in range(hit_miss_len):
                            if HML == 208 and len(idx) < hit_miss_len:
                                HM_Data[seedcount][TVc][HMV][HML] = 0
                            else:
                                HM_Data[seedcount][TVc][HMV][HML] = hit_miss[HML][HMV]

            # loop through seeds and TV set-ups - POOLING STEP
            for xseed in range(seedr):
                for ytv in range(TV_tot):

                    # Convert to pd df
                    HM_dfx = pd.DataFrame(np.transpose(HM_Data[xseed][ytv]))
                    HM_dfx.columns = ['CorrOrNo', 'Conf', 'ActClass', 'WinClass', 'Bmaps']

                    # remove last row of df's where we added an extra row for balance
                    if ytv == 0 or ytv == 2 or ytv == 4:
                        HM_dfx.drop(HM_dfx.tail(1).index, inplace=True)

                    # stack all arrays together
                    if ytv == 0:
                        HM_df_all = HM_dfx
                    else:
                        HM_df_all = pd.concat([HM_df_all, HM_dfx], ignore_index=True)

                # Sort array and convert to numpy
                HM_df_all = HM_df_all.sort_values(by=['Conf'])
                HM_np = HM_df_all.to_numpy()

                # Get necessary data
                Bmaps = HM_np[:, 4]  # idx's of maps (least to most confident)
                WinClass = HM_np[:, 3]  # Predicted Classes (least to most confident)
                ActClass = HM_np[:, 2]  # Actual Classes (least to most confident)
                CorrOrNo = HM_np[:, 0]  # array for if pred correct [1] or not [0] (least to most confident)

                # # # # # # # # # # # ACCURACY & HEAT MAP INFO # # # # # # # # # #

                # Iterate through accuracy dictionary
                for key, item in Acc_main_dict.items():
                    Bmaps20 = Bmaps[int(len(Bmaps) * item[1]):]  # top 20% idx's - .8 <-> top 20% data
                    WinClass20 = WinClass[int(len(WinClass) * item[1]):]  # Predicted class
                    ActClass20 = ActClass[int(len(ActClass) * item[1]):]  # Actual Class
                    CorrOrNo20 = CorrOrNo[int(len(CorrOrNo) * item[1]):]  # Correct (1) or not (0) ?

                    # list for heatmaps
                    grid_dict[key][0] += [Bmaps20]
                    grid_dict[key][1] += [CorrOrNo20]
                    grid_dict[key][2] += [WinClass20]
                    grid_dict[key][5] += [ActClass20]

                    # Create pd arrays based on season
                    sum_pd = idx_all[idx_all["Season"] == 'Summer']
                    fall_pd = idx_all[idx_all["Season"] == 'Fall']
                    wint_pd = idx_all[idx_all["Season"] == 'Winter']
                    spr_pd = idx_all[idx_all["Season"] == 'Spring']

                    # Create list of seasonal pd arrays
                    X_pd = [sum_pd, fall_pd, wint_pd, spr_pd]

                    # Base pandas array for calculating accuracies
                    CorrCount_pd = pd.DataFrame({'Corr?': CorrOrNo20, 'WinClass': WinClass20,
                                                 'ActClass': ActClass20, 'Index': Bmaps20})

                    # # # # # # # # # # # ACCURACY MAP - ALL SAMPLES # # # # # # # # # #
                    CorrCount = np.count_nonzero(CorrOrNo20 == 1)
                    grid_dict[key][4][0] += [CorrCount / len(CorrOrNo20)]

                    # Calculate Accuracy for each Season
                    for ij in range(4):  # 4 seasons

                        # Get specific season
                        szn_pd = pd.merge(X_pd[ij], CorrCount_pd, on='Index')

                        # accuracy calculation
                        if 1 in szn_pd['Corr?'].values:  # Check if there's any correct values for this szn / class
                            CorrCountsumX = szn_pd['Corr?'].value_counts()[1]
                            grid_dict[key][4][7 + ij] += [CorrCountsumX / len(szn_pd['Index'])]
                        else:
                            grid_dict[key][4][7 + ij] += [np.nan]

                    # # # # # # # # # # # ACCURACY MAP - BY CLASS # # # # # # # # # #
                    for iii in range(Classes):

                        # Get specific class
                        CorrCountX_pd = CorrCount_pd[CorrCount_pd["WinClass"] == iii]

                        # Correct samples only (for calculating accuracy)
                        CorrCountX_list = CorrCountX_pd.iloc[:, 0]
                        CorrCountX = np.count_nonzero(CorrCountX_list == 1)

                        # Calculate Accuracy for Predicted Class
                        if len(CorrCountX_list) == 0:
                            grid_dict[key][4][iii + 1] += [np.nan]
                        else:
                            grid_dict[key][4][iii + 1] += [CorrCountX / len(CorrCountX_list)]

                        # Calculate Accuracy for Actual Class
                        CorrCountX_pd_a = CorrCount_pd[CorrCount_pd["ActClass"] == iii]
                        CorrCountX_list_a = CorrCountX_pd_a.iloc[:, 0]
                        CorrCountX_a = np.count_nonzero(CorrCountX_list_a == 1)
                        if len(CorrCountX_list_a) == 0:
                            grid_dict[key][4][iii + 4] += [np.nan]
                        else:
                            grid_dict[key][4][iii + 4] += [CorrCountX_a / len(CorrCountX_list_a)]

                        # Calculate Accuracy for each Season
                        for ii in range(4):  # 4 seasons

                            # Get specific season
                            szn_pd = pd.merge(X_pd[ii], CorrCountX_pd, on='Index')

                            # Accuracy calculation
                            if 1 in szn_pd['Corr?'].values:  # Check if correct values for this szn/class exist
                                CorrCountsumX = szn_pd['Corr?'].value_counts()[1]
                                grid_dict[key][4][11 + iii * 4 + ii] += [CorrCountsumX / len(szn_pd['Index'])]
                            else:
                                grid_dict[key][4][11 + iii * 4 + ii] += [np.nan]

                        # Define which percent accuracy we're looking at
                        grid_dict[key][3] += grid_dict[key][4]

            # Average each season / class over all the seeds (for each location)
            for cc, key in enumerate(Acc_main_dict):
                for abc in range(acc_map_tot):
                    Acc_Map_Data[cc][abc][c1][c2] = np.nanmean(grid_dict[key][3][abc])

            # Average each season / class over all the seeds (for each location)
            for key, item in Acc_main_dict.items():
                Acc_main_dict[key][2] += [grid_dict[key][0]]
                Acc_main_dict[key][3] += [grid_dict[key][1]]
                Acc_main_dict[key][4] += [grid_dict[key][2]]
                Acc_main_dict[key][5] += [grid_dict[key][5]]

# Organize necessary info for hmaps
for key, item in Acc_main_dict.items():

    # unravel list of arrays into just one array
    hmap_idx = np.concatenate(Acc_main_dict[key][2]).ravel()
    hmap_CN = np.concatenate(Acc_main_dict[key][3]).ravel()
    hmap_cls = np.concatenate(Acc_main_dict[key][4]).ravel()
    hmap_act = np.concatenate(Acc_main_dict[key][5]).ravel()

    # convert to pandas
    hmap_key_dfX = pd.DataFrame(data=[hmap_idx, hmap_CN, hmap_cls, hmap_act])
    hmap_key_df = hmap_key_dfX.T
    hmap_key_df.columns = ['Index_{}'.format(key), 'CorrOrNo_{}'.format(key), 'Class_{}'.format(key),
                           'ActCls_{}'.format(key)]

    # Merge dfs
    if key == '20':
        hmap_df = hmap_key_df
    else:
        hmap_df = pd.concat([hmap_df, hmap_key_df], axis=1)

print(hmap_df[:5])
# %%
# Save files
hmap_df.to_csv('Hmap_Data_Thresh_{}_{}.csv'.format(Reg, Pred))
np.save('Acc_Map_Data_Thresh_{}_{}.npy'.format(Reg, Pred), Acc_Map_Data)

