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

# Define Input Map variable
variable = 'olr'
if variable == 'olr':
    lat_slice = slice(-25, 25)  # lat and lon region make up the Tropics
    lon_slice = slice(60, 300)
    Map = 1
elif variable == 'u200':
    lat_slice = slice(0, 75)  # lat and lon region make up NH over Pacific
    lon_slice = slice(120, 300)
    Map = 2

# Define predictor
Pred = 'h500'
if Pred == 'prcp':
    Predictor = 0
elif Pred == 'h500':
    Predictor = 1

# Helps sort data
cdata = 'clima_gefs'
LT_tot = 35  # how many total lead times are there in the UFS forecast?

# Decrease input resolution?
dec = 0  # 0 if normal, 1 if decrease

# Whole map or just a subset?
Reg = 'full'  # 'full', '4': SW (h500 - All classes), 'NW': NW (h500 - Underestimates), 'SE': SE (h500 - Overestimates)

# Dict for region (seeds, lat, lon, class name, class number, map size, epoch size, seeds)
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
            'test': [slice(33.9, 37.6), slice(255.4, 260.6), 'All Classes', [0, 1, 2], 0, 10000, [92, 95]]}

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
ds_obs1_base = xr.open_dataset('/Users/jcahill4/DATA/{}/Obs/{}/clima_{}.nc'.format(Pred, cdata, Pred))
ds_UFS_base = xr.open_mfdataset('/Users/jcahill4/DATA/{}/UFS/{}/*.nc'.format(Pred, cdata), concat_dim='time',
                                combine='nested')
CONUS = ds_obs1_base['h500'].sel(lat=Reg_dict[Reg][0], lon=Reg_dict[Reg][1])
CONUS_lats = CONUS.lat.values[::4]
CONUS_lons = CONUS.lon.values[::4]
CONUS_lons = CONUS_lons[:len(CONUS_lons) - 1]
print(CONUS_lats)
print(CONUS_lons)

# WINDS
dfu = xr.open_mfdataset('/Users/jcahill4/DATA/u200/Obs/daily_gefs/*.nc', concat_dim='time', combine='nested')
dfv = xr.open_mfdataset('/Users/jcahill4/DATA/v200/Obs/daily_gefs/*.nc', concat_dim='time', combine='nested')
dfu_ufs = xr.open_mfdataset('/Users/jcahill4/DATA/u200/UFS/raw_gefs/*.nc', concat_dim='time', combine='nested')
dfv_ufs = xr.open_mfdataset('/Users/jcahill4/DATA/v200/UFS/raw_gefs/*.nc', concat_dim='time', combine='nested')

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
acc_pcents = 3  # 20%, 30% and all (see acc_main_dict)
TV_tot = 5
Acc_Map_Data = np.empty((TV_tot, acc_pcents, acc_map_tot, len(CONUS_lons), len(CONUS_lats)))
Acc_Map_Data[:] = np.nan

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
           'TT5': [834, 2000, 0, 834, 2000, 2001, 834, 208, 'TT5', 'Validation - 5th Quintile']}

# 3 is for different classes, 4 is for different types of hmaps (correct, class, etc), 3x9 is for hmap shape
TV_hmap_mean = np.empty((TV_tot, acc_pcents, 3, 4, 3, 9))
TV_hmap_mean[:] = np.nan

# %% MEGA LOOP
for TVc, KEY in enumerate(TV_dict):

    # Create a pandas arrays that has index, dates, and obs maps of validation data
    val_idx = np.arange(0, TV_dict[KEY][7])
    obs_maps = ds['{} obs'.format(variable)][TV_dict[KEY][0]:TV_dict[KEY][1]]
    date_list = ds['time'][TV_dict[KEY][0]:TV_dict[KEY][1]]
    idx_date = pd.DataFrame({'Index': val_idx, 'dates': date_list, 'obs': obs_maps})

    # Merge with time data
    idx_all = pd.merge(idx_date, mdata_ymd, on='dates')

    # # # # # # # # # #  ACC SPLITS (20, 30, ALL) # # # # # # # # # #

    # We want to save accuracies for multiple different top %s
    # key: [fig title, % ignored for acc, idx list, corr_or_no list, class list]
    Acc_main_dict = {'20': ['20% Most Confident and Correct Samples', 0.8, [], [], []],
                     '30': ['30% Most Confident and Correct Samples', 0.7, [], [], []],
                     'All': ['All Correct Samples', 0, [], [], []]}


    # Set up a counter, so we can determine how many grid points we're running for
    counter = 0

    # # # # # # # # # #  RUN NEURAL NETWORK FOR EACH LOCATION AND SEED # # # # # # # # # #
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
                grid_dict = {'20': [[], [], [], [], [[] for _ in range(acc_map_tot)]],
                             '30': [[], [], [], [], [[] for _ in range(acc_map_tot)]],
                             'All': [[], [], [], [], [[] for _ in range(acc_map_tot)]]}

                # Loop through seeds
                for x in Reg_dict[Reg][6]:

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

                    # # # # # # # # # # READ IN DATA - TRIPLE CHECKED # # # # # # # # # #

                    # PART1: MERGE UFS AND OBS DATA SO THEY MATCH
                    z = 0
                    while z < len(ds_UFS1.time.values) / LT_tot:

                        # Grab UFS data between lead_times
                        UFS = ds_UFS1[z * LT_tot:z * LT_tot + LT_tot]
                        UFS = UFS[lead_time1:lead_time2 + 1]

                        # Convert UFS data to pandas
                        UFS = list(zip(UFS.time.values, UFS.values.flatten()))
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
                    idx = np.arange(0, len(Conf), 1)  # array of idx of map (0 is first map in validation)
                    hit_miss1 = np.stack((Correct_R, Conf, y_val, WConf, idx), axis=-1)
                    hit_miss = hit_miss1[np.argsort(hit_miss1[:, 1])]  # sort from least to most confident

                    Bmaps = hit_miss[:, 4]  # idx's of maps (least to most confident)
                    WinClass = hit_miss[:, 3]  # Predicted Classes (least to most confident)
                    ActClass = hit_miss[:, 2]  # Actual Classes (least to most confident)
                    CorrOrNo = hit_miss[:, 0]  # array for if pred correct [1] or not [0] (least to most confident)

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
                                if 1 in szn_pd['Corr?'].values:    # Check if correct values for this szn/class exist
                                    CorrCountsumX = szn_pd['Corr?'].value_counts()[1]
                                    grid_dict[key][4][11 + iii * 4 + ii] += [CorrCountsumX / len(szn_pd['Index'])]
                                else:
                                    grid_dict[key][4][11 + iii * 4 + ii] += [np.nan]

                            # Define which percent accuracy we're looking at
                            grid_dict[key][3] += grid_dict[key][4]

                # Average each season / class over all the seeds
                for cc, key in enumerate(Acc_main_dict):
                    for abc in range(acc_map_tot):
                        Acc_Map_Data[TVc][cc][abc][c1][c2] = np.nanmean(grid_dict[key][3][abc])

                for key, item in Acc_main_dict.items():
                    Acc_main_dict[key][2] += [grid_dict[key][0]]
                    Acc_main_dict[key][3] += [grid_dict[key][1]]
                    Acc_main_dict[key][4] += [grid_dict[key][2]]

    # # # # # # # # #  ACCURACY MAPS - PLOTTING # # # # # # # # # #

    # Map inputs
    ecolor = 'dimgray'  # makes lakes and borders light gray
    fsize = 24  # fontsize
    lons_p = np.append(CONUS_lons, CONUS_lons[-1] + 2) - 1  # pcolor are boxes, so set start and end pts of the boxes
    lats_p = np.append(CONUS_lats, CONUS_lats[-1] + 2) - 1  # lons_p / lats_p make it so box centers are lons / lats
    Seas_Name = [' - Summer', ' - Fall', ' - Winter', ' - Spring', '']
    Cls_Name = ['(UFS Underestimates)', '(UFS Accurate Estimates)', '(UFS Overestimates)', '(All Samples)']

    # Make Each Map
    for acm_mid, key in enumerate(Acc_main_dict):
        for acm in range(Acc_Map_Data.shape[2]):

            # Set-up maps
            fig = plt.figure(figsize=(10, 8))

            # Define map type and its size
            ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
            ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

            # Set levels and output name variables
            if acm == 0:
                vmin = .325  # all seasons - all classes
                vmax = .5
                vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.4125, vmax=vmax)
                seas = 4
                clss = 3
                Pr = 0
            elif acm < 7:
                vmin = .3  # all seasons - specific class
                vmax = .7
                vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.5, vmax=vmax)
                seas = 4
                clss = (acm + 2) % 3
                if acm < 4:
                    Pr = 0  # Predicted Class (0) or True Class (1)
                else:
                    Pr = 1
            elif acm < 11:
                if key == 'All':
                    vmin = .35  # specific season - all classes
                    vmax = .6
                    vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.475, vmax=vmax)
                else:
                    vmin = .35  # specific season - all classes
                    vmax = .6
                    vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.475, vmax=vmax)
                seas = (acm + 1) % 4
                clss = 3
                Pr = 0
            else:
                if key == 'All':
                    vmin = 0.0  # specific season - specific class
                    vmax = 0.6
                    vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.3, vmax=vmax)
                else:
                    vmin = 0.4  # specific season - specific class
                    vmax = 1.0
                    vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.7, vmax=vmax)
                seas = (acm + 1) % 4
                clss = (acm - 11) // 4
                Pr = 0

            # Add features
            # ax.add_feature(cfeature.LAND, color='white')
            ax.add_feature(cfeature.COASTLINE)
            ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
            ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
            ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

            # plot
            cf = ax.pcolor(lons_p - 180, lats_p, Acc_Map_Data[TVc][acm_mid][acm].T, vmin=vmin, vmax=vmax, norm=vmid,
                           cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))

            # plot info
            cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)
            plt.title(
                'Accuracy of Predicting {} Errors in the UFS{}\n{}\nLead Time of 10-14 days {}'.format(Pred,
                                                                                                       Seas_Name[seas],
                                                                                                       Acc_main_dict[
                                                                                                           key][0],
                                                                                                       Cls_Name[clss]),
                fontsize=fsize)
            plt.tight_layout()
            plt.savefig(
                'AccMap_{}_Re{}Se{}Cls{}Pr{}Lt{}{}TV{}N{}Lr{}Acc{}'.format(Pred, Reg, seas, clss, Pr, lead_time1,
                                                                           lead_time2, TV_dict[KEY][8], nodes,
                                                                           str(LR).split('.')[1], key), dpi=300)
            plt.show()

    print('Done2_{}'.format(TVc))

    # # # # # # # # # #  HEAT MAPS - PLOTTING # # # # # # # # # #
    for cc, key in enumerate(Acc_main_dict):

        # Specify which class we're interested in plotting
        for class_num in Reg_dict[Reg][3]:

            # Make a "total" hmap that can we can divide by to normalize the hmaps
            hmap_main = np.zeros((3, 9))
            for x in np.arange(-1, 2, 1):
                ENSOs = idx_all.loc[idx_all['E_Phase'] == x]
                for y in np.arange(0, 9, 1):
                    MJO_ENSO = ENSOs.loc[ENSOs['M_Phase'] == y]
                    hmap_main[x + 1, y] = len(MJO_ENSO['Index'])

            # Organize necessary info for hmaps
            for xy in range(counter):

                # flatten each list (each seed has own array)
                flat_idx = np.concatenate(Acc_main_dict[key][2][xy]).ravel()  # index
                flat_CNO = np.concatenate(Acc_main_dict[key][3][xy]).ravel()  # correct or not?
                flat_CLS = np.concatenate(Acc_main_dict[key][4][xy]).ravel()  # class

                # Then combine the lists within our specified region
                if xy == 0:
                    flat_idx_all = flat_idx
                    flat_CNO_all = flat_CNO
                    flat_CLS_all = flat_CLS
                else:
                    flat_idx_all = np.concatenate([flat_idx_all, flat_idx])
                    flat_CNO_all = np.concatenate([flat_CNO_all, flat_CNO])
                    flat_CLS_all = np.concatenate([flat_CLS_all, flat_CLS])

            # Separate based on class and correct samples
            all_all = pd.DataFrame({'Index': flat_idx_all, 'CorrOrNo': flat_CNO_all, 'Class': flat_CLS_all})
            all_corr = all_all[all_all['CorrOrNo'] == 1].reset_index(drop=True)
            class_all = all_all[all_all['Class'] == class_num].reset_index(drop=True)
            class_corr = class_all[class_all['CorrOrNo'] == 1].reset_index(drop=True)

            # Put them in a list, so we can loop through them
            info_list = [all_all, all_corr, class_all, class_corr]

            # Output Names
            out_names = ['all_all', 'all_corr', '{}_all'.format(class_num), '{}_corr'.format(class_num)]

            # Organize and plot each hmap
            for inf in range(len(info_list)):

                # Count indices
                info_count = info_list[inf].groupby(['Index'])['Index'].count()
                info_count_df = info_count.to_frame()
                info_idx = info_count_df.index  # Indices
                info_counts = info_count_df['Index'].reset_index(drop=True)  # Count

                # Convert index and counts into pandas array
                countpd = pd.DataFrame({'Index': info_idx, 'count': info_counts})

                # Merge with ENSO and MJO info
                EMJO = pd.merge(countpd, idx_all, on='Index')

                # Create hmap to show frequency of ENSO and MJO
                hmap = np.zeros((3, 9))
                for x in np.arange(-1, 2, 1):
                    ENSOs = EMJO.loc[EMJO['E_Phase'] == x]
                    for y in np.arange(0, 9, 1):
                        MJO_ENSO = ENSOs.loc[ENSOs['M_Phase'] == y]
                        hmap[x + 1, y] = len(MJO_ENSO['Index'])
                hmap = hmap / hmap_main

                # Save hmap for means
                for xh in range(TV_hmap_mean.shape[4]):
                    for yh in range(TV_hmap_mean.shape[5]):
                        TV_hmap_mean[TVc][cc][class_num][inf][xh][yh] = hmap[xh][yh]

                # Plot hmap
                plt.imshow(hmap, cmap=plt.cm.Reds)
                plt.xlabel('MJO Phase')
                plt.ylabel('ENSO Phase')
                plt.title('Frequency of Oscillation Phase\n{}'.format(Acc_main_dict[key][0]), fontsize=16)
                plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
                plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
                plt.colorbar(shrink=0.75, label='Frequency')
                plt.tight_layout()
                plt.savefig(
                    'Heatmap_{}_{}_Re{}_Lt{}{}TV{}N{}Lr{}Acc{}'.format(out_names[inf], Pred, Reg, lead_time1, lead_time2,
                                                                     TV_dict[KEY][8], nodes, str(LR).split('.')[1],
                                                                     key),
                    dpi=300)
                plt.show()



# %% OVERALL ACCURACY AND HEAT MAPS

#  ACCURACY MAPS #

# Mean acc maps
Acc_Map_Mean = np.mean(Acc_Map_Data, axis=0)

# Map inputs
ecolor = 'dimgray'  # makes lakes and borders light gray
fsize = 24  # fontsize
lons_p = np.append(CONUS_lons, CONUS_lons[-1] + 2) - 1  # pcolor are boxes, so set start and end pts of the boxes
lats_p = np.append(CONUS_lats, CONUS_lats[-1] + 2) - 1  # lons_p / lats_p make it so box centers are lons / lats
Seas_Name = [' - Summer', ' - Fall', ' - Winter', ' - Spring', '']
Cls_Name = ['(UFS Underestimates)', '(UFS Accurate Estimates)', '(UFS Overestimates)', '(All Samples)']

# Make Each Map
for acm_mid, key in enumerate(Acc_main_dict):
    for acm in range(Acc_Map_Mean.shape[1]):

        # Set-up maps
        fig = plt.figure(figsize=(10, 8))

        # Define map type and its size
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

        # Set levels and output name variables
        if acm == 0:
            vmin = .325  # all seasons - all classes
            vmax = .5
            vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.4125, vmax=vmax)
            seas = 4
            clss = 3
            Pr = 0
        elif acm < 7:
            vmin = .3  # all seasons - specific class
            vmax = .7
            vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.5, vmax=vmax)
            seas = 4
            clss = (acm + 2) % 3
            if acm < 4:
                Pr = 0  # Predicted Class (0) or True Class (1)
            else:
                Pr = 1
        elif acm < 11:
            vmin = .35  # specific season - all classes
            vmax = .6
            vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.475, vmax=vmax)
            seas = (acm + 1) % 4
            clss = 3
            Pr = 0
        else:
            vmin = .4  # specific season - specific class
            vmax = 1.0
            vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.7, vmax=vmax)
            seas = (acm + 1) % 4
            clss = (acm - 11) // 4
            Pr = 0

        # Add features
        # ax.add_feature(cfeature.LAND, color='white')
        ax.add_feature(cfeature.COASTLINE)
        ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
        ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
        ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

        # plot
        cf = ax.pcolor(lons_p - 180, lats_p, Acc_Map_Mean[acm_mid][acm].T, vmin=vmin, vmax=vmax, norm=vmid,
                       cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))

        # plot info
        cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)
        plt.title(
            'Accuracy of Predicting {} Errors in the UFS{}\n{}\nLead Time of 10-14 days {}'.format(Pred,
                                                                                                   Seas_Name[seas],
                                                                                                   Acc_main_dict[key][
                                                                                                       0],
                                                                                                   Cls_Name[clss]),
            fontsize=fsize)
        plt.tight_layout()
        plt.savefig(
            'AccMapOVRALL_{}_Re{}Se{}Cls{}Pr{}Lt{}{}N{}Lr{}Acc{}'.format(Pred, Reg, seas, clss, Pr, lead_time1,
                                                                         lead_time2,
                                                                         nodes, str(LR).split('.')[1], key),
            dpi=300)
        plt.show()

print('Done2_{}'.format(TVc))

# %% # HEAT MAPS OVERALL #

# Mean hmaps
TV_Mean = np.mean(TV_hmap_mean, axis=0)

for cc, key in enumerate(Acc_main_dict):

    # Specify which class we're interested in plotting
    for class_num in Reg_dict[Reg][3]:

        # Organize and plot each hmap
        for inf in range(len(info_list)):

            hmap = TV_Mean[cc][class_num][inf]

            # Output Names
            out_names = ['all_all', 'all_corr', '{}_all'.format(class_num), '{}_corr'.format(class_num)]

            # Plot hmap
            plt.imshow(hmap, cmap=plt.cm.Reds)
            plt.xlabel('MJO Phase')
            plt.ylabel('ENSO Phase')
            plt.title('Frequency of Oscillation Phase\n{}'.format(Acc_main_dict[key][0]), fontsize=16)
            plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
            plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
            plt.colorbar(shrink=0.75, label='Frequency')
            plt.tight_layout()
            plt.savefig(
                'HeatmapOVRALL_{}_{}_Re{}_Lt{}{}N{}Lr{}Acc{}'.format(out_names[inf], Pred, Reg, lead_time1, lead_time2,
                                                                     nodes, str(LR).split('.')[1], key),
                dpi=300)
            plt.show()


# %% Save necessary numpy arrays

np.save('Acc_Map_Data.npy', Acc_Map_Data)
np.save('TV_hmap_mean.npy', TV_hmap_mean)
print(counter)
