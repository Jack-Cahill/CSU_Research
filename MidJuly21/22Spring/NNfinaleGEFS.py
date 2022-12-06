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
seeds = [92, 95, 100, 137, 141, 142]
#seeds = np.arange(88, 89, 1)

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
bignum = 862  # selects testing / training data
smlnum = 180  # " "
LT_tot = 35  # how many total lead times are there in the UFS forecast?

# Decrease input resolution?
dec = 0  # 0 if normal, 1 if decrease

# Whole map or just a subset?
Reg = '4'  # 'full', '4': SW (h500 - All classes), 'NW': NW (h500 - Underestimates), 'SE': SE (h500 - Overestimates)

# Dict for region (lat, lon, Class name, class number, map size, epoch size)
Reg_dict = {'full': [slice(23.9, 49.6), slice(235.4, 293.6), 'All Classes', 0, 6, 10_000],
            '4': [slice(33.9, 37.6), slice(255.4, 260.6), 'All Classes', 0, 0, 10_000],
            'NW': [slice(23.9, 49.6), slice(235.4, 293.6), 'Underestimates', 0, 7, 10_000],
            'SE': [slice(23.9, 49.6), slice(235.4, 293.6), 'Overestimates', 2, 6, 10_000],
            'SW': [slice(23.9, 49.6), slice(235.4, 293.6), 'Underestimates', 0, 7, 10_000],
            'full_epoch': [slice(23.9, 49.6), slice(235.4, 293.6), 'All Classes', 0, 6, 1000]}

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

# # # # # # # # # #  MJO + ENSO COMPARISON SET-UP # # # # # # # # # #

# Create a pandas arrays that has index, dates, and obs maps of validation data
val_idx = np.arange(0, smlnum)
obs_maps = ds['{} obs'.format(variable)][bignum:]
date_list = ds['time'][bignum:]
idx_date = pd.DataFrame({'Index': val_idx, 'dates': date_list, 'obs': obs_maps})

# Merge MJO and ENSO info
mdata = pd.read_csv('/Users/jcahill4/Downloads/mjo_phase1.csv')
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

# Merge with time data
idx_all = pd.merge(idx_date, mdata_ymd, on='dates')

# # # # # # # # # # PANDAS PRED OBS (FOR OUT MAPS) # # # # # # # # # #

Pred_sliceLat = slice(-30, 65)
Pred_sliceLon = slice(60, 300)

# Turn preds into a pandas array
ds_obs1_base_Pred = ds_obs1_base[Pred].sel(lat=Pred_sliceLat, lon=Pred_sliceLon)
ds_obs1_base_l = list(zip(ds_obs1_base_Pred.time.values, ds_obs1_base_Pred.values))
pred_obs = pd.DataFrame(ds_obs1_base_l)
pred_obs.columns = ['dates', '{} obs'.format(Pred)]

# drop data that's not validation to speed up runtime
pred_obs = pred_obs[~(pred_obs['dates'] < '2016-07-01')]
pred_obs = pred_obs.reset_index(drop=True)

# # # # # # # # # #  WINDS # # # # # # # # # #

dfu = xr.open_mfdataset('/Users/jcahill4/DATA/u200/Obs/daily_gefs/*.nc', concat_dim='time', combine='nested')
dfv = xr.open_mfdataset('/Users/jcahill4/DATA/v200/Obs/daily_gefs/*.nc', concat_dim='time', combine='nested')

# # # # # # # # # #  ACCURACY MAP SET-UP # # # # # # # # # #

# 23 is for the 23 maps I'll make
# 1 all samples, 3 classes (predicted class), 3 classes (actual class), 4 all samples each season, 3*4 (class*season)
acc_map_tot = 23
Acc_Map_Data = np.empty((acc_map_tot, len(CONUS_lons), len(CONUS_lats)))
Acc_Map_Data[:] = np.nan

# # # # # # # # # #  REGIONS # # # # # # # # # #

# If we're not running a full map, save data for the region / group of 4 grid points
if Reg != 'full':
    loc_arr = []  # idx
    loc_arrCN = []  # Correct or No
    loc_arrCLS = []  # Class

    # Set up a counter, so we can determine how many grid points we're running for
    counter = 0

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

# %% # # # # # # # # #  RUN NEURAL NETWORK FOR EACH LOCATION AND SEED # # # # # # # # # #
for c1, xxx in enumerate(CONUS_lons):
    for c2, xx in enumerate(CONUS_lats):

        # Check if the lats and lons fall within specified region - if so, run NN
        df_chk = reg_lon_lat.loc[reg_lon_lat.latitude == xx]
        if (xxx - 360) in df_chk.longitude.unique():

            # count many grid points we're running for and print grid point
            counter = counter + 1
            print(xx, xxx)

            # Create lists of lists (inner lists contain the average for each seed) (outer lists are for each cls / szn)
            acc_lists = [[] for _ in range(acc_map_tot)]

            # arrays for 4 adjacent grid points
            if Reg != 0:
                locos = []
                locosCN = []
                locosCLS = []

            # Loop through seeds
            for x in seeds:

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
                if TTLoco == 0:
                    x_train = ALLx[:bignum]  # input maps w/ shape (bignum, 50, 241) : (# cases, lats, lons)
                    y_train = ALLy[:bignum]  # predicted error classes of the area of interest shape (bignum)
                    x_val = ALLx[bignum:]
                    y_val = ALLy[bignum:]
                else:
                    x_val = ALLx[:smlnum]
                    y_val = ALLy[:smlnum]
                    x_train = ALLx[smlnum:]
                    y_train = ALLy[smlnum:]

                # Change x_train and x_val shape if we wanted to decrease input map resolution
                if dec == 1:
                    x_train = x_train[:, ::4, ::4]
                    x_val = x_val[:, ::4, ::4]
                    x_train_shp = x_train.reshape(bignum, len(lats[::4]) * len(lons[::4]))
                    x_val_shp = x_val.reshape(smlnum, len(lats[::4]) * len(lons[::4]))
                else:
                    x_train_shp = x_train.reshape(bignum, len(lats) * len(lons))
                    x_val_shp = x_val.reshape(smlnum, len(lats) * len(lons))

                # # # # # # # # # # BUILD NEURAL NETWORK # # # # # # # # # #
                # CHANGE
                es_callback = tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=PATIENCE, mode=mode,
                                                               restore_best_weights=True, verbose=0)
                callbacks = [es_callback]

                model, loss_function = make_model(nodes, x_train_shp, RIDGE1, DROPOUT, NP_SEED, LR, y_train)

                # y_train one-hot labels
                enc = preprocessing.OneHotEncoder()
                onehotlabels = enc.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
                hotlabels_train = onehotlabels[:, :model.output_shape[-1]]

                # y_val one-hot labels
                onehotlabels = enc.fit_transform(np.array(y_val).reshape(-1, 1)).toarray()
                hotlabels_val = onehotlabels[:, :model.output_shape[-1]]

                # # # # # # # # # # TRAIN NETWORK # # # # # # # # # #

                # CHANGE
                start_time = time.time()
                history = model.fit(x_train_shp, hotlabels_train,
                                    validation_data=(x_val_shp, hotlabels_val),
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    shuffle=True,  # shuffle data before each epoch
                                    verbose=0,
                                    callbacks=callbacks)
                stop_time = time.time()

                '''start_time = time.time()
                history = model.fit(x_train_shp, hotlabels_train,
                                    validation_data=(x_val_shp, hotlabels_val),
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    shuffle=True,  # shuffle data before each epoch
                                    verbose=0)
                stop_time = time.time()'''

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

                # If we want to only look at seeds that are "good enough" - cross a certain threshold, we use this
                CP_Corr_R = []
                w = 0
                while w < len(hit_miss[:, 0]):
                    CP_Corr_R += [(np.sum(hit_miss[:, 0][w:]) / len(hit_miss[:, 0][w:])) * 100]
                    w = w + 1
                CP_Corr_R = np.array(CP_Corr_R)
                CP_Corr_R = np.mean(CP_Corr_R.reshape(-1, 9), axis=1)

                # Add data to array only if the run is "good" (over 40% acc, never under 20% acc for 50% most conf runs)
                # if np.min(CP_Corr_R[-6:]) > 20.0 and np.max(CP_Corr_R) > 40.0:
                if np.min(CP_Corr_R[-6:]) > -1.0 and np.max(CP_Corr_R) > -1.0:  # This just mean grab all files
                    Bmaps20 = Bmaps[int(len(Bmaps) * 0.80):]  # top 20% idx's - .8 <-> top 20% data
                    WinClass20 = WinClass[int(len(WinClass) * 0.8):]  # Predicted class
                    ActClass20 = ActClass[int(len(ActClass) * 0.8):]  # Actual Class
                    CorrOrNo20 = CorrOrNo[int(len(CorrOrNo) * 0.8):]  # Correct (1) or not (0) ?

                    # list for heatmaps
                    locos += [Bmaps20]
                    locosCN += [CorrOrNo20]
                    locosCLS += [WinClass20]

                    # Create pd arrays based on season
                    sum_pd = idx_all[idx_all["Season"] == 'Summmer']
                    fall_pd = idx_all[idx_all["Season"] == 'Fall']
                    wint_pd = idx_all[idx_all["Season"] == 'Winter']
                    spr_pd = idx_all[idx_all["Season"] == 'Spring']

                    # Create list of seasonal pd arrays
                    X_pd = [sum_pd, fall_pd, wint_pd, spr_pd]

                    # Base pandas array for calculating accuracies
                    CorrCount_pd = pd.DataFrame({'Corr?': CorrOrNo20, 'WinClass': WinClass20, 'ActClass': ActClass20,
                                                 'Index': Bmaps20})

                    # # # # # # # # # # # ACCURACY MAP - ALL SAMPLES # # # # # # # # # #
                    CorrCount = np.count_nonzero(CorrOrNo20 == 1)
                    acc_lists[0] += [CorrCount / len(CorrOrNo20)]

                    # Calculate Accuracy for each Season
                    for ij in range(4):  # 4 seasons

                        # Get specific season
                        szn_pd = pd.merge(X_pd[ij], CorrCount_pd, on='Index')

                        # accuracy calculation
                        if 1 in szn_pd['Corr?'].values:  # Check if there's any correct values for this szn / class
                            CorrCountsumX = szn_pd['Corr?'].value_counts()[1]
                            acc_lists[7 + ij] += [CorrCountsumX / len(szn_pd['Index'])]
                        else:
                            acc_lists[7 + ij] += [np.nan]

                    # # # # # # # # # # # ACCURACY MAP - BY CLASS # # # # # # # # # #
                    for iii in range(Classes):

                        # Get specific class
                        CorrCountX_pd = CorrCount_pd[CorrCount_pd["WinClass"] == iii]

                        # Correct samples only (for calculating accuracy)
                        CorrCountX_list = CorrCountX_pd.iloc[:, 0]
                        CorrCountX = np.count_nonzero(CorrCountX_list == 1)

                        # Calculate Accuracy for Predicted Class
                        if len(CorrCountX_list) == 0:
                            acc_lists[iii + 1] += [np.nan]
                        else:
                            acc_lists[iii + 1] += [CorrCountX / len(CorrCountX_list)]

                        # Calculate Accuracy for Actual Class
                        CorrCountX_pd_a = CorrCount_pd[CorrCount_pd["ActClass"] == iii]
                        CorrCountX_list_a = CorrCountX_pd_a.iloc[:, 0]
                        CorrCountX_a = np.count_nonzero(CorrCountX_list_a == 1)
                        if len(CorrCountX_list_a) == 0:
                            acc_lists[iii + 4] += [np.nan]
                        else:
                            acc_lists[iii + 4] += [CorrCountX_a / len(CorrCountX_list_a)]

                        # Calculate Accuracy for each Season
                        for ii in range(4):  # 4 seasons

                            # Get specific season
                            szn_pd = pd.merge(X_pd[ii], CorrCountX_pd, on='Index')

                            # Accuracy calculation
                            if 1 in szn_pd['Corr?'].values:  # Check if there's any correct values for this szn / class
                                CorrCountsumX = szn_pd['Corr?'].value_counts()[1]
                                acc_lists[11 + iii * 4 + ii] += [CorrCountsumX / len(szn_pd['Index'])]
                            else:
                                acc_lists[11 + iii * 4 + ii] += [np.nan]

                # If we are throwing out certain seeds, this counts how many we throw out
                else:
                    Trash = Trash + 1

            # Average each season / class over all the seeds
            for abc in range(acc_map_tot):
                Acc_Map_Data[abc][c1][c2] = np.nanmean(acc_lists[abc])

            if Reg != 'full':
                loc_arr += [locos]
                loc_arrCN += [locosCN]
                loc_arrCLS += [locosCLS]
print('DONE')

# %% # # # # # # # # #  ACCURACY MAPS - PLOTTING # # # # # # # # # #

# Map inputs
ecolor = 'dimgray'  # makes lakes and borders light gray
fsize = 24  # fontsize
lons_p = np.append(CONUS_lons, CONUS_lons[-1] + 2) - 1  # pcolor are boxes, so set start and end pts of the boxes
lats_p = np.append(CONUS_lats, CONUS_lats[-1] + 2) - 1  # lons_p / lats_p make it so box centers are lons / lats
Seas_Name = [' - Summer', ' - Fall', ' - Winter', ' - Spring', '']
Cls_Name = ['(UFS Underestimates)', '(UFS Accurate Estimates)', '(UFS Overestimates)', '(All Samples)']

# Make Each Map
for acm in range(Acc_Map_Data.shape[0]):

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
            Pr = 0  # Predicted Class (0) or True Class (0)
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
    cf = ax.pcolor(lons_p - 180, lats_p, Acc_Map_Data[acm].T, vmin=vmin, vmax=vmax, norm=vmid,
                   cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))

    # plot info
    cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)
    plt.title('Accuracy of Predicting {} Errors in the UFS{}\nLead Time of 10-14 days {}'.format(Pred, Seas_Name[seas],
                                                                                                 Cls_Name[clss]),
              fontsize=fsize)
    plt.tight_layout()
    plt.savefig('AccMap_{}_Re{}Se{}Cls{}Pr{}Lt{}{}Tt{}N{}Lr{}'.format(Pred, Reg, seas, clss, Pr, lead_time1, lead_time2,
                                                                      TTLoco, nodes, str(LR).split('.')[1]), dpi=300)
    plt.show()
print('Done2')


# %% # # # # # # # # #  HEAT MAPS - PLOTTING # # # # # # # # # #

# Specify which class we're interested in plotting
class_num = Reg_dict[Reg][3]

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
    flat_idx = np.concatenate(loc_arr[xy]).ravel()  # index
    flat_CNO = np.concatenate(loc_arrCN[xy]).ravel()  # correct or not?
    flat_CLS = np.concatenate(loc_arrCLS[xy]).ravel()  # class

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

    # Plot hmap
    plt.imshow(hmap, cmap=plt.cm.Reds)
    plt.xlabel('MJO Phase')
    plt.ylabel('ENSO Phase')
    plt.title('Frequency of Oscillation Phase', fontsize=16)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
    plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
    plt.colorbar(shrink=0.75, label='Frequency')
    plt.tight_layout()
    plt.savefig(
        'Heatmap_{}_{}_Re{}_Lt{}{}Tt{}N{}Lr{}'.format(out_names[inf], Pred, Reg, lead_time1, lead_time2, TTLoco, nodes,
                                                      str(LR).split('.')[1]), dpi=300)
    plt.show()

# %% Looking at correct and confident samples' inputs

# Only look at samples that were correctly predicted in 75% of the samples
cidx = class_corr.groupby('Index')['Index'].transform('count')
class_corr_doop = class_corr[cidx > int(counter*0.75)]

# Only keep one of each of these samples for analysis
class_corr_drop = class_corr_doop.drop_duplicates()

# Merge to get dates
corr_conf_datesM = pd.merge(class_corr_drop, idx_all, on='Index')

# Trim the fat
corr_conf_dates = corr_conf_datesM[['dates', 'obs']]
print(corr_conf_dates)
# %%

# Plot all input maps
proj = ccrs.PlateCarree(central_longitude=180)
fig, axs = plt.subplots(6, 3, subplot_kw={'projection': proj}, figsize=(11.5, 7.5))
#fig, axs = plt.subplots(Reg_dict[Reg][4], 3, subplot_kw={'projection': proj}, figsize=(11.5, 7.5))

# axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
axs = axs.flatten()

for xmap in range(len(corr_conf_dates['dates'])):

    # lat, lon extents
    axs[xmap].set_extent([-120, 120, -25, 25], ccrs.PlateCarree(central_longitude=180))

    # Plot
    clevs = np.arange(-150, 155, 5)
    cs = axs[xmap].contourf(lons, lats, corr_conf_dates['obs'][xmap], clevs, transform=proj, cmap=plt.cm.bwr,
                            extend='both')

    # Add extra features
    axs[xmap].add_feature(cfeature.COASTLINE.with_scale('50m'))
    axs[xmap].add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)

fig.subplots_adjust(bottom=0.25, top=0.875, left=0.1, right=0.95, wspace=0.15, hspace=0.02)

# Add a colorbar axis at the bottom of the graph
cbar_ax = fig.add_axes([0.2, 0.22, 0.6, 0.02])
cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
cbar.set_label('For Error...\nRed: Obs<UFS & Blue: Obs>UFS', fontsize=20)

# Add a big title at the top
plt.suptitle('OLR Input Maps of 20% Most Confident Samples\nUFS {}'.format(Reg_dict[Reg][2]), fontsize=26)

# Save
plt.savefig('InMapAll_{}_Re{}Se{}Cls{}Pr{}Lt{}{}Tt{}N{}Lr{}'.format(Pred, Reg, seas, clss, Pr, lead_time1, lead_time2,
                                                                TTLoco, nodes, str(LR).split('.')[1]), dpi=300)
plt.show()
print('saved')

# %% Take a composite of all maps

# Composite
corr_conf_avg = corr_conf_dates['obs'].mean()

# Map
fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([-120, 120, -25, 25], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents
clevs = np.arange(-80, 85, 5)
cf = ax.contourf(lons, lats, corr_conf_avg, clevs, cmap=plt.cm.bwr, transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)  # aspect=50 flattens cbar
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)
plt.title('OLR Input Map Composite of\n20% Most Confident Samples\nUFS {}'.format(Reg_dict[Reg][2]), fontsize=26)
plt.savefig('InMapAvg_{}_Re{}Se{}Cls{}Pr{}Lt{}{}Tt{}N{}Lr{}'.format(Pred, Reg, seas, clss, Pr, lead_time1, lead_time2,
                                                                    TTLoco, nodes, str(LR).split('.')[1]), dpi=300)
plt.show()
print('saved2')

# %% Seasonal Out maps

# List of LT avg
LT_avg = []

# Select season
pred_szn = 'All'  # 'All', ' Winter', 'Fall', 'Summer', 'Spring'
corr_conf_szn_all = pd.merge(idx_all, corr_conf_dates, on='dates')
if pred_szn != 'All':
    corr_conf_szn = corr_conf_szn_all.loc[corr_conf_szn_all['Season'] == pred_szn]
else:
    corr_conf_szn = corr_conf_szn_all
    pred_szn = ''

# Trim the fat
corr_conf_pred = corr_conf_szn[['dates']]
corr_conf_pred = corr_conf_pred.reset_index(drop=True)
print(corr_conf_pred)
# define LTs
LTs = np.arange(1, lead_time2+2)

# Loop through lead times
for ipred in LTs:

    # Get lead time dates
    corr_conf_pred['LT{}'.format(ipred)] = corr_conf_pred['dates'] + pd.DateOffset(ipred)

    # Merge with prediction data
    ccpd = pd.merge(pred_obs, corr_conf_pred, left_on='dates', right_on='LT{}'.format(ipred))

    # Average corr_conf samples
    LT_avg += [ccpd['{} obs'.format(Pred)].mean()]

# Set-up dataframe
pred_df_all = pd.DataFrame({'LT': LTs, 'obs avg': LT_avg})

# Repeat process, but for UFS forecast

# List of LT avg
LT_avg_ufs = []

# Loop through lead times
for ipred2 in LTs:

    # Set-up value list for each corr & conf samples
    CC_val = []

    # Loop through corr & conf samples
    for ipred3 in range(len(corr_conf_pred['dates'])):

        # Grab specific LTs and then the sample that matches that lead time
        UFS_LT = ds_UFS1_base['{}'.format(Pred)][ipred2::LT_tot].sel(lat=Pred_sliceLat, lon=Pred_sliceLon,
                                                                     time=corr_conf_pred['dates'][
                                                                              ipred3] + pd.DateOffset(ipred2))

        # Grab values
        CC_val += [UFS_LT.values]

    # Average CC_vals and throw it into LT_avg_ufs
    LT_avg_ufs += [sum(CC_val) / len(CC_val)]

# Put in dataframe with obs
pred_df_all['ufs avg'] = LT_avg_ufs

# Calculate Error
pred_df_all['Error'] = pred_df_all['ufs avg'] - pred_df_all['obs avg']

# Output map

# Plot all out maps
proj = ccrs.PlateCarree(central_longitude=180)
fig, axs = plt.subplots(3, 3, subplot_kw={'projection': proj}, figsize=(11.5, 8.5))

# axs is a 2 dimensional array of `GeoAxes`.  We will flatten it into a 1-D array
axs = axs.flatten()

# Helps with map placement
qmap = [0, 0, 0, 5, 5, 5, 10, 10, 10]

for xpmap in range(9):

    # Set-up
    axs[xpmap].set_extent([-120, 120, -30, 65], ccrs.PlateCarree(central_longitude=180))
    clevs = np.arange(-180, 190, 10)

    if xpmap == 0 or xpmap == 3 or xpmap == 6:
        avg5 = pred_df_all['ufs avg'].iloc[qmap[xpmap]:qmap[xpmap] + 5].mean()

    elif xpmap == 1 or xpmap == 4 or xpmap == 7:
        avg5 = pred_df_all['obs avg'].iloc[qmap[xpmap]:qmap[xpmap] + 5].mean()

    else:
        avg5 = pred_df_all['Error'].iloc[qmap[xpmap]:qmap[xpmap] + 5].mean()

    # Plot
    cs = axs[xpmap].contourf(UFS_LT.lon.values+180, UFS_LT.lat.values, avg5, clevs,
                             transform=proj, cmap=plt.cm.bwr, extend='both')

    # Column titles
    ctits = ['UFS Forecast', 'Observations', 'Error (UFS - Obs)']
    if xpmap < 3:
        axs[xpmap].set_title('{}'.format(ctits[xpmap]), fontsize=16)

    '''# LT labels
    if xpmap == 0 or xpmap == 3 or xpmap == 6:
        axs[xpmap].set_ylabel('LT\n{}-{}'.format(qmap[xpmap] + 1, qmap[xpmap] + 5), rotation=0, fontsize=15)'''

    # Add extra features
    axs[xpmap].add_feature(cfeature.COASTLINE.with_scale('50m'))
    axs[xpmap].add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)

# Add a colorbar axis at the bottom of the graph
fig.subplots_adjust(bottom=0.25, top=0.875, left=0.1, right=0.95, wspace=0.15, hspace=0.02)
cbar_ax = fig.add_axes([0.2, 0.22, 0.6, 0.02])
cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
cbar.set_label('For Error...\nRed: UFS>Obs & Blue: UFS<Obs', fontsize=20)

# Add a big title at the top
plt.suptitle('OLR Output Maps of 20% Most Confident Samples\nUFS {} - {}'.format(Reg_dict[Reg][2], pred_szn),
             fontsize=26)

# Save
plt.savefig('OutMap_{}_Re{}Se{}Cls{}Pr{}Lt{}{}Tt{}N{}Lr{}{}'.format(Pred, Reg, seas, clss, Pr, lead_time1,
                                                                    lead_time2, TTLoco, nodes, str(LR).split('.')[1],
                                                                    pred_szn), dpi=300)
plt.show()
print('saved3')

# %% Look at individual out samples in time (w/ & w/0 u/v winds)
corr_conf_idv = corr_conf_datesM[['dates', 'obs', 'Index']]

# Get obs
pred_obs_map = pd.merge(pred_obs, corr_conf_idv, on='dates')

for omap1 in range(len(pred_obs_map['Index'])):

    # idx_counter accounts for validation data occurring after training & that the UFS forecasts are every 35 days
    idx_counter = (bignum * LT_tot) + (int(pred_obs_map['Index'][omap1]) * LT_tot)
    idx_counter_obs = (bignum + int(pred_obs_map['Index'][omap1])) * 7 + 4

    # For each sample, get their UFS leads out to 15
    outufs = ds_UFS1_base['{}'.format(Pred)].sel(lat=Pred_sliceLat, lon=Pred_sliceLon)[idx_counter:idx_counter+14]
    outobs = ds_obs1_base['{}'.format(Pred)].sel(lat=Pred_sliceLat, lon=Pred_sliceLon)[
             idx_counter_obs:idx_counter_obs + 14]
    outu = dfu['u_3'].sel(lat=Pred_sliceLat, lon=Pred_sliceLon)[idx_counter_obs:idx_counter_obs + 14]
    outv = dfv['v_3'].sel(lat=Pred_sliceLat, lon=Pred_sliceLon)[idx_counter_obs:idx_counter_obs + 14]

    # set-up lats/lons
    outlon = outufs.lon.values + 180
    outlat = outufs.lat.values
    lonmesh, latmesh = np.meshgrid(outlon, outlat)

    # Set-up map
    proj = ccrs.PlateCarree(central_longitude=180)
    fig, axs = plt.subplots(14, 3, subplot_kw={'projection': proj}, figsize=(11.5, 15.5))
    axs = axs.flatten()
    clevs = np.arange(-180, 190, 10)

    # Grab values for each date
    for omap2 in range(len(outufs.time.values)*3):

        # Set extents
        axs[omap2].set_extent([-120, 120, -30, 65], ccrs.PlateCarree(central_longitude=180))

        # Plot ufs map
        if (omap2 + 3) % 3 == 0:
            cs = axs[omap2].contourf(outlon, outlat, outufs[int(omap2 / 3)].values, clevs,
                                     transform=proj, cmap=plt.cm.bwr, extend='both')

        # Plot obs map (w/ winds)
        elif (omap2 + 3) % 3 == 1:
            cs = axs[omap2].contourf(outlon, outlat, outobs[int(omap2 / 3)].values, clevs, transform=proj,
                                     cmap=plt.cm.bwr, extend='both')
            axs[omap2].quiver(lonmesh[::7, ::7], latmesh[::7, ::7], outu[int(omap2 / 3)].values[::7, ::7],
                              outv[int(omap2 / 3)].values[::7, ::7])

        # Plot error
        else:
            err = outufs[int(omap2 / 3)].values - outobs[int(omap2 / 3)].values
            cs = axs[omap2].contourf(outlon, outlat, err,clevs, transform=proj, cmap=plt.cm.bwr, extend='both')

        # Column titles
        ctits = ['UFS Forecast', 'Observations', 'Error (UFS - Obs)']
        if omap2 < 3:
            axs[omap2].set_title('{}'.format(ctits[omap2]), fontsize=16)

        # Add extra features
        axs[omap2].add_feature(cfeature.COASTLINE.with_scale('50m'))
        axs[omap2].add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)

    # Add a colorbar axis at the bottom of the graph
    fig.subplots_adjust(bottom=0.25, top=0.9, left=0.1, right=0.95, wspace=0.15, hspace=0.15)
    cbar_ax = fig.add_axes([0.2, 0.22, 0.6, 0.02])
    cbar = fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('For Error...\nRed: UFS>Obs & Blue: UFS<Obs', fontsize=20)

    # Add a big title at the top
    plt.suptitle('OLR Output Map Sample {}\nUFS {}'.format(int(pred_obs_map['Index'][omap1]),
                                                           Reg_dict[Reg][2]), fontsize=26)

    # Save
    plt.savefig('OutMap{}_{}_Re{}Cls{}Pr{}Lt{}{}Tt{}N{}Lr{}'.format(int(pred_obs_map['Index'][omap1]), Pred, Reg,
                                                                    clss, Pr, lead_time1, lead_time2, TTLoco, nodes,
                                                                    str(LR).split('.')[1],
                                                                    pred_szn), dpi=300)

    plt.show()
    print(omap1)
print('saved 4')

# %% # # # # # # # # #  EXTRA STUFF # # # # # # # # # #

# In case we wanted to run hmaps for indices that occurred x+ times for each grid point
'''for xyz in range(counter):

    # flatten each grid point
    flat1 = np.concatenate(loc_arr[xyz]).ravel()

    # Count each values occurences in array
    idx1, cun1 = np.unique(flat1, return_counts=True)
    count1 = np.column_stack((idx1, cun1))

    # Convert index vs count into panda arrays
    count1pd = pd.DataFrame({'Index': idx1, 'count1': cun1})

    # Only look at indices that occurred 3 or more times in all grids
    over3p_1 = count1pd.drop(count1pd[(count1pd['count1'] < 2.5)].index)

    # Merge all the grids together
    if xyz == 0:
        countmerg1 = over3p_1  # if doing 3+ switch count1pd with over3p_1 - this line and one below it
    else:  # if not doing 3+ switch over3p_1 with count1pd
        countmerg1 = pd.merge(countmerg1, over3p_1, on='Index')

# Merge with the main index df and drop count columns
common_idx = pd.merge(countmerg1, idx_all, on='Index')

# Create heat map to show frequency of MJO and ENSO phases
hmap3p = np.zeros((3, 9))
for x in np.arange(-1, 2, 1):
    ENSOs = common_idx.loc[common_idx['E_Phase'] == x]
    for y in np.arange(0, 9, 1):
        MJO_ENSO = ENSOs.loc[ENSOs['M_Phase'] == y]
        hmap3p[x + 1, y] = len(MJO_ENSO['Index'])
hmap3p = hmap3p / hmap_main
print(hmap3p)

plt.imshow(hmap3p, cmap=plt.cm.Reds)
plt.xlabel('MJO Phase')
plt.ylabel('ENSO Phase')
plt.title('Frequency of Oscillation Phase', fontsize=16)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
plt.colorbar(shrink=0.75, label='Frequency')
plt.tight_layout()
plt.savefig('Heatmap3+_{}_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]), dpi=300)
plt.show()'''

# Look at Index distributions
'''# Use this for indexes and counts
# flatten each grid point
flat1 = np.concatenate(loc_arr[0]).ravel()
flat2 = np.concatenate(loc_arr[1]).ravel()
flat3 = np.concatenate(loc_arr[2]).ravel()
flat4 = np.concatenate(loc_arr[3]).ravel()

# Count each values occurences in array
idx1, cun1 = np.unique(flat1, return_counts=True)
idx2, cun2 = np.unique(flat2, return_counts=True)
idx3, cun3 = np.unique(flat3, return_counts=True)
idx4, cun4 = np.unique(flat4, return_counts=True)


# Comment out next 3 chunks
### compare plots
plt.bar(idx1, cun1)
plt.plot(idx1, cun1)
plt.show()
plt.bar(idx2, cun2)
plt.plot(idx2, cun2)
plt.show()
plt.bar(idx3, cun3)
plt.plot(idx3, cun3)
plt.show()
plt.bar(idx4, cun4)
plt.plot(idx4, cun4)
plt.show()

# smooth plots
smooth1 = savitzky_golay(cun1, 45, 3)  # window size 45, polynomial order 3
smooth2 = savitzky_golay(cun2, 45, 3)  # window size 45, polynomial order 3
smooth3 = savitzky_golay(cun3, 45, 3)  # window size 45, polynomial order 3
smooth4 = savitzky_golay(cun4, 45, 3)  # window size 45, polynomial order 3

# plot
plt.plot(idx1, smooth1)
plt.plot(idx2, smooth2)
plt.plot(idx3, smooth3)
plt.plot(idx4, smooth4)
plt.ylabel('Count', fontsize=14)
plt.xlabel('Index', fontsize=14)
plt.title('Predicted Indices for 4 Adjacent Grid Points', fontsize=16)
plt.show()'''

# Take composite of all maps
'''print(common_idx['obs'][x].shape)
for x in range(len(common_idx['obs'])):
    if x == 0:
        compos = common_idx['obs'][x]
        compos = np.reshape(compos, compos.shape + (1,))
    else:
        compos = np.dstack((compos, common_idx['obs'][x]))

comp_mean = compos.mean(axis=2)

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([-120, 80, -55, 70], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents
clevs = np.arange(-50, 51, 1)
# Contour for temps
# look up cmap colors for more color options
cf = ax.contourf(lons, lats, comp_mean, clevs, cmap=plt.cm.bwr, transform=ccrs.PlateCarree(central_longitude=180))
# Add lat and lon grid lines
ax.grid(linestyle='dotted', linewidth=2)  #
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)
plt.savefig('del')
plt.show()'''
