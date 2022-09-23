"""
Created on Thu Jul 28 09:38:16 2022

@author: jcahill4
"""

import geopandas as gp
import tensorflow as tf
import sys
import numpy as np
import xarray as xr
import glob
import pandas as pd
import time
import matplotlib.pyplot as plt
import cmasher as cmr
from sklearn import preprocessing
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
import matplotlib.colors as colors
from Functions import defineNN, PredictionAccuracy, CategoricalTruePositives

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


# If you want maps to include correct and incorrect top confidences uncomment CONLY

# %% -------------------------------- INPUTS --------------------------------

# Define a set of random seeds (or just one seed if you want to check)
r = [92, 95, 100, 137, 141, 142]
# r = np.arange(88, 89, 1)

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

# Define
Pred = 'h500'
if Pred == 'prcp':
    Predictor = 0
elif Pred == 'h500':
    Predictor = 1

# Helps sort data
cdata = 'clima_gefs'
bignum = 862  # selects testing / training data
smlnum = 180  # " "
LT_tot = 35  # how many total lead times are there?

# Whole map or just a subset?
adj4 = 2  # 0 is no, 1 is for SW, 2, is for NW

# Create CONUS grid
if adj4 == 1:
    xlat_slice = slice(33.9, 37.6)
    xlon_slice = slice(255.4, 260.6)

elif adj4 == 2:
    xlat_slice = slice(45.9, 49.6)
    xlon_slice = slice(237.4, 242.6)

else:
    xlat_slice = slice(23.9, 49.6)
    xlon_slice = slice(235.4, 293.6)

dfe = xr.open_dataset('/Users/jcahill4/DATA/h500/Obs/clima_gefs/clima_h500.nc')
dfe = dfe['h500'].sel(lat=xlat_slice, lon=xlon_slice)

CONUS_lats = dfe.lat.values[::4]
CONUS_lons = dfe.lon.values[::4]
CONUS_lons = CONUS_lons[:len(CONUS_lons) - 1]
print(CONUS_lats)
print(CONUS_lons)

# Decrease input resolution?
dec = 0  # 0 if normal, 1 if decreases

# # # # # # # # # #  NEURAL NETWORK INPUTS # # # # # # # # # #

error = 0  # Are we prediting UFS errors? (0 if error, 1 if no error)
lead_time1 = 10  # will be averaged from LT1-LT2
lead_time2 = 14
epochs = 10_000
nodes = 20
batch_size = 32  # Not identified in plot naming scheme
LR = 0.01  # Learning Rate
Classes = 3  # 3 classes (-1: Under_est, 0: Acc_est, 1: Over_est)
PATIENCE = 20
MinMax = 0  # 0 if minimizing loss and 1 if maximizing accuracy
TTLoco = 0  # Order of Training and Validation datasets (0 if TrainVal, 1 if ValTrain)

# # # # # # # # # #  READ IN DATA # # # # # # # # # #

# OBS
# Read in ALL obs using xarray and paramters
ds_obs = xr.open_mfdataset('/Users/jcahill4/DATA/{}/Obs/{}/clima_{}.nc'.format(
variable, cdata, variable), concat_dim='time', combine='nested')
ds_obs = ds_obs[variable].sel(lon=lon_slice, lat=lat_slice)

# Specify Variables for plots
lats = ds_obs.lat.values  # lat is latitude name given by data
lons = ds_obs.lon.values
lons = lons + 180
times = ds_obs.time.values

# Turn obs into a pandas array
ds_obs = list(zip(ds_obs.time.values, ds_obs.values))
ds_obs = pd.DataFrame(ds_obs)
ds_obs.columns = ['time', '{} obs'.format(variable)]

# UFS
# Read in each h500/OLR file (in a loop) so that each day 1 forecast gets made into its own array
TimeBuild = []
VarBuild = []  # start with empty lists
for filepath in glob.iglob('/Users/jcahill4/DATA/{}/UFS/{}/*.nc'.format(variable, cdata)):
    ds_UFS = xr.open_dataset(filepath)  # read in each UFS file
    ds_UFS = ds_UFS[variable].sel(lon=lon_slice, lat=lat_slice)
    ds_UFS = ds_UFS[0]
    TimeBuild += [ds_UFS.time.values]
    VarBuild += [ds_UFS.values]

# Turn h500/OLR into a pandas array
ds_UFS = list(zip(TimeBuild, VarBuild))
ds_UFS = pd.DataFrame(ds_UFS)
ds_UFS.columns = ['time', '{} UFS'.format(variable)]  # Name columns
ds_UFS = ds_UFS.sort_values(by='time')

# Merge data into a pandas array
ds = pd.merge(ds_UFS, ds_obs)
ds = ds.sort_values(by='time')
timer = ds['time']

# Create a pandas arrays that has index, dates, and obs maps
val_idx = np.arange(0, smlnum)
obs_maps = ds['olr obs'][bignum:]
date_list = ds['time'][bignum:]
idx_date = pd.DataFrame({'idx': val_idx, 'dates': date_list, 'obs': obs_maps})

# Merge MJO and ENSO info
mdata = pd.read_csv('/Users/jcahill4/Downloads/mjo_phase1.csv')
mdata_ymd = mdata.iloc[:, :3]  # Just grab YMD data
mdata_dt1 = pd.to_datetime(mdata_ymd[['year', 'month', 'day']])  # convert YMD columns to one dt column

# Add Date and Phase to YMD data
mdata_ymd['dates'] = mdata_dt1
mdata_ymd['M_Phase'] = mdata['Phase']
mdata_ymd['E_Phase'] = mdata['ENSO']
mdata_ymd['Amp'] = mdata['Amplitude']
mdata_ymd['Amp'] = pd.to_numeric(mdata_ymd['Amp'], downcast="float")

# RMM adjustment (if Amplitude (sqrt(RMM1^2 + RMM2^2)) < 1, then convert phase to 0 (phase 0 is just no MJO Phase)
i = 0
while i < len(mdata_ymd['Amp']):
    if mdata_ymd['Amp'][i] < 1.0:
        mdata_ymd['M_Phase'][i] = 0
        i = i + 1
    else:
        i = i + 1

# Drop Y-M-D
mdata_ymd = mdata_ymd.drop(['year', 'month', 'day'], axis=1)

# Merge with time data
idx_all = pd.merge(idx_date, mdata_ymd, on='dates')

# Pred data base
ds_UFS1_base = xr.open_mfdataset('/Users/jcahill4/DATA/{}/UFS/{}/*.nc'.format(Pred, cdata), concat_dim='time',
                                 combine='nested')
ds_obs1_base = xr.open_dataset('/Users/jcahill4/DATA/{}/Obs/{}/clima_{}.nc'.format(Pred, cdata, Pred))
ds_UFS_base = xr.open_mfdataset('/Users/jcahill4/DATA/{}/UFS/{}/*.nc'.format(Pred, cdata), concat_dim='time',
                                combine='nested')

# SET UP MAP
xmap_full = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_full[:] = np.nan
xmap_und = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_und[:] = np.nan
xmap_acc = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_acc[:] = np.nan
xmap_ovr = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_ovr[:] = np.nan
xmap_und_a = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_und_a[:] = np.nan
xmap_acc_a = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_acc_a[:] = np.nan
xmap_fall_full = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_fall_full[:] = np.nan
xmap_ovr_a = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_ovr_a[:] = np.nan
xmap_sum_full = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_sum_full[:] = np.nan
xmap_wint_full = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_wint_full[:] = np.nan
xmap_spr_full = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_spr_full[:] = np.nan
xmap_sum_und = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_sum_und[:] = np.nan
xmap_fall_und = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_fall_und[:] = np.nan
xmap_wint_und = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_wint_und[:] = np.nan
xmap_spr_und = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_spr_und[:] = np.nan
xmap_sum_acc = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_sum_acc[:] = np.nan
xmap_fall_acc = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_fall_acc[:] = np.nan
xmap_wint_acc = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_wint_acc[:] = np.nan
xmap_spr_acc = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_spr_acc[:] = np.nan
xmap_sum_ovr = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_sum_ovr[:] = np.nan
xmap_fall_ovr = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_fall_ovr[:] = np.nan
xmap_wint_ovr = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_wint_ovr[:] = np.nan
xmap_spr_ovr = np.empty((len(CONUS_lons), len(CONUS_lats)))
xmap_spr_ovr[:] = np.nan

# location array
if adj4 != 0:
    loc_arr = []
    loc_arrCN = []
    loc_arrCLS = []

    # set-up a counter so we can determine how many grid points we're running
    counter = 0

### SHAPEFILES

# Read in shapefiles & set coords
states = gp.read_file('/Users/jcahill4/DATA/Udders/usa-states-census-2014.shp')
states.crs = 'EPSG:4326'

NW = states[states['STUSPS'].isin(['WA', 'OR', 'ID'])]
us_boundary_map = states.boundary.plot(color="white", linewidth=1)

### Convert lat and lons into grid pts

# repeat list of lats multiple times
ptlats = []
for i in range(len(CONUS_lons)):
    for element in CONUS_lats:
        ptlats += [element]
ptlats = np.array(ptlats)

# repeat first lon multiple times, then second lon, etc
ptlons = []
for i in range(len(CONUS_lons)):
    for j in range(len(CONUS_lats)):
        ptlons += [CONUS_lons[i]]
ptlons = np.array(ptlons)

# convert lons into lons that match polygons
ptlons = ptlons - 360

# put lat and lons into array as gridpoints        
df1 = pd.DataFrame({'longitude': ptlons, 'latitude': ptlats})
gdf = gp.GeoDataFrame(df1, geometry=gp.points_from_xy(df1.longitude, df1.latitude))
US_pts = gp.sjoin(gdf, states, op='within')
US_pts.plot(ax=us_boundary_map, color='pink')
plt.show()

# LOOP THROUGH MAP
# After merging lets put this in a pd df
dfC = pd.DataFrame({'longitude': US_pts.longitude, 'latitude': US_pts.latitude})
for c1, xxx in enumerate(CONUS_lons):
    for c2, xx in enumerate(CONUS_lats):

        df_oop = dfC.loc[dfC.latitude == xx]
        if (xxx - 360) in df_oop.longitude.unique():
            counter = counter + 1
            print(xx, xxx)

            # arrays that restart for every location
            acc_full = []
            acc_und = []
            acc_acc = []
            acc_ovr = []
            acc_und_a = []
            acc_acc_a = []
            acc_ovr_a = []
            acc_sum_full = []
            acc_fall_full = []
            acc_wint_full = []
            acc_spr_full = []
            acc_sum_und = []
            acc_fall_und = []
            acc_wint_und = []
            acc_spr_und = []
            acc_sum_acc = []
            acc_fall_acc = []
            acc_wint_acc = []
            acc_spr_acc = []
            acc_sum_ovr = []
            acc_fall_ovr = []
            acc_wint_ovr = []
            acc_spr_ovr = []

            # arrays for 4 adjacent grid points
            if adj4 != 0:
                locos = []
                locosCN = []
                locosCLS = []

            for x in r:

                # Select map point
                lat_sliceP = slice(xx - .01, xx + .01)
                lon_sliceP = slice(xxx - .01, xxx + .01)

                # Seed
                NP_SEED = x
                np.random.seed(NP_SEED)
                tf.random.set_seed(NP_SEED)

                # Pred

                # Read in data (UFS and Obs)
                ds_UFS1 = ds_UFS1_base['{}'.format(Pred)].sel(lat=lat_sliceP, lon=lon_sliceP)
                ds_obs1 = ds_obs1_base['{}'.format(Pred)].sel(lat=lat_sliceP, lon=lon_sliceP)

                # Predictor (for error 1)
                # Read in data
                ds_UFS = ds_UFS_base['{}'.format(Pred)].sel(lat=lat_sliceP, lon=lon_sliceP)

                ### DATA BEING READ IN AND MERGED CORRECTLY HAS BEEN TRIPLE CHECKED

                if error == 0:

                    # PART1: WE START BY ONLY GRABBING A FEW UFS DATES AND ALL OBS. 
                    # THEN THE UFS AND OBS ARE MERGED SO THEY MATCH
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
                    ds_obsp = list(zip(ds_obs1.time.values, ds_obs1.values.flatten()))
                    ds_obsp = pd.DataFrame(ds_obsp)
                    ds_obsp.columns = ['time', '{} Obs'.format(Pred)]

                    # Remove time (non-date) aspect
                    ds_UFS10['time'] = pd.to_datetime(ds_UFS10['time']).dt.date
                    ds_obsp['time'] = pd.to_datetime(ds_obsp['time']).dt.date

                    # Merge UFS and obs
                    ds10x = pd.merge(ds_obsp, ds_UFS10, on='time')
                    ### DATA BEING READ IN AND MERGED CORRECTLY HAS BEEN TRIPLE CHECKED

                    # PART 2: AVERAGE EACH FILE BETWEEN LEAD_TIMES 10-14, SO WE HAVE TWO VECTORS (OBS AND UFS)
                    # Mean over LT10-14
                    divisor = int(lead_time2 - lead_time1 + 1)
                    ds10x = ds10x.sort_values(by='time')
                    ds10x = ds10x.drop(['time'], axis=1)
                    ds10 = ds10x.groupby(np.arange(len(ds10x)) // divisor).mean()

                    ### DATA BEING AVERAGED HAS BEEN TRIPLE CHECKED

                    # PART 3: CREATE TWO MORE COLUMNS/VECTORS FOR ERRORS AND CLASSES

                    # Find ranges for classes (this uses 3 classes)
                    ds10['Error'] = ds10['{} UFS'.format(Pred)] - ds10['{} Obs'.format(Pred)]

                    if Classes == 3:
                        # Find ranges for classes (this uses 3 classes)
                        q1 = ds10['Error'].quantile(0.33)
                        q2 = ds10['Error'].quantile(0.67)

                        # Create classes
                        ds10.loc[ds10['Error'].between(float('-inf'), q1), 'Class'] = 0
                        ds10.loc[ds10['Error'].between(q1, q2), 'Class'] = 1
                        ds10.loc[ds10['Error'].between(q2, float('inf')), 'Class'] = 2
                        ds_UFSp = ds10

                    elif Classes == 5:
                        # Find ranges for classes (this uses 3 classes)
                        q1 = ds10['Error'].quantile(0.2)
                        q2 = ds10['Error'].quantile(0.4)
                        q3 = ds10['Error'].quantile(0.6)
                        q4 = ds10['Error'].quantile(0.8)

                        # Create classes
                        ds10.loc[ds10['Error'].between(float('-inf'), q1), 'Class'] = 0
                        ds10.loc[ds10['Error'].between(q1, q2), 'Class'] = 1
                        ds10.loc[ds10['Error'].between(q2, q3), 'Class'] = 2
                        ds10.loc[ds10['Error'].between(q3, q4), 'Class'] = 3
                        ds10.loc[ds10['Error'].between(q4, float('inf')), 'Class'] = 4
                        ds_UFSp = ds10

                elif error == 1:

                    # PART1: WE START BY ONLY GRABBING A FEW UFS DATES AND ALL OBS. 
                    # THEN THE UFS AND OBS ARE MERGED SO THEY MATCH
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

                    ds10x = ds_UFS10

                    # PART 2: AVERAGE EACH FILE BETWEEN LEAD_TIMES 10-14, SO WE HAVE TWO VECTORS (OBS AND UFS)

                    # Mean over LT
                    divisor = int(lead_time2 - lead_time1 + 1)
                    ds10x = ds10x.sort_values(by='time')
                    ds10x = ds10x.drop(['time'], axis=1)
                    ds10 = ds10x.groupby(np.arange(len(ds10x)) // divisor).mean()

                    # PART 3: CREATE TWO MORE COLUMNS/VECTORS FOR ERRORS AND CLASSES

                    if Classes == 3:
                        # Find ranges for classes (this uses 3 classes)
                        q1 = ds10['{} UFS'.format(Pred)].quantile(0.33)
                        q2 = ds10['{} UFS'.format(Pred)].quantile(0.67)

                        # Create classes
                        ds10.loc[ds10['{} UFS'.format(Pred)].between(float('-inf'), q1), 'Class'] = 0
                        ds10.loc[ds10['{} UFS'.format(Pred)].between(q1, q2), 'Class'] = 1
                        ds10.loc[ds10['{} UFS'.format(Pred)].between(q2, float('inf')), 'Class'] = 2
                        ds_UFSp = ds10

                    elif Classes == 5:
                        # Find ranges for classes (this uses 3 classes)
                        q1 = ds10['{} UFS'.format(Pred)].quantile(0.2)
                        q2 = ds10['{} UFS'.format(Pred)].quantile(0.4)
                        q3 = ds10['{} UFS'.format(Pred)].quantile(0.6)
                        q4 = ds10['{} UFS'.format(Pred)].quantile(0.8)

                        # Create classes
                        ds10.loc[ds10['{} UFS'.format(Pred)].between(float('-inf'), q1), 'Class'] = 0
                        ds10.loc[ds10['{} UFS'.format(Pred)].between(q1, q2), 'Class'] = 1
                        ds10.loc[ds10['{} UFS'.format(Pred)].between(q2, q3), 'Class'] = 2
                        ds10.loc[ds10['{} UFS'.format(Pred)].between(q3, q4), 'Class'] = 3
                        ds10.loc[ds10['{} UFS'.format(Pred)].between(q4, float('inf')), 'Class'] = 4
                        ds_UFSp = ds10

                # DATA RETRIEVAL

                # Run a loop so all the h500/OLR obs maps are available 
                # (this is before the code is split into Training and Validation)
                x = 0
                ALLx = []
                while x < len(ds['{} obs'.format(variable)]):
                    ALLx += [ds['{} obs'.format(variable)][x]]
                    x = x + 1
                ALLx = np.array(ALLx)

                # Get only Classes from pandas Predictor array
                y = 0
                ALLy = []
                while y < len(ds_UFSp['Class']):
                    ALLy += [ds_UFSp['Class'][y]]
                    y = y + 1
                ALLy = np.array(ALLy)
                ALLy = ALLy.astype(int)

                ###### ALLx and ALLy  HAS BEEN TRIPLE CHECKED AND ITS CORRECT

                # SPLITING UP DATA

                if TTLoco == 0:
                    x_train = ALLx[:bignum]
                    y_train = ALLy[:bignum]
                    x_val = ALLx[bignum:]
                    y_val = ALLy[bignum:]
                else:
                    x_val = ALLx[:smlnum]
                    y_val = ALLy[:smlnum]
                    x_train = ALLx[smlnum:]
                    y_train = ALLy[smlnum:]

                # x_train are the input maps w/ shape (bignum, 50, 241) : (# cases, lats, lons)
                # y_train are the predicted error classes of the area of interest shape (bignum)

                # Change x_train and x_val shape if we wanted to decrease input map resolution
                if dec == 1:
                    x_train = x_train[:, ::4, ::4]
                    x_val = x_val[:, ::4, ::4]
                    x_train_shp = x_train.reshape(bignum, len(lats[::4]) * len(lons[::4]))
                    x_val_shp = x_val.reshape(smlnum, len(lats[::4]) * len(lons[::4]))
                else:
                    x_train_shp = x_train.reshape(bignum, len(lats) * len(lons))
                    x_val_shp = x_val.reshape(smlnum, len(lats) * len(lons))

                # -------------------------------- SPECIAL PARAMETERS --------------------------------

                RIDGE1 = 0.0
                DROPOUT = 0.3


                # -------------------------------- NETWORK SETUP --------------------------------

                # MAKE THE NN ARCHITECTURE
                def make_model():
                    # Define and train the model
                    tf.keras.backend.clear_session()

                    # NORMAL ANN
                    model = defineNN([nodes],
                                     input1_shape=np.shape(x_train_shp)[1],
                                     output_shape=3,
                                     ridge_penalty1=RIDGE1,
                                     dropout=DROPOUT,
                                     act_fun='relu',
                                     network_seed=NP_SEED)

                    # tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
                    # tf.keras.losses.CategoricalCrossentropy()
                    loss_function = tf.keras.losses.CategoricalCrossentropy()

                    model.compile(
                        # optimizer = tf.keras.optimizers.SGD(
                        # learning_rate=LR, momentum=0.9, nesterov=True),
                        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
                        loss=loss_function,
                        metrics=[
                            # tf.keras.metrics.SparseCategoricalAccuracy(
                            # name="sparse_categorical_accuracy", dtype=None),
                            tf.keras.metrics.CategoricalAccuracy(
                                name="categorical_accuracy", dtype=None),
                            PredictionAccuracy(y_train.shape[0]),
                            CategoricalTruePositives()
                        ]
                    )

                    # model.summary()
                    return model, loss_function


                # -------------------------------- MAKE NN --------------------------------

                if MinMax == 0:
                    es_callback = tf.keras.callbacks.EarlyStopping(  # monitor='val_prediction_accuracy',
                        monitor='val_loss',
                        patience=PATIENCE,
                        # mode='max',
                        mode='min',
                        restore_best_weights=True,
                        verbose=0)
                else:
                    es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_prediction_accuracy',
                                                                   # monitor='val_loss',
                                                                   patience=PATIENCE,
                                                                   mode='max',
                                                                   # mode='min',
                                                                   restore_best_weights=True,
                                                                   verbose=0)
                callbacks = [es_callback]

                model, loss_function = make_model()

                # y_train one-hot labels
                enc = preprocessing.OneHotEncoder()
                onehotlabels = enc.fit_transform(np.array(y_train).reshape(-1, 1)).toarray()
                hotlabels_train = onehotlabels[:, :model.output_shape[-1]]

                # y_val one-hot labels
                onehotlabels = enc.fit_transform(np.array(y_val).reshape(-1, 1)).toarray()
                hotlabels_val = onehotlabels[:, :model.output_shape[-1]]

                # -------------------------------- TRAINING NETWORK --------------------------------

                start_time = time.time()
                history = model.fit(x_train_shp, hotlabels_train,
                                    validation_data=(x_val_shp, hotlabels_val),
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    shuffle=True,  # shuffle data before each epoch
                                    verbose=0,
                                    callbacks=callbacks)
                stop_time = time.time()

                # -------------------------------- PREDICTION & OUTPUT INFO --------------------------------

                # Grab loss and acc values from training

                out = history.history
                out_list = list(out.items())

                loss = out_list[0]
                loss = loss[1]  # training loss

                acc = out_list[5]
                acc = np.array(acc[1]) * 100  # acc of validation

                ep = np.arange(1, len(acc) + 1, 1)  # epoch array

                # Output from prediction

                preds = model.predict(x_val_shp)  # Confidences for all classes
                max_idxs = np.argmax(preds, axis=1)  # index of winning confidence

                # First make an array of all the times when our array is correct [1] or not [0]
                Correct_R = []
                k = 0
                while k < len(max_idxs):
                    if max_idxs[k] == y_val[k]:
                        Correct_R += [1]
                    else:
                        Correct_R += [0]
                    k = k + 1
                Correct_R = np.array(Correct_R)

                # Make an array of all the confidences (largest one for each row)
                Conf = preds  # Confidences for all classes
                WConf = max_idxs  # index of winning confidence

                Conf = np.amax(Conf, 1)  # array of just the winning confidences
                idx = np.arange(0, len(Conf), 1)  # array of idx of map (0 is first map in validation)
                hit_miss1 = np.stack((Correct_R, Conf, y_val, WConf, idx), axis=-1)
                hit_miss = hit_miss1[np.argsort(hit_miss1[:, 1])]  # sort from least to most confident

                Bmaps = hit_miss[:, 4]  # idx's of maps (least to most confident)
                WinClass = hit_miss[:, 3]  # Predicted Classes (least to most confident)
                ActClass = hit_miss[:, 2]  # Actual Classes (least to most confident)
                CorrOrNo = hit_miss[:, 0]  # array for if pred correct [1] or not [0] (least to most confident)

                # tf.print(f"Elapsed time during fit = {stop_time - start_time:.2f} seconds\n")

                # -------------------------------- PLOTS --------------------------------    

                # Then build the line plot 

                CP_Corr_R = []
                w = 0
                while w < len(hit_miss[:, 0]):
                    CP_Corr_R += [(np.sum(hit_miss[:, 0][w:]) / len(hit_miss[:, 0][w:])) * 100]
                    w = w + 1

                CPs = np.linspace(100, 0, num=13)
                CPs = CPs[:-1]
                CP_Corr_R = np.array(CP_Corr_R)
                CP_Corr_R = np.mean(CP_Corr_R.reshape(-1, 9), axis=1)

                ###### Cluster maps

                # x-Vals aren't shuffled so dates will match

                # Add data to array only if the run is "good" (goes over 40% acc, never under 20% acc for 50% most confident runs)
                # if np.min(CP_Corr_R[-6:]) > 20.0 and np.max(CP_Corr_R) > 40.0:
                if np.min(CP_Corr_R[-6:]) > 0.0 and np.max(CP_Corr_R) > 0.0:
                    Bmaps20 = Bmaps[int(len(Bmaps) * 0.80):]  # top 20%
                    WinClass20 = WinClass[int(len(WinClass) * 0.8):]  # .8 because we're grabbing the top 20% data
                    CorrOrNo20 = CorrOrNo[int(len(CorrOrNo) * 0.8):]
                    ActClass20 = ActClass[int(len(ActClass) * 0.8):]

                    # indices for now
                    locos += [Bmaps20]
                    locosCN += [CorrOrNo20]
                    locosCLS += [WinClass20]

                    # Seasonal indices
                    sIndex1 = np.arange(0, 8)
                    sIndex2 = np.arange(47, 60)
                    sIndex3 = np.arange(99, 112)
                    sIndex4 = np.arange(151, 164)
                    sIndex = np.concatenate((sIndex1, sIndex2, sIndex3, sIndex4), axis=None)
                    sum_pd = pd.DataFrame({'Index': sIndex})

                    fIndex1 = np.arange(8, 21)
                    fIndex2 = np.arange(60, 73)
                    fIndex3 = np.arange(122, 125)
                    fIndex4 = np.arange(164, 774)
                    fIndex = np.concatenate((fIndex1, fIndex2, fIndex3, fIndex4), axis=None)
                    fall_pd = pd.DataFrame({'Index': fIndex})

                    wIndex1 = np.arange(21, 33)
                    wIndex2 = np.arange(73, 86)
                    wIndex3 = np.arange(125, 138)
                    wIndex4 = np.arange(177, 180)
                    wIndex = np.concatenate((wIndex1, wIndex2, wIndex3, wIndex4), axis=None)
                    wint_pd = pd.DataFrame({'Index': wIndex})

                    spIndex1 = np.arange(33, 47)
                    spIndex2 = np.arange(86, 99)
                    spIndex3 = np.arange(138, 151)
                    spIndex = np.concatenate((spIndex1, spIndex2, spIndex3), axis=None)
                    spr_pd = pd.DataFrame({'Index': spIndex})

                    # ACC map based on class and predicted class
                    CorrCount_pd = pd.DataFrame({'Corr?': CorrOrNo20, 'WinClass': WinClass20, 'Index': Bmaps20})

                    # ACC MAP: get accuracy of top 20% most confident samples and put them in a list
                    CorrCount = np.count_nonzero(CorrOrNo20 == 1)
                    acc_full += [CorrCount / len(CorrOrNo20)]

                    CorrCount0_pd = CorrCount_pd[CorrCount_pd["WinClass"] == 0]
                    CorrCount0_list = CorrCount0_pd.iloc[:, 0]
                    CorrCount0 = np.count_nonzero(CorrCount0_list == 1)
                    if len(CorrCount0_list) == 0:
                        acc_und += [np.nan]
                    else:
                        acc_und += [CorrCount0 / len(CorrCount0_list)]

                    CorrCount1_pd = CorrCount_pd[CorrCount_pd["WinClass"] == 1]
                    CorrCount1_list = CorrCount1_pd.iloc[:, 0]
                    CorrCount1 = np.count_nonzero(CorrCount1_list == 1)
                    if len(CorrCount1_list) == 0:
                        acc_acc += [np.nan]
                    else:
                        acc_acc += [CorrCount1 / len(CorrCount1_list)]

                    CorrCount2_pd = CorrCount_pd[CorrCount_pd["WinClass"] == 2]
                    CorrCount2_list = CorrCount2_pd.iloc[:, 0]
                    CorrCount2 = np.count_nonzero(CorrCount2_list == 1)
                    if len(CorrCount2_list) == 0:
                        acc_ovr += [np.nan]
                    else:
                        acc_ovr += [CorrCount2 / len(CorrCount2_list)]

                    # Seasonal groups
                    s_pd = pd.merge(sum_pd, CorrCount_pd, on='Index')
                    f_pd = pd.merge(fall_pd, CorrCount_pd, on='Index')
                    w_pd = pd.merge(wint_pd, CorrCount_pd, on='Index')
                    sp_pd = pd.merge(spr_pd, CorrCount_pd, on='Index')

                    s_u_pd = pd.merge(sum_pd, CorrCount0_pd, on='Index')
                    f_u_pd = pd.merge(fall_pd, CorrCount0_pd, on='Index')
                    w_u_pd = pd.merge(wint_pd, CorrCount0_pd, on='Index')
                    sp_u_pd = pd.merge(spr_pd, CorrCount0_pd, on='Index')

                    s_a_pd = pd.merge(sum_pd, CorrCount1_pd, on='Index')
                    f_a_pd = pd.merge(fall_pd, CorrCount1_pd, on='Index')
                    w_a_pd = pd.merge(wint_pd, CorrCount1_pd, on='Index')
                    sp_a_pd = pd.merge(spr_pd, CorrCount1_pd, on='Index')

                    s_o_pd = pd.merge(sum_pd, CorrCount2_pd, on='Index')
                    f_o_pd = pd.merge(fall_pd, CorrCount2_pd, on='Index')
                    w_o_pd = pd.merge(wint_pd, CorrCount2_pd, on='Index')
                    sp_o_pd = pd.merge(spr_pd, CorrCount2_pd, on='Index')

                    # summer
                    if 1 in s_pd['Corr?'].values:
                        CorrCountSum = s_pd['Corr?'].value_counts()[1]
                        if len(s_pd['Index']) == 0:
                            acc_sum_full += [np.nan]
                        else:
                            acc_sum_full += [CorrCountSum / len(s_pd['Index'])]
                    else:
                        acc_sum_full += [np.nan]

                    if 1 in s_u_pd['Corr?'].values:
                        CorrCountSumu = s_u_pd['Corr?'].value_counts()[1]
                        if len(s_u_pd['Index']) == 0:
                            acc_sum_und += [np.nan]
                        else:
                            acc_sum_und += [CorrCountSumu / len(s_u_pd['Index'])]
                    else:
                        acc_sum_und += [np.nan]

                    if 1 in s_a_pd['Corr?'].values:
                        CorrCountSuma = s_a_pd['Corr?'].value_counts()[1]
                        if len(s_a_pd['Index']) == 0:
                            acc_sum_acc += [np.nan]
                        else:
                            acc_sum_acc += [CorrCountSuma / len(s_a_pd['Index'])]
                    else:
                        acc_sum_acc += [np.nan]

                    if 1 in s_o_pd['Corr?'].values:
                        CorrCountSumo = s_o_pd['Corr?'].value_counts()[1]
                        if len(s_o_pd['Index']) == 0:
                            acc_sum_ovr += [np.nan]
                        else:
                            acc_sum_ovr += [CorrCountSumo / len(s_o_pd['Index'])]
                    else:
                        acc_sum_ovr += [np.nan]

                    # fall
                    if 1 in f_pd['Corr?'].values:
                        CorrCountFall = f_pd['Corr?'].value_counts()[1]
                        if len(f_pd['Index']) == 0:
                            acc_fall_full += [np.nan]
                        else:
                            acc_fall_full += [CorrCountFall / len(f_pd['Index'])]
                    else:
                        acc_fall_full += [np.nan]

                    if 1 in f_u_pd['Corr?'].values:
                        CorrCountFallu = f_u_pd['Corr?'].value_counts()[1]
                        if len(f_u_pd['Index']) == 0:
                            acc_fall_und += [np.nan]
                        else:
                            acc_fall_und += [CorrCountFallu / len(f_u_pd['Index'])]
                    else:
                        acc_fall_und += [np.nan]

                    if 1 in f_a_pd['Corr?'].values:
                        CorrCountFalla = f_a_pd['Corr?'].value_counts()[1]
                        if len(f_a_pd['Index']) == 0:
                            acc_fall_acc += [np.nan]
                        else:
                            acc_fall_acc += [CorrCountFalla / len(f_a_pd['Index'])]
                    else:
                        acc_fall_acc += [np.nan]

                    if 1 in f_o_pd['Corr?'].values:
                        CorrCountFallo = f_o_pd['Corr?'].value_counts()[1]
                        if len(f_o_pd['Index']) == 0:
                            acc_fall_ovr += [np.nan]
                        else:
                            acc_fall_ovr += [CorrCountFallo / len(f_o_pd['Index'])]
                    else:
                        acc_fall_ovr += [np.nan]

                    # winter
                    if 1 in w_pd['Corr?'].values:
                        CorrCountWint = w_pd['Corr?'].value_counts()[1]
                        if len(w_pd['Index']) == 0:
                            acc_wint_full += [np.nan]
                        else:
                            acc_wint_full += [CorrCountWint / len(w_pd['Index'])]
                    else:
                        acc_wint_full += [np.nan]

                    if 1 in w_u_pd['Corr?'].values:
                        CorrCountWintu = w_u_pd['Corr?'].value_counts()[1]
                        if len(w_u_pd['Index']) == 0:
                            acc_wint_und += [np.nan]
                        else:
                            acc_wint_und += [CorrCountWintu / len(w_u_pd['Index'])]
                    else:
                        acc_wint_und += [np.nan]

                    if 1 in w_a_pd['Corr?'].values:
                        CorrCountWinta = w_a_pd['Corr?'].value_counts()[1]
                        if len(w_a_pd['Index']) == 0:
                            acc_wint_acc += [np.nan]
                        else:
                            acc_wint_acc += [CorrCountWinta / len(w_a_pd['Index'])]
                    else:
                        acc_wint_acc += [np.nan]

                    if 1 in w_o_pd['Corr?'].values:
                        CorrCountWinto = w_o_pd['Corr?'].value_counts()[1]
                        if len(w_o_pd['Index']) == 0:
                            acc_wint_ovr += [np.nan]
                        else:
                            acc_wint_ovr += [CorrCountWinto / len(w_o_pd['Index'])]
                    else:
                        acc_wint_ovr += [np.nan]

                    # spring
                    if 1 in sp_pd['Corr?'].values:
                        CorrCountSpr = sp_pd['Corr?'].value_counts()[1]
                        if len(sp_pd['Index']) == 0:
                            acc_spr_full += [np.nan]
                        else:
                            acc_spr_full += [CorrCountSpr / len(sp_pd['Index'])]
                    else:
                        acc_spr_full += [np.nan]

                    if 1 in sp_u_pd['Corr?'].values:
                        CorrCountSpru = sp_u_pd['Corr?'].value_counts()[1]
                        if len(sp_u_pd['Index']) == 0:
                            acc_spr_und += [np.nan]
                        else:
                            acc_spr_und += [CorrCountSpru / len(sp_u_pd['Index'])]
                    else:
                        acc_spr_und += [np.nan]

                    if 1 in sp_a_pd['Corr?'].values:
                        CorrCountSpra = sp_a_pd['Corr?'].value_counts()[1]
                        if len(sp_a_pd['Index']) == 0:
                            acc_spr_acc += [np.nan]
                        else:
                            acc_spr_acc += [CorrCountSpra / len(sp_a_pd['Index'])]
                    else:
                        acc_spr_acc += [np.nan]

                    if 1 in sp_o_pd['Corr?'].values:
                        CorrCountSpro = sp_o_pd['Corr?'].value_counts()[1]
                        if len(sp_o_pd['Index']) == 0:
                            acc_spr_ovr += [np.nan]
                        else:
                            acc_spr_ovr += [CorrCountSpro / len(sp_o_pd['Index'])]
                    else:
                        acc_spr_ovr += [np.nan]

                    # Repeat process, but instead of Predicted Class (WinClass), grab True Class 
                    CorrCount_pd_a = pd.DataFrame({'Corr?': CorrOrNo20, 'ActClass': ActClass20})

                    CorrCount0_pd_a = CorrCount_pd_a[CorrCount_pd_a["ActClass"] == 0]
                    CorrCount0_list_a = CorrCount0_pd_a.iloc[:, 0]
                    CorrCount0_a = np.count_nonzero(CorrCount0_list_a == 1)
                    if len(CorrCount0_list_a) == 0:
                        acc_und_a += [np.nan]
                    else:
                        acc_und_a += [CorrCount0_a / len(CorrCount0_list_a)]

                    CorrCount1_pd_a = CorrCount_pd_a[CorrCount_pd_a["ActClass"] == 1]
                    CorrCount1_list_a = CorrCount1_pd_a.iloc[:, 0]
                    CorrCount1_a = np.count_nonzero(CorrCount1_list_a == 1)
                    if len(CorrCount1_list_a) == 0:
                        acc_acc_a += [np.nan]
                    else:
                        acc_acc_a += [CorrCount1_a / len(CorrCount1_list_a)]

                    CorrCount2_pd_a = CorrCount_pd_a[CorrCount_pd_a["ActClass"] == 2]
                    CorrCount2_list_a = CorrCount2_pd_a.iloc[:, 0]
                    CorrCount2_a = np.count_nonzero(CorrCount2_list_a == 1)
                    if len(CorrCount2_list_a) == 0:
                        acc_ovr_a += [np.nan]
                    else:
                        acc_ovr_a += [CorrCount2_a / len(CorrCount2_list_a)]


                else:
                    Trash = Trash + 1

            xmap_full[c1][c2] = np.nanmean(acc_full)
            xmap_acc[c1][c2] = np.nanmean(acc_acc)
            xmap_und[c1][c2] = np.nanmean(acc_und)
            xmap_ovr[c1][c2] = np.nanmean(acc_ovr)
            xmap_acc_a[c1][c2] = np.nanmean(acc_acc_a)
            xmap_und_a[c1][c2] = np.nanmean(acc_und_a)
            xmap_ovr_a[c1][c2] = np.nanmean(acc_ovr_a)
            xmap_sum_full[c1][c2] = np.nanmean(acc_sum_full)
            xmap_fall_full[c1][c2] = np.nanmean(acc_fall_full)
            xmap_wint_full[c1][c2] = np.nanmean(acc_wint_full)
            xmap_spr_full[c1][c2] = np.nanmean(acc_spr_full)
            xmap_sum_und[c1][c2] = np.nanmean(acc_sum_und)
            xmap_fall_und[c1][c2] = np.nanmean(acc_fall_und)
            xmap_wint_und[c1][c2] = np.nanmean(acc_wint_und)
            xmap_spr_und[c1][c2] = np.nanmean(acc_spr_und)
            xmap_sum_acc[c1][c2] = np.nanmean(acc_sum_acc)
            xmap_fall_acc[c1][c2] = np.nanmean(acc_fall_acc)
            xmap_wint_acc[c1][c2] = np.nanmean(acc_wint_acc)
            xmap_spr_acc[c1][c2] = np.nanmean(acc_spr_acc)
            xmap_sum_ovr[c1][c2] = np.nanmean(acc_sum_ovr)
            xmap_fall_ovr[c1][c2] = np.nanmean(acc_fall_ovr)
            xmap_wint_ovr[c1][c2] = np.nanmean(acc_wint_ovr)
            xmap_spr_ovr[c1][c2] = np.nanmean(acc_spr_ovr)

            if adj4 != 0:
                loc_arr += [locos]
                loc_arrCN += [locosCN]
                loc_arrCLS += [locosCLS]
print('DONE')
# %%
'''print(np.amax(xmap_full))
print(np.amin(xmap_full))
print(np.average(xmap_full))'''
print(xmap_full)


# %%
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(int(window_size))
        order = np.abs(int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


# %%
# Speicify which class we're interested in
class_num = 0

# Make a "total" hmap that can we can divide by to normalize hamps
hmap_main = np.zeros((3, 9))
for x in np.arange(-1, 2, 1):
    ENSOs = idx_all.loc[idx_all['E_Phase'] == x]
    for y in np.arange(0, 9, 1):
        MJO_ENSO = ENSOs.loc[ENSOs['M_Phase'] == y]
        hmap_main[x + 1, y] = len(MJO_ENSO['idx'])
print(hmap_main)

for xy in range(counter):

    flat_idx = np.concatenate(loc_arr[xy]).ravel()
    flat_CNO = np.concatenate(loc_arrCN[xy]).ravel()
    flat_CLS = np.concatenate(loc_arrCLS[xy]).ravel()

    if xy == 0:
        flat_idx_all = flat_idx
        flat_CNO_all = flat_CNO
        flat_CLS_all = flat_CLS
    else:
        flat_idx_all = np.concatenate([flat_idx_all, flat_idx])
        flat_CNO_all = np.concatenate([flat_CNO_all, flat_CNO])
        flat_CLS_all = np.concatenate([flat_CLS_all, flat_CLS])

# Separate based on class and correct samples
info_all_all = pd.DataFrame({'idx': flat_idx_all, 'CorrOrNo': flat_CNO_all, 'Class': flat_CLS_all})
info_all_corr = info_all_all[info_all_all['CorrOrNo'] == 1].reset_index(drop=True)
info_class_all = info_all_all[info_all_all['Class'] == class_num].reset_index(drop=True)
info_class_corr = info_class_all[info_class_all['CorrOrNo'] == 1].reset_index(drop=True)

# Count indices
info_all_all_count = info_all_all.groupby(['idx'])['idx'].count()
info_all_corr_count = info_all_corr.groupby(['idx'])['idx'].count()
info_class_all_count = info_class_all.groupby(['idx'])['idx'].count()
info_class_corr_count = info_class_corr.groupby(['idx'])['idx'].count()

info_all_all_count_df = info_all_all_count.to_frame()
info_all_corr_count_df = info_all_corr_count.to_frame()
info_class_all_count_df = info_class_all_count.to_frame()
info_class_corr_count_df = info_class_corr_count.to_frame()

info_all_all_idx = info_all_all_count_df.index  # Indices
info_all_corr_idx = info_all_corr_count_df.index
info_class_all_idx = info_class_all_count_df.index
info_class_corr_idx = info_class_corr_count_df.index

info_all_all_count = info_all_all_count_df['idx'].reset_index(drop=True)  # Count
info_all_corr_count = info_all_corr_count_df['idx'].reset_index(drop=True)
info_class_all_count = info_class_all_count_df['idx'].reset_index(drop=True)
info_class_corr_count = info_class_corr_count_df['idx'].reset_index(drop=True)

# Convert index and counts into pandas array
countpd_all_all = pd.DataFrame({'idx': info_all_all_idx, 'count': info_all_all_count})
countpd_all_corr = pd.DataFrame({'idx': info_all_corr_idx, 'count': info_all_corr_count})
countpd_class_all = pd.DataFrame({'idx': info_class_all_idx, 'count': info_class_all_count})
countpd_class_corr = pd.DataFrame({'idx': info_class_corr_idx, 'count': info_class_corr_count})

# merge with ENSO and MJO info
EMJO_all_all = pd.merge(countpd_all_all, idx_all, on='idx')
EMJO_all_corr = pd.merge(countpd_all_corr, idx_all, on='idx')
EMJO_class_all = pd.merge(countpd_class_all, idx_all, on='idx')
EMJO_class_corr = pd.merge(countpd_class_corr, idx_all, on='idx')

# Make relevant hmaps
# Create heat map to show frequency of MJO and ENSO phases
hmap = np.zeros((3, 9))
for x in np.arange(-1, 2, 1):
    ENSOs = EMJO_class_all.loc[EMJO_class_all['E_Phase'] == x]
    for y in np.arange(0, 9, 1):
        MJO_ENSO = ENSOs.loc[ENSOs['M_Phase'] == y]
        hmap[x + 1, y] = len(MJO_ENSO['idx'])
hmap = hmap / hmap_main
print(hmap)

plt.imshow(hmap, cmap=plt.cm.Reds)
plt.xlabel('MJO Phase')
plt.ylabel('ENSO Phase')
plt.title('Frequency of Oscillation Phase', fontsize=16)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
plt.colorbar(shrink=0.75, label='Frequency')
plt.tight_layout()
plt.savefig(
    'HeatmapClass{}_{}_Lt{}{}N{}Lr{}'.format(class_num, Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
    dpi=300)
plt.show()

hmapCN = np.zeros((3, 9))
for x in np.arange(-1, 2, 1):
    ENSOs_CN = EMJO_class_corr.loc[EMJO_class_corr['E_Phase'] == x]
    for y in np.arange(0, 9, 1):
        MJO_ENSO_CN = ENSOs_CN.loc[ENSOs_CN['M_Phase'] == y]
        hmapCN[x + 1, y] = len(MJO_ENSO_CN['idx'])
hmapCN = hmapCN / hmap_main
print(hmapCN)

plt.imshow(hmapCN, cmap=plt.cm.Reds)
plt.xlabel('MJO Phase')
plt.ylabel('ENSO Phase')
plt.title('Frequency of Oscillation Phase\n(Correctly Predicted Samples)', fontsize=16)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
plt.colorbar(shrink=0.75, label='Frequency')
plt.tight_layout()
plt.savefig(
    'HeatmapClass{}Corr_{}_Lt{}{}N{}Lr{}'.format(class_num, Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
    dpi=300)
plt.show()

hmap_all_all = np.zeros((3, 9))
for x in np.arange(-1, 2, 1):
    ENSOs_CN = EMJO_all_all.loc[EMJO_all_all['E_Phase'] == x]
    for y in np.arange(0, 9, 1):
        MJO_ENSO_CN = ENSOs_CN.loc[ENSOs_CN['M_Phase'] == y]
        hmap_all_all[x + 1, y] = len(MJO_ENSO_CN['idx'])
hmap_all_all = hmap_all_all / hmap_main
print(hmap_all_all)

plt.imshow(hmap_all_all, cmap=plt.cm.Reds)
plt.xlabel('MJO Phase')
plt.ylabel('ENSO Phase')
plt.title('Frequency of Oscillation Phase', fontsize=16)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
plt.colorbar(shrink=0.75, label='Frequency')
plt.tight_layout()
plt.savefig('Heatmap_all_all_Corr_{}_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

hmap_all_corr = np.zeros((3, 9))
for x in np.arange(-1, 2, 1):
    ENSOs_CN = EMJO_all_corr.loc[EMJO_all_corr['E_Phase'] == x]
    for y in np.arange(0, 9, 1):
        MJO_ENSO_CN = ENSOs_CN.loc[ENSOs_CN['M_Phase'] == y]
        hmap_all_corr[x + 1, y] = len(MJO_ENSO_CN['idx'])
hmap_all_corr = hmap_all_corr / hmap_main
print(hmap_all_corr)

plt.imshow(hmap_all_corr, cmap=plt.cm.Reds)
plt.xlabel('MJO Phase')
plt.ylabel('ENSO Phase')
plt.title('Frequency of Oscillation Phase\n(Correctly Predicted Samples)', fontsize=16)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8], ['None', 1, 2, 3, 4, 5, 6, 7, 8])
plt.yticks([0, 1, 2], ['La Niña', 'Neutral', 'El Niño'])
plt.colorbar(shrink=0.75, label='Frequency')
plt.tight_layout()
plt.savefig('Heatmap_all_corr_Corr_{}_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

# %%
# Utilize counter
print(counter)
for xyz in range(counter):

    # flatten each grid point
    flat1 = np.concatenate(loc_arr[xyz]).ravel()

    # Count each values occurences in array
    idx1, cun1 = np.unique(flat1, return_counts=True)
    count1 = np.column_stack((idx1, cun1))

    # Convert index vs count into panda arrays
    count1pd = pd.DataFrame({'idx': idx1, 'count1': cun1})

    # Only look at indices that occurred 3 or more times in all grids
    over3p_1 = count1pd.drop(count1pd[(count1pd['count1'] < 2.5)].index)

    # Merge all the grids together
    if xyz == 0:
        countmerg1 = over3p_1  # if doing 3+ switch count1pd with over3p_1 - this line and one below it
    else:  # if not doing 3+ switch over3p_1 with count1pd
        countmerg1 = pd.merge(countmerg1, over3p_1, on='idx')

# Merge with the main index df and drop count columns
common_idx = pd.merge(countmerg1, idx_all, on='idx')

# Create heat map to show frequency of MJO and ENSO phases
hmap3p = np.zeros((3, 9))
for x in np.arange(-1, 2, 1):
    ENSOs = common_idx.loc[common_idx['E_Phase'] == x]
    for y in np.arange(0, 9, 1):
        MJO_ENSO = ENSOs.loc[ENSOs['M_Phase'] == y]
        hmap3p[x + 1, y] = len(MJO_ENSO['idx'])
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
plt.show()

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

# %%

ecolor = 'dimgray'
fsize = 24

# FULL

# Since pcolor are boxes, we have to set start and end pts of the boxes: lons_p and lats_p make it so the center of the box
# are the lons and lats
lons_p = np.append(CONUS_lons, CONUS_lons[-1] + 2) - 1
lats_p = np.append(CONUS_lats, CONUS_lats[-1] + 2) - 1

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .325
vmax = .5
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.4125, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_full.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS\nLead Time of 10-14 days (All Samples)', fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_all_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]), dpi=300)
plt.show()
# print(map_full)

# UNDER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .3
vmax = .7
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.5, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plo
cf = ax.pcolor(lons_p - 180, lats_p, xmap_und.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS\nLead Time of 10-14 days (UFS Underestimates)', fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_und_PredClass_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_und)

# ACCURATE

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_acc.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS\nLead Time of 10-14 days (UFS Accurate Estimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_acc_PredClass_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_acc)

# OVER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_ovr.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS\nLead Time of 10-14 days (UFS Overestimates)', fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_ovr_PredClass_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

# %% True Class

# UNDER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .3
vmax = .7
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.5, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plo
cf = ax.pcolor(lons_p - 180, lats_p, xmap_und_a.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS\nLead Time of 10-14 days (UFS Underestimates)', fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_und_TrueClass_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_und)

# ACCURATE

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_acc_a.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS\nLead Time of 10-14 days (UFS Accurate Estimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_acc_TrueClass_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_acc)

# OVER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_ovr_a.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS\nLead Time of 10-14 days (UFS Overestimates)', fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_ovr_TrueClass_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# %%

'''xmap_sum_full[np.isnan(xmap_sum_full)] = 0
xmap_fall_full[np.isnan(xmap_fall_full)] = 0
xmap_wint_full[np.isnan(xmap_wint_full)] = 0
xmap_spr_full[np.isnan(xmap_spr_full)] = 0
xmap_sum_ovr[np.isnan(xmap_sum_ovr)] = 0
xmap_fall_ovr[np.isnan(xmap_fall_ovr)] = 0
xmap_wint_ovr[np.isnan(xmap_wint_ovr)] = 0
xmap_spr_ovr[np.isnan(xmap_spr_ovr)] = 0
xmap_sum_und[np.isnan(xmap_sum_und)] = 0
xmap_fall_und[np.isnan(xmap_fall_und)] = 0
xmap_wint_und[np.isnan(xmap_wint_und)] = 0
xmap_spr_und[np.isnan(xmap_spr_und)] = 0
xmap_sum_acc[np.isnan(xmap_sum_acc)] = 0
xmap_fall_acc[np.isnan(xmap_fall_acc)] = 0
xmap_wint_acc[np.isnan(xmap_wint_acc)] = 0
xmap_spr_acc[np.isnan(xmap_spr_acc)] = 0'''

# %% Summer

# FULL

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .35
vmax = .6
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.475, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_sum_full.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Summer\nLead Time of 10-14 days (All Samples)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_all_sum_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_full)

# UNDER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .4
vmax = 1.0
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.7, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plo
cf = ax.pcolor(lons_p - 180, lats_p, xmap_sum_und.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Summer\nLead Time of 10-14 days (UFS Underestimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_und_sum_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_und)

# ACCURATE

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_sum_acc.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Summer\nLead Time of 10-14 days (UFS Accurate Estimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_acc_sum_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_acc)

# OVER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_sum_ovr.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Summer\nLead Time of 10-14 days (UFS Overestimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_ovr_sum_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

# %% Fall

# FULL

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .35
vmax = .6
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.475, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_fall_full.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Fall\nLead Time of 10-14 days (All Samples)', fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_all_fall_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_full)

# UNDER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .4
vmax = 1.0
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.7, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plo
cf = ax.pcolor(lons_p - 180, lats_p, xmap_fall_und.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Fall\nLead Time of 10-14 days (UFS Underestimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_und_fall_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_und)

# ACCURATE

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_fall_acc.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Fall\nLead Time of 10-14 days (UFS Accurate Estimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_acc_fall_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_acc)

# OVER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_fall_ovr.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Fall\nLead Time of 10-14 days (UFS Overestimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_ovr_fall_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

# %% Winter

# FULL

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .35
vmax = .6
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.475, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_wint_full.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Winter\nLead Time of 10-14 days (All Samples)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_all_wint_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_full)

# UNDER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .4
vmax = 1.0
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.7, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plo
cf = ax.pcolor(lons_p - 180, lats_p, xmap_wint_und.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Winter\nLead Time of 10-14 days (UFS Underestimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_und_wint_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

# ACCURATE

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_wint_acc.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Winter\nLead Time of 10-14 days (UFS Accurate Estimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_acc_wint_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

# OVER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_wint_ovr.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Winter\nLead Time of 10-14 days (UFS Overestimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_ovr_wint_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

# %% Spring

# FULL

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .35
vmax = .6
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.475, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_spr_full.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Spring\nLead Time of 10-14 days (All Samples)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_all_spr_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

# UNDER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Set levels
vmin = .4
vmax = 1.0
vmid = colors.TwoSlopeNorm(vmin=vmin, vcenter=.6, vmax=vmax)

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plo
cf = ax.pcolor(lons_p - 180, lats_p, xmap_spr_und.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Spring\nLead Time of 10-14 days (UFS Underestimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_und_spr_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
# print(map_und)

# ACCURATE

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_spr_acc.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Spring\nLead Time of 10-14 days (UFS Accurate Estimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_acc_spr_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()

# OVER

fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([53.5, 113.5, 22, 50], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Add lat and lon grid lines
# Add variety of features
# ax.add_feature(cfeature.LAND, color='white')
ax.add_feature(cfeature.COASTLINE)

# Can also supply matplotlib kwargs
ax.add_feature(cfeature.STATES.with_scale('10m'), edgecolor=ecolor)
ax.add_feature(cfeature.BORDERS, edgecolor=ecolor)
ax.add_feature(cfeature.LAKES, color=ecolor, alpha=0.5)

# plot
cf = ax.pcolor(lons_p - 180, lats_p, xmap_spr_ovr.T, vmin=vmin, vmax=vmax, norm=vmid,
               cmap=plt.cm.get_cmap('Reds'), transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)

plt.title('Accuracy of Predicting h500 Errors in the UFS - Spring\nLead Time of 10-14 days (UFS Overestimates)',
          fontsize=fsize)
plt.tight_layout()
plt.savefig('AccMap_{}_ovr_spr_Lt{}{}N{}Lr{}'.format(Pred, lead_time1, lead_time2, nodes, str(LR).split('.')[1]),
            dpi=300)
plt.show()
