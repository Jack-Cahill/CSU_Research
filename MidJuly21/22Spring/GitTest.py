# This File is solely practice for Git and Github within PyCharm
# This files read in data and outputs a map of the Pacific and Indian Oceans

# import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Specify variable
var = 'olr'

# Read in data
df = xr.open_dataset('/Users/jcahill4/DATA/{}/Obs/clima_gefs/clima_{}.nc'.format(var, var))
df = df['{}'.format(var)]

# define lats and lons
lats = df.lat.values  # lat is latitude name given by data
lons = df.lon.values
lons = lons + 180  # !!!!! only necessary if graph is centered on 180

# Select the first day (0)
Day1_vals = df[0]

# MAKE MAP
fig = plt.figure(figsize=(10, 8))

# Define map type and its size
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_extent([-140, 110, -25, 75], ccrs.PlateCarree(central_longitude=180))  # lat, lon extents

# Contours levels
if var == 'prcp':
    clevs = np.arange(-35, 35.5, .5)  # prcp clima
else:
    clevs = np.arange(-350, 355, 5)  # olr clima

# Colorbar
cf = ax.contourf(lons, lats, Day1_vals, clevs, cmap=plt.cm.bwr, transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)  # aspect=50 flattens cbar

# Set title
ax.set_title('{} Plot - Git Practice'.format(var), fontsize=16)

# Set x and y labels
ax.set_yticks([0, 25, 50])  # at a specific latitude
ax.set_yticklabels(['0\N{DEGREE SIGN}', '25\N{DEGREE SIGN}N', '25\N{DEGREE SIGN}N'])
ax.set_xticks([-120, -60, 0, 60])  # at a specific longitude
ax.set_xticklabels(['60\N{DEGREE SIGN}E', '120\N{DEGREE SIGN}E', '0\N{DEGREE SIGN}', '120\N{DEGREE SIGN}W'])

# Add extra features
ax.grid(linestyle='dotted', linewidth=2)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)

# Output
plt.tight_layout()
plt.savefig('GitMapPractice_{}'.format(var), dpi=300)
plt.show()
