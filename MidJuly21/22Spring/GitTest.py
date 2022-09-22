# This File is solely practice for Git and Github within PyCharm
# This files read in data and outputs a map of the Pacific and Indian Oceans

# import necessary packages
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Read in data
df = xr.open_dataset('/Users/jcahill4/DATA/prcp/Obs/clima_gefs/clima_prcp.nc')
df = df['prcp']

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
clevs = np.arange(-35, 35.5, .5)  # prcp clima

# Colorbar
cf = ax.contourf(lons, lats, Day1_vals, clevs, cmap=plt.cm.bwr, transform=ccrs.PlateCarree(central_longitude=180))
cbar = plt.colorbar(cf, orientation='horizontal', pad=0.04, aspect=50, extendrect=True)  # aspect=50 flattens cbar

# Add extra features
ax.grid(linestyle='dotted', linewidth=2)
ax.add_feature(cfeature.COASTLINE.with_scale('50m'))
ax.add_feature(cfeature.LAKES.with_scale('50m'), color='black', linewidths=0.5)

# Output
plt.tight_layout()
plt.savefig('GitMapPractice', dpi=300)
plt.show()
