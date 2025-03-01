import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import numpy as np

data = [
    {"source": "Farm A", "lat1": 37.7749, "lon1": -122.4194, "destination": "Warehouse 1", "lat2": 40.7128, "lon2": -74.0060, "contamination": 0.2},
    {"source": "Farm B", "lat1": 34.0522, "lon1": -118.2437, "destination": "Warehouse 1", "lat2": 40.7128, "lon2": -74.0060, "contamination": 0.6},
    {"source": "Warehouse 1", "lat1": 40.7128, "lon1": -74.0060, "destination": "Retailer X", "lat2": 42.3601, "lon2": -71.0589, "contamination": 0.8},
]

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": ccrs.PlateCarree()})

ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=":")
ax.add_feature(cfeature.STATES, linestyle=":")

for _, row in df.iterrows():
    lon1, lat1, lon2, lat2 = row["lon1"], row["lat1"], row["lon2"], row["lat2"]

    lons = np.linspace(lon1, lon2, 100)
    lats = np.linspace(lat1, lat2, 100) + np.sin(np.linspace(0, np.pi, 100)) * 2

    contamination_level = row["contamination"]
    color = plt.cm.Reds(contamination_level)

    ax.plot(lons, lats, color=color, linewidth=2, transform=ccrs.PlateCarree())

ax.set_title("Food Supply Flow with Contamination Levels")
plt.show()
