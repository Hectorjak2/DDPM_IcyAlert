"""Location of the CARRA_CROPPED region on a map, outlined from the grid's own latitude/longitude."""
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import xarray as xr
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter

from plotting.config import savefig, TEXTWIDTH_IN, COLORS, MUTED, INK, GRID
from utils.dataloader import CARRA2_DATA_ROOT, CROPPED_X, CROPPED_Y
from utils.prepare_west_zarr import WEST_X, WEST_Y

# west.zarr dropped latitude/longitude, so read them from the full-domain store.
# CROPPED is given in WEST-local indices; shift it to full-domain indices.
Y = slice(WEST_Y.start + CROPPED_Y.start, WEST_Y.start + CROPPED_Y.stop)
X = slice(WEST_X.start + CROPPED_X.start, WEST_X.start + CROPPED_X.stop)

# CARRA2 grid: polar stereographic on a sphere, central meridian 30W, true scale at 90N, 2.5 km
# (fitted to the stored latitude/longitude; matches them to < 0.01%).
CARRA_CRS = ccrs.NorthPolarStereo(
    central_longitude=-30, true_scale_latitude=90,
    globe=ccrs.Globe(semimajor_axis=6371000, semiminor_axis=6371000),
)
MARGIN_KM = 600


ds = xr.open_zarr(f"{CARRA2_DATA_ROOT}/dataset.zarr")
lat = ds.latitude.isel(y=Y, x=X).values
lon = ds.longitude.isel(y=Y, x=X).values

# Closed outline through the edge pixel centres: bottom, right, top, left
edge_lat = np.concatenate([lat[0, :], lat[:, -1], lat[-1, ::-1], lat[::-1, 0]])
edge_lon = np.concatenate([lon[0, :], lon[:, -1], lon[-1, ::-1], lon[::-1, 0]])

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection=CARRA_CRS)

xy = CARRA_CRS.transform_points(ccrs.PlateCarree(), edge_lon, edge_lat)
m = MARGIN_KM * 1e3
ax.set_extent([xy[:, 0].min() - m, xy[:, 0].max() + m, xy[:, 1].min() - m, xy[:, 1].max() + m], crs=CARRA_CRS)

ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=GRID, edgecolor="none")
ax.add_feature(cfeature.COASTLINE.with_scale("50m"), edgecolor=MUTED, linewidth=0.4)
ax.fill(xy[:, 0], xy[:, 1], facecolor=COLORS[0], alpha=0.12, edgecolor="none", transform=CARRA_CRS)
ax.plot(xy[:, 0], xy[:, 1], color=COLORS[0], linewidth=1.4, transform=CARRA_CRS)

gl = ax.gridlines(
    crs=ccrs.PlateCarree(), draw_labels=True, color=MUTED, linewidth=0.4, linestyle=":",
    xlocs=mticker.FixedLocator(range(-180, 180, 20)), ylocs=mticker.FixedLocator(range(50, 90, 10)),
    x_inline=False, y_inline=False, rotate_labels=False,
)
gl.top_labels = gl.right_labels = False
gl.xformatter = LongitudeFormatter(degree_symbol=r"$^\circ$")
gl.yformatter = LatitudeFormatter(degree_symbol=r"$^\circ$")
gl.xlabel_style = gl.ylabel_style = {"color": INK}
ax.spines["geo"].set_edgecolor(MUTED)

savefig(fig, "figures", "cropped_region_map", width=TEXTWIDTH_IN * 0.6, height=TEXTWIDTH_IN * 0.6)
