from py_eddy_tracker.observations.tracking import TrackEddiesObservations
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker.generic import distance_grid

from netCDF4 import num2date, date2num
from datetime import datetime
import numpy as np
from xarray import open_dataset
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

# Graphics %matplotlib inline
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'serif',
        'weight' : 'bold',
        'size'   : 14}
matplotlib.rc('font', **font)

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
data_crs=ccrs.PlateCarree()

def plt_cartopy(ax,labels=True):
    ax.set_aspect('auto') # The figure will follow the ratio of the figsize
    ax.coastlines()       # Plot the coastlines
    ax.set_xticks([-80,-70,-60,-50,-40,-30], crs=data_crs) # Set ticks for longitude
    ax.set_yticks([30,40,50,60], crs=data_crs)             # Set ticks for latitude
    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    
def reverse_index(index, nb):
    m = np.ones(nb, dtype="bool")
    m[index] = False
    return np.where(m)[0]

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

# Anticyclonic and Cyclonic 4days tracking

filename_AVISO_AC = '/ec/res4/scratch/ita6648/miost_science/eddy_output_north_atlantic/eddy_tracking/Anticyclonic_cmems_4days.nc'
filename_AVISO_CC= '/ec/res4/scratch/ita6648/miost_science/eddy_output_north_atlantic/eddy_tracking/Cyclonic_cmems_4days.nc'
#/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking
filename_RIOPS_v2_AC = '/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking/Anticyclonic_cmems_4days.nc'
filename_RIOPS_v2_CC = '/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking/Cyclonic_cmems_4days.nc'
#/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking
filename_RIOPS_v4_AC = '/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/eddy_tracking/Anticyclonic_cmems_4days.nc'
filename_RIOPS_v4_CC = '/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/eddy_tracking/Cyclonic_cmems_4days.nc'
AVISO_AC = TrackEddiesObservations.load_file(filename_AVISO_AC)
AVISO_CC = TrackEddiesObservations.load_file(filename_AVISO_CC)
RIOPS_v2_AC = TrackEddiesObservations.load_file(filename_RIOPS_v2_AC)
RIOPS_v2_CC = TrackEddiesObservations.load_file(filename_RIOPS_v2_CC)
RIOPS_v4_AC = TrackEddiesObservations.load_file(filename_RIOPS_v4_AC)
RIOPS_v4_CC = TrackEddiesObservations.load_file(filename_RIOPS_v4_CC)

# ==========================================================
# Choose the exact day to validate
# ==========================================================
# Change this date whenever you want, format: YYYY-MM-DD
TARGET_DATE_STR = "2025-01-01"

TIME_UNITS = "days since 1950-01-01"
TIME_CALENDAR = "standard"

target_datetime = datetime.strptime(TARGET_DATE_STR, "%Y-%m-%d")
target_day = int(date2num(target_datetime, TIME_UNITS, calendar=TIME_CALENDAR))

print("Selected date:", TARGET_DATE_STR)
print("Numeric time:", target_day)
print("Date check:", num2date(target_day, TIME_UNITS, TIME_CALENDAR))


def check_date_available(dataset, dataset_name, target_day):
    times = np.unique(dataset.obs["time"])
    if target_day not in times:
        tmin = int(times.min())
        tmax = int(times.max())
        print("\nERROR: date not available in", dataset_name)
        print("Requested:", TARGET_DATE_STR, "numeric time =", target_day)
        print("Available range:",
              str(num2date(tmin, TIME_UNITS, TIME_CALENDAR))[:10], "->",
              str(num2date(tmax, TIME_UNITS, TIME_CALENDAR))[:10])
        raise ValueError("Selected date is not available in " + dataset_name)


check_date_available(AVISO_AC, "AVISO_AC", target_day)
check_date_available(AVISO_CC, "AVISO_CC", target_day)
check_date_available(RIOPS_v2_AC, "GLORYS2V4_AC", target_day)
check_date_available(RIOPS_v2_CC, "GLORYS2V4_CC", target_day)
check_date_available(RIOPS_v4_AC, "GLORYS12V1_AC", target_day)
check_date_available(RIOPS_v4_CC, "GLORYS12V1_CC", target_day)

# Tracked eddies for the selected day
## Anticyclonic
aviso_ac_track = AVISO_AC.extract_with_period((target_day, target_day))
riops_ac_track = RIOPS_v2_AC.extract_with_period((target_day, target_day))
mercator_ac_track = RIOPS_v4_AC.extract_with_period((target_day, target_day))

## Cyclonic, currently loaded but not plotted in this script
aviso_cc_track = AVISO_CC.extract_with_period((target_day, target_day))
riops_cc_track = RIOPS_v2_CC.extract_with_period((target_day, target_day))
mercator_cc_track = RIOPS_v4_CC.extract_with_period((target_day, target_day))

## List
AC_track_list, CC_track_list = [aviso_ac_track, riops_ac_track, mercator_ac_track], [aviso_cc_track, riops_cc_track, mercator_cc_track]

def plot_map(ax,eddy_aviso,eddy_model,figname):

    if('riops' in namestr(eddy_model,globals())[0]):
        model = 'GLORYS2V4'
    else:
        model = 'GLORYS12V1'

    aviso_id, model_id, cost = eddy_aviso.tracking(eddy_model)
    aviso_id_junk, model_id_junk = reverse_index(aviso_id, len(eddy_aviso)), reverse_index(model_id, len(eddy_model))

    plt.title(figname)
    plt_cartopy(ax)
    ax.set_extent([-80,-30,30,60])

    ## Plot Hits
    ### AVISO
    y = eddy_aviso.obs['contour_lat_e'][aviso_id]
    x = eddy_aviso.obs['contour_lon_e'][aviso_id]
    for i in range(len(aviso_id)):
        if i == 0:
            plt.plot(x[i],y[i],color='k',lw=2,linestyle=':',label=('Hits (SWOT MIOST): '+ str(len(aviso_id))),transform=proj)
        else:
            plt.plot(x[i],y[i],color='k',lw=2,linestyle=':',transform=proj)
    ### RIOPS_v2
    y = eddy_model.obs['contour_lat_e'][model_id]
    x = eddy_model.obs['contour_lon_e'][model_id]

    for i in range(len(model_id)):
        if i == 0:
            plt.plot(x[i],y[i],color='k',lw=2,linestyle='-',label=('Hits ('+model+'): '+ str(len(model_id))),transform=proj)
        else:
            plt.plot(x[i],y[i],color='k',lw=2,linestyle='-',transform=proj)
    ## Plot Misses
    ### AVISO
    y = eddy_aviso.obs['contour_lat_e'][aviso_id_junk]
    x = eddy_aviso.obs['contour_lon_e'][aviso_id_junk]
    for i in range(len(aviso_id_junk)):
        if i == 0:
            plt.plot(x[i],y[i],color='r',lw=2,linestyle=':',label=('Misses (SWOT MIOST): '+ str(len(aviso_id_junk))),transform=proj)
        else:
            plt.plot(x[i],y[i],color='r',lw=2,linestyle=':',transform=proj)
    ## Plot False Alarm
    ### RIOPS_v2
    y = eddy_model.obs['contour_lat_e'][model_id_junk]
    x = eddy_model.obs['contour_lon_e'][model_id_junk]
    for i in range(len(model_id_junk)):
        if i == 0:
            plt.plot(x[i],y[i],color='r',lw=2,linestyle='-',label=('False Alarm ('+model+'): '+ str(len(model_id_junk))),transform=proj)
        else:
            plt.plot(x[i],y[i],color='r',lw=2,linestyle='-',transform=proj)
        
    plt.legend(loc = 'upper left', prop={'size': 11})

  
plt.subplots(figsize=(16,6))
plt.suptitle('')
proj = ccrs.PlateCarree()
ax = plt.subplot(121,projection=proj)
plot_map(ax,aviso_ac_track,riops_ac_track,'')
ax = plt.subplot(122,projection=proj)
plot_map(ax,aviso_ac_track,mercator_ac_track,'')

# Save or show
plt.savefig('Validation_of_Anticyclonic_Eddies_' + TARGET_DATE_STR.replace('-', '') + '_swot.png', dpi=300, bbox_inches='tight')




