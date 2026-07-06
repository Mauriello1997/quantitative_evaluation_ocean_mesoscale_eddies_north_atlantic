from py_eddy_tracker.observations.tracking import TrackEddiesObservations
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker.generic import distance_grid

from netCDF4 import num2date
from datetime import datetime
from collections import OrderedDict
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
#filename_AVISO_AC = f'/ec/res4/scratch/ita6648/AVISO_DUACS/eddy_output_north_atlantic/eddy_tracking/Anticyclonic.nc'
#filename_AVISO_CC = f'/ec/res4/scratch/ita6648/AVISO_DUACS/eddy_output_north_atlantic/eddy_tracking/Cyclonic.nc'
filename_AVISO_AC = f'/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/eddy_tracking/Anticyclonic_4days.nc'
filename_AVISO_CC= f'/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/eddy_tracking/Cyclonic_4days.nc'

filename_RIOPS_v2_AC = f'/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking/Anticyclonic_cmems_4days.nc'
filename_RIOPS_v2_CC = f'/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking/Cyclonic_cmems_4days.nc'
filename_RIOPS_v4_AC = f'/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/eddy_tracking/Anticyclonic_cmems_4days.nc'
filename_RIOPS_v4_CC = f'/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/eddy_tracking/Cyclonic_cmems_4days.nc'

AVISO_AC = TrackEddiesObservations.load_file(filename_AVISO_AC)
AVISO_CC = TrackEddiesObservations.load_file(filename_AVISO_CC)
GORP_1_12_AC = TrackEddiesObservations.load_file(filename_RIOPS_v4_AC)
GORP_1_12_CC = TrackEddiesObservations.load_file(filename_RIOPS_v4_CC)
GORP_1_4_AC = TrackEddiesObservations.load_file(filename_RIOPS_v2_AC)
GORP_1_4_CC = TrackEddiesObservations.load_file(filename_RIOPS_v2_CC)
dates = np.arange(GORP_1_12_AC.obs['time'].min(),GORP_1_12_AC.obs['time'].max()) # dates since 1950-01-01 (01/01/20 to 29/12/2021)

# Hits, Misses, False Alarm
## GORP 1/12°
AG_hits   = np.ma.masked_array(np.zeros((len(dates))),mask=True)
AG_misses = np.ma.masked_array(np.zeros((len(dates))),mask=True)
AG_falarm = np.ma.masked_array(np.zeros((len(dates))),mask=True)
## GORP 1/4°
AR_hits   = np.ma.masked_array(np.zeros((len(dates))),mask=True)
AR_misses = np.ma.masked_array(np.zeros((len(dates))),mask=True)
AR_falarm = np.ma.masked_array(np.zeros((len(dates))),mask=True)

# Costs (Hits)
## GORP 1/12°
AG_costs           = []
AG_cost_radius     = []
AG_cost_amplitudes = []
AG_cost_distances  = []
## GORP 1/4°
AR_costs           = []
AR_cost_radius     = []
AR_cost_amplitudes = []
AR_cost_distances  = []

# Radius & Amplitudes of Hits, Misses, & False Alarm
## GORP 1/12°
### Radius
radius_AG_hits       = []
radius_AG_falarm     = []
radius_AG_misses     = []
### Amplitude
amplitude_AG_hits    = []
amplitude_AG_falarm  = []
amplitude_AG_misses  = []
## GORP 1/4°
### Radius
radius_AR_hits       = []
radius_AR_falarm     = []
radius_AR_misses     = []
### Amplitude
amplitude_AR_hits    = []
amplitude_AR_falarm  = []
amplitude_AR_misses  = []

for date in dates:
    
    
    index = int(date - dates[0])
    # Extract by date
    ## Anticyclonic
    aviso_ac = AVISO_AC.extract_with_period((round(date),round(date))) # Le format de date dans AVISO est different de celui de RIOPS
    riops_v2_ac = GORP_1_12_AC.extract_with_period((date,date))
    riops_v4_ac = GORP_1_4_AC.extract_with_period((date,date))
    ## Cyclonic
    aviso_cc = AVISO_CC.extract_with_period((round(date),round(date))) # Le format de date dans AVISO est different de celui de RIOPS
    riops_v2_cc = GORP_1_12_CC.extract_with_period((date,date))
    riops_v4_cc = GORP_1_4_CC.extract_with_period((date,date))
    
    # Fill arrays
    ## GORP 1/12°
    ### Anticyclonic
    aviso_id, riops_v2_id, cost = aviso_ac.tracking(riops_v2_ac)
    aviso_id_junk, riops_v2_id_junk = reverse_index(aviso_id, len(aviso_ac)), reverse_index(riops_v2_id, len(riops_v2_ac))
    #### Hits, Misses, False Alarm
    AG_hits[index]   = len(aviso_id)
    AG_misses[index] = len(aviso_id_junk)
    AG_falarm[index] = len(riops_v2_id_junk)
    #### Costs (Hits)
    ##### Compute cost radius
    cost_radii = np.sqrt(((aviso_ac.obs['radius_e'][aviso_id]-riops_v2_ac.obs['radius_e'][riops_v2_id]))**2)/1000
    ##### Compute cost amplitude
    cost_amplitude = np.sqrt(((aviso_ac.obs['amplitude'][aviso_id]-riops_v2_ac.obs['amplitude'][riops_v2_id]))**2)*100
    ##### Compute cost distance
    aviso_lon = np.mean(aviso_ac.obs['contour_lon_e'][aviso_id,:],axis=1)
    aviso_lat = np.mean(aviso_ac.obs['contour_lat_e'][aviso_id,:],axis=1)
    riops_v2_lon = np.mean(riops_v2_ac.obs['contour_lon_e'][riops_v2_id,:],axis=1)
    riops_v2_lat = np.mean(riops_v2_ac.obs['contour_lat_e'][riops_v2_id,:],axis=1)
    dist = distance_grid(aviso_lon,aviso_lat,riops_v2_lon,riops_v2_lat)
    distance = []
    for i in range(len(aviso_id)):
        distance.append(dist[i,i])
    cost_distance = np.array(distance)
    ##### Fill cost lists
    for item in cost:
        AG_costs.append(item)
    for item in cost_radii:
        AG_cost_radius.append(item)
    for item in cost_amplitude:
        AG_cost_amplitudes.append(item)
    for item in cost_distance:
        AG_cost_distances.append(item)
    #### Radius & Amplitudes of Hits, Misses, & False Alarm
    ##### Subset
    aviso_ac_junk = aviso_ac.index(aviso_id_junk)
    riops_v2_ac_junk = riops_v2_ac.index(riops_v2_id_junk)
    riops_v2_ac = riops_v2_ac.index(riops_v2_id)
    for item in riops_v2_ac.obs['radius_e']/1000:
        radius_AG_hits.append(item)
    for item in aviso_ac_junk.obs['radius_e']/1000:
        radius_AG_misses.append(item)
    for item in riops_v2_ac_junk.obs['radius_e']/1000:
        radius_AG_falarm.append(item)
    for item in riops_v2_ac.obs['amplitude']*100:
        amplitude_AG_hits.append(item)
    for item in aviso_ac_junk.obs['amplitude']*100:
        amplitude_AG_misses.append(item)
    for item in riops_v2_ac_junk.obs['amplitude']*100:
        amplitude_AG_falarm.append(item)
    
    ### Cyclonic
    aviso_id, riops_v2_id, cost = aviso_cc.tracking(riops_v2_cc)
    aviso_id_junk, riops_v2_id_junk = reverse_index(aviso_id, len(aviso_cc)), reverse_index(riops_v2_id, len(riops_v2_cc))
    #### Hits, Misses, False Alarm
    AG_hits[index]   += len(aviso_id)
    AG_misses[index] += len(aviso_id_junk)
    AG_falarm[index] += len(riops_v2_id_junk)
    #### Costs (Hits)
    ##### Compute cost radius
    cost_radii = np.sqrt(((aviso_cc.obs['radius_e'][aviso_id]-riops_v2_cc.obs['radius_e'][riops_v2_id]))**2)/1000
    ##### Compute cost amplitude
    cost_amplitude = np.sqrt(((aviso_cc.obs['amplitude'][aviso_id]-riops_v2_cc.obs['amplitude'][riops_v2_id]))**2)*100
    ##### Compute cost distance
    aviso_lon = np.mean(aviso_cc.obs['contour_lon_e'][aviso_id,:],axis=1)
    aviso_lat = np.mean(aviso_cc.obs['contour_lat_e'][aviso_id,:],axis=1)
    riops_v2_lon = np.mean(riops_v2_cc.obs['contour_lon_e'][riops_v2_id,:],axis=1)
    riops_v2_lat = np.mean(riops_v2_cc.obs['contour_lat_e'][riops_v2_id,:],axis=1)
    dist = distance_grid(aviso_lon,aviso_lat,riops_v2_lon,riops_v2_lat)
    distance = []
    for i in range(len(aviso_id)):
        distance.append(dist[i,i])
    cost_distance = np.array(distance)
    ##### Fill cost lists
    for item in cost:
        AG_costs.append(item)
    for item in cost_radii:
        AG_cost_radius.append(item)
    for item in cost_amplitude:
        AG_cost_amplitudes.append(item)
    for item in cost_distance:
        AG_cost_distances.append(item)
    #### Radius & Amplitudes of Hits, Misses, & False Alarm
    # Subset
    aviso_cc_junk = aviso_cc.index(aviso_id_junk)
    riops_v2_cc_junk = riops_v2_cc.index(riops_v2_id_junk)
    riops_v2_cc = riops_v2_cc.index(riops_v2_id)

    for item in riops_v2_cc.obs['radius_e']/1000:
        radius_AG_hits.append(item)
    for item in aviso_cc_junk.obs['radius_e']/1000:
        radius_AG_misses.append(item)
    for item in riops_v2_cc_junk.obs['radius_e']/1000:
        radius_AG_falarm.append(item)
    for item in riops_v2_cc.obs['amplitude']*100:
        amplitude_AG_hits.append(item)
    for item in aviso_cc_junk.obs['amplitude']*100:
        amplitude_AG_misses.append(item)
    for item in riops_v2_cc_junk.obs['amplitude']*100:
        amplitude_AG_falarm.append(item)
        
    ## GORP 1/4°
    ### Anticyclonic
    aviso_id, riops_v4_id, cost = aviso_ac.tracking(riops_v4_ac)
    aviso_id_junk, riops_v4_id_junk = reverse_index(aviso_id, len(aviso_ac)), reverse_index(riops_v4_id, len(riops_v4_ac))
    #### Hits, Misses, False Alarm
    AR_hits[index]   = len(aviso_id)
    AR_misses[index] = len(aviso_id_junk)
    AR_falarm[index] = len(riops_v4_id_junk)
    #### Costs (Hits)
    ##### Compute cost radius
    cost_radii = np.sqrt(((aviso_ac.obs['radius_e'][aviso_id]-riops_v4_ac.obs['radius_e'][riops_v4_id]))**2)/1000
    ##### Compute cost amplitude
    cost_amplitude = np.sqrt(((aviso_ac.obs['amplitude'][aviso_id]-riops_v4_ac.obs['amplitude'][riops_v4_id]))**2)*100
    ##### Compute cost distance
    aviso_lon = np.mean(aviso_ac.obs['contour_lon_e'][aviso_id,:],axis=1)
    aviso_lat = np.mean(aviso_ac.obs['contour_lat_e'][aviso_id,:],axis=1)
    riops_v4_lon = np.mean(riops_v4_ac.obs['contour_lon_e'][riops_v4_id,:],axis=1)
    riops_v4_lat = np.mean(riops_v4_ac.obs['contour_lat_e'][riops_v4_id,:],axis=1)
    dist = distance_grid(aviso_lon,aviso_lat,riops_v4_lon,riops_v4_lat)
    distance = []
    for i in range(len(aviso_id)):
        distance.append(dist[i,i])
    cost_distance = np.array(distance)
    ##### Fill cost lists
    for item in cost:
        AR_costs.append(item)
    for item in cost_radii:
        AR_cost_radius.append(item)
    for item in cost_amplitude:
        AR_cost_amplitudes.append(item)
    for item in cost_distance:
        AR_cost_distances.append(item)
    #### Radius & Amplitudes of Hits, Misses, & False Alarm
    # Subset
    aviso_ac_junk = aviso_ac.index(aviso_id_junk)
    riops_v4_ac_junk = riops_v4_ac.index(riops_v4_id_junk)
    riops_v4_ac = riops_v4_ac.index(riops_v4_id)

    for item in riops_v4_ac.obs['radius_e']/1000:
        radius_AR_hits.append(item)
    for item in aviso_ac_junk.obs['radius_e']/1000:
        radius_AR_misses.append(item)
    for item in riops_v4_ac_junk.obs['radius_e']/1000:
        radius_AR_falarm.append(item)
    for item in riops_v4_ac.obs['amplitude']*100:
        amplitude_AR_hits.append(item)
    for item in aviso_ac_junk.obs['amplitude']*100:
        amplitude_AR_misses.append(item)
    for item in riops_v4_ac_junk.obs['amplitude']*100:
        amplitude_AR_falarm.append(item)
    ### Cyclonic
    aviso_id, riops_v4_id, cost = aviso_cc.tracking(riops_v4_cc)
    aviso_id_junk, riops_v4_id_junk = reverse_index(aviso_id, len(aviso_cc)), reverse_index(riops_v4_id, len(riops_v4_cc))
    #### Hits, Misses, False Alarm
    AR_hits[index]   += len(aviso_id)
    AR_misses[index] += len(aviso_id_junk)
    AR_falarm[index] += len(riops_v4_id_junk)
    #### Costs (Hits)
    ##### Compute cost radius
    cost_radii = np.sqrt(((aviso_cc.obs['radius_e'][aviso_id]-riops_v4_cc.obs['radius_e'][riops_v4_id]))**2)/1000
    ##### Compute cost amplitude
    cost_amplitude = np.sqrt(((aviso_cc.obs['amplitude'][aviso_id]-riops_v4_cc.obs['amplitude'][riops_v4_id]))**2)*100
    ##### Compute cost distance
    aviso_lon = np.mean(aviso_cc.obs['contour_lon_e'][aviso_id,:],axis=1)
    aviso_lat = np.mean(aviso_cc.obs['contour_lat_e'][aviso_id,:],axis=1)
    riops_v4_lon = np.mean(riops_v4_cc.obs['contour_lon_e'][riops_v4_id,:],axis=1)
    riops_v4_lat = np.mean(riops_v4_cc.obs['contour_lat_e'][riops_v4_id,:],axis=1)
    dist = distance_grid(aviso_lon,aviso_lat,riops_v4_lon,riops_v4_lat)
    distance = []
    for i in range(len(aviso_id)):
        distance.append(dist[i,i])
    cost_distance = np.array(distance)
    ##### Fill cost lists
    for item in cost:
        AR_costs.append(item)
    for item in cost_radii:
        AR_cost_radius.append(item)
    for item in cost_amplitude:
        AR_cost_amplitudes.append(item)
    for item in cost_distance:
        AR_cost_distances.append(item)
    #### Radius & Amplitudes of Hits, Misses, & False Alarm
    # Subset
    aviso_cc_junk = aviso_cc.index(aviso_id_junk)
    riops_v4_cc_junk = riops_v4_cc.index(riops_v4_id_junk)
    riops_v4_cc = riops_v4_cc.index(riops_v4_id)

    for item in riops_v4_cc.obs['radius_e']/1000:
        radius_AR_hits.append(item)
    for item in aviso_cc_junk.obs['radius_e']/1000:
        radius_AR_misses.append(item)
    for item in riops_v4_cc_junk.obs['radius_e']/1000:
        radius_AR_falarm.append(item)
    for item in riops_v4_cc.obs['amplitude']*100:
        amplitude_AR_hits.append(item)
    for item in aviso_cc_junk.obs['amplitude']*100:
        amplitude_AR_misses.append(item)
    for item in riops_v4_cc_junk.obs['amplitude']*100:
        amplitude_AR_falarm.append(item)

#####################################################################################

# Amplitude
bins = np.arange(0,64,2)

# Compute BIAS & TS for GORP 1/12°
#hist_hits,   bins_hits   = np.histogram(amplitude_AG_hits,bins=bins)
#hist_misses, bins_misses = np.histogram(amplitude_AG_misses,bins=bins)
#hist_falarm, bins_falarm = np.histogram(amplitude_AG_falarm,bins=bins)
#TS_AG = hist_hits   / (hist_hits + hist_misses + hist_falarm)
#BIAS_AG = (hist_hits + hist_falarm) / (hist_hits + hist_misses)
 
# Compute BIAS & TS for GORP 1/4°
#hist_hits,   bins_hits   = np.histogram(amplitude_AR_hits,bins=bins)
#hist_misses, bins_misses = np.histogram(amplitude_AR_misses,bins=bins)
#hist_falarm, bins_falarm = np.histogram(amplitude_AR_falarm,bins=bins)
#TS_AR = hist_hits   / (hist_hits + hist_misses + hist_falarm)
#BIAS_AR = (hist_hits + hist_falarm) / (hist_hits + hist_misses)

# Compute POD & FAR for GORP 1/12°
hist_hits,   bins_hits   = np.histogram(amplitude_AG_hits,bins=bins)
hist_misses, bins_misses = np.histogram(amplitude_AG_misses,bins=bins)
hist_falarm, bins_falarm = np.histogram(amplitude_AG_falarm,bins=bins)

POD_AG = hist_hits / (hist_hits + hist_misses)
FAR_AG = hist_falarm / (hist_hits + hist_falarm)

POD_AG = np.nan_to_num(POD_AG, nan=0.0)
FAR_AG = np.nan_to_num(FAR_AG, nan=0.0)
hist_hits_12 = hist_hits
hist_misses_12 = hist_misses
hist_falarm_12 = hist_falarm


# Compute POD & FAR for GORP 1/4°
hist_hits,   bins_hits   = np.histogram(amplitude_AR_hits,bins=bins)
hist_misses, bins_misses = np.histogram(amplitude_AR_misses,bins=bins)
hist_falarm, bins_falarm = np.histogram(amplitude_AR_falarm,bins=bins)

POD_AR = hist_hits / (hist_hits + hist_misses)
FAR_AR = hist_falarm / (hist_hits + hist_falarm)

POD_AR = np.nan_to_num(POD_AR, nan=0.0)
FAR_AR = np.nan_to_num(FAR_AR, nan=0.0)

hist_hits_4 = hist_hits
hist_misses_4 = hist_misses
hist_falarm_4 = hist_falarm


# Plot
plt.subplots(figsize=(12,9))
# GORP 1/12° Hits, Misses, False Alarm


# Plot aggiornato con POD e FAR
plt.subplots(figsize=(12,9))

# GORP 1/12° Hits, Misses, False Alarm
plt.subplot(221)
plt.hist(amplitude_AG_hits, bins=bins, fill=False, linewidth=4, histtype='step', label='hits')
plt.hist(amplitude_AG_misses, bins=bins, fill=False, linewidth=3, histtype='step', label='misses')
plt.hist(amplitude_AG_falarm, bins=bins, fill=False, linewidth=2, histtype='step', label='false alarm')
plt.gca().set(title='GLORYS12V1 - DUACS 1/8° Comparison', ylim=(0,90000), xlim=(0,60), xlabel='Amplitude (cm)')
plt.legend(); plt.grid()

# GORP 1/4° Hits, Misses, False Alarm
plt.subplot(222)
plt.hist(amplitude_AR_hits, bins=bins, fill=False, linewidth=4, histtype='step', label='hits')
plt.hist(amplitude_AR_misses, bins=bins, fill=False, linewidth=3, histtype='step', label='misses')
plt.hist(amplitude_AR_falarm, bins=bins, fill=False, linewidth=2, histtype='step', label='false alarm')
plt.gca().set(title='GLORYS2V4 - DUACS 1/8° Comparison', ylim=(0, 90000), xlim=(0,60), xlabel='Amplitude (cm)')
plt.legend(); plt.grid()

# POD (Probability of Detection)
plt.subplot(223)
plt.step(bins[:-1], POD_AG, label='GLORYS12V1', color='black', linewidth=3)
plt.step(bins[:-1], POD_AR, label='GLORYS2V4', color='green', linewidth=2)
plt.gca().set(title='Probability of Detection - DUACS 1/8° Comparison', ylim=(0,1.05), ylabel='POD', xlim=(0,60), xlabel='Amplitude (cm)')
plt.xticks(np.arange(0, 61, 10))
plt.legend(loc='lower right'); plt.grid()

# FAR (False Alarm Ratio)
plt.subplot(224)
plt.step(bins[:-1], FAR_AG, label='GLORYS12V1', color='black', linewidth=3)
plt.step(bins[:-1], FAR_AR, label='GLORYS2V4', color='green', linewidth=2)
plt.gca().set(title='False Alarm Ratio - DUACS 1/8° Comparison', ylim=(0,1.05), ylabel='FAR', xlim=(0,60), xlabel='Amplitude (cm)')
plt.xticks(np.arange(0, 61, 10))
plt.legend(loc='upper right'); plt.grid()
plt.suptitle("", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("Histogram_FAR_POD_Amplitude_duacs_comparison_4days.png", dpi=300)
plt.show()





# ====================================================
# 📈 Cumulative Fraction Plot for Amplitude (GLORYS2V4 and GLORYS12V1)
# ====================================================
def cumulative_fraction(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    cum = np.cumsum(hist) / np.sum(hist)
    return bin_edges[1:], cum

bins_amp = np.arange(0, 64, 0.5)

# GLORYS2V4
bins_hits_aviso, cdf_hits_aviso     = cumulative_fraction(amplitude_AR_hits, bins_amp)
bins_misses_aviso, cdf_misses_aviso = cumulative_fraction(amplitude_AR_misses, bins_amp)
bins_fa_aviso, cdf_fa_aviso         = cumulative_fraction(amplitude_AR_falarm, bins_amp)

# GLORYS12V1
bins_hits_merc, cdf_hits_merc     = cumulative_fraction(amplitude_AG_hits, bins_amp)
bins_misses_merc, cdf_misses_merc = cumulative_fraction(amplitude_AG_misses, bins_amp)
bins_fa_merc, cdf_fa_merc         = cumulative_fraction(amplitude_AG_falarm, bins_amp)

# Plot
plt.figure(figsize=(10,6))
plt.plot(bins_hits_merc, cdf_hits_merc, label='Hits GLORYS12V1', color='dodgerblue')
plt.plot(bins_hits_aviso, cdf_hits_aviso, linestyle='dotted', label='Hits GLORYS2V4', color='dodgerblue', linewidth=2)
plt.plot(bins_misses_merc, cdf_misses_merc, label='Misses Hits GLORYS12V1', color='orange')
plt.plot(bins_misses_aviso, cdf_misses_aviso, linestyle='dotted', label='Misses GLORYS2V4', color='orange', linewidth=2)

plt.plot(bins_fa_merc, cdf_fa_merc, label='False alarms GLORYS12V1', color='green')
plt.plot(bins_fa_aviso, cdf_fa_aviso, linestyle='dotted', label='False alarm GLORYS2V4', color='green', linewidth=2)

import matplotlib.ticker as ticker

TITLE_FS  = 22
LABEL_FS  = 20
TICK_FS   = 16
LEGEND_FS = 14

ax = plt.gca()

ax.set_xlabel("Amplitude (cm)", fontsize=LABEL_FS, fontweight="bold", labelpad=10)
ax.set_ylabel("", fontsize=LABEL_FS, fontweight="bold", labelpad=10)

ax.set_title("Cumulative Fraction - DUACS 1/8° Comparison",
             fontsize=TITLE_FS, fontweight="bold", pad=12)

ax.set_xlim(0, 60)
ax.set_ylim(0, 1.05)

ax.tick_params(axis="both", which="major", labelsize=TICK_FS, width=1.8, length=8, direction="out")
ax.tick_params(axis="both", which="minor", width=1.2, length=4, direction="out")

ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

for s in ax.spines.values():
    s.set_linewidth(1.6)

ax.grid(True, which="major", linewidth=0.8, alpha=0.6)
ax.grid(True, which="minor", linestyle="--", alpha=0.25)

ax.legend(fontsize=LEGEND_FS, frameon=True, framealpha=0.95, loc="lower right")

plt.tight_layout()
plt.savefig(
    "Cumulative_Fraction_Amplitude_GLORYS2V4_vs_GLORYS12V1_4days_duacs.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()



def print_quartiles(data, label):
    q25, q50, q75, q90 = np.percentile(data, [25, 50, 75, 90])
    print(f"{label}\n Quartile (25%):  {q25:.2f} (cm)\n Quartile (50%):  {q50:.2f} (cm)\n Quartile (75%):  {q75:.2f} (cm)\n Quartile (90%): {q90:.2f} (cm)")

print("\n📊 Quartiles amplitude for GLORYS2V4")
print_quartiles(amplitude_AR_hits, "Hits")
print_quartiles(amplitude_AR_misses, "Misses")
print_quartiles(amplitude_AR_falarm, "False Alarm")

print("\n📊 Quartiles amplitude for GLORYS12V1")
print_quartiles(amplitude_AG_hits, "Hits")
print_quartiles(amplitude_AG_misses, "Misses")
print_quartiles(amplitude_AG_falarm, "False Alarm")

print("Min amplitude AVISO hits:", np.min(amplitude_AR_hits))
print("Min amplitude MERCATOR hits:", np.min(amplitude_AG_hits))

def get_cdf_value(bins, cdf, threshold):
    # Trova l'indice del bin più vicino a 10 cm
    idx = np.searchsorted(bins, threshold, side="right") - 1
    return cdf[idx] * 100  # in percentuale

print(f"Hits AVISO: {get_cdf_value(bins_hits_aviso, cdf_hits_aviso, 10):.2f}% sotto 10 cm")
print(f"Misses AVISO: {get_cdf_value(bins_misses_aviso, cdf_misses_aviso, 10):.2f}% sotto 10 cm")
print(f"False Alarm AVISO: {get_cdf_value(bins_fa_aviso, cdf_fa_aviso, 10):.2f}% sotto 10 cm")

print(f"Hits MERCATOR: {get_cdf_value(bins_hits_merc, cdf_hits_merc, 10):.2f}% sotto 10 cm")
print(f"Misses MERCATOR: {get_cdf_value(bins_misses_merc, cdf_misses_merc, 10):.2f}% sotto 10 cm")
print(f"False Alarm MERCATOR: {get_cdf_value(bins_fa_merc, cdf_fa_merc, 10):.2f}% sotto 10 cm")


