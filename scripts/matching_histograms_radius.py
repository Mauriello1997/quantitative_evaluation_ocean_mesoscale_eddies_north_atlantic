from py_eddy_tracker.observations.tracking import TrackEddiesObservations
from py_eddy_tracker.dataset.grid import RegularGridDataset
from py_eddy_tracker.generic import distance_grid

from netCDF4 import num2date
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

# Anticyclonic and Cyclonic 31days tracking
#filename_AVISO_AC = '/ec/res4/scratch/ita6648/AVISO_DUACS/eddy_output_north_atlantic/eddy_tracking/Anticyclonic_4days.nc'
#filename_AVISO_CC = '/ec/res4/scratch/ita6648/AVISO_DUACS/eddy_output_north_atlantic/eddy_tracking/Cyclonic_4days.nc'
filename_AVISO_AC = '/ec/res4/scratch/ita6648/miost_science/eddy_output_north_atlantic/eddy_tracking/Anticyclonic_cmems_4days.nc'
filename_AVISO_CC= '/ec/res4/scratch/ita6648/miost_science/eddy_output_north_atlantic/eddy_tracking/Cyclonic_cmems_4days.nc'
#/ec/res4/scratch/ita6648/miost_science/eddy_output_north_atlantic/eddy_tracking
filename_RIOPS_v2_AC = '/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking/Anticyclonic_cmems_4days.nc'

filename_RIOPS_v2_CC = '/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking/Cyclonic_cmems_4days.nc'

filename_RIOPS_v4_AC = '/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_rid/eddy_tracking/Anticyclonic_cmems_4days.nc'
filename_RIOPS_v4_CC = '/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/eddy_tracking/Cyclonic_cmems_4days.nc'

AVISO_AC = TrackEddiesObservations.load_file(filename_AVISO_AC)
AVISO_CC = TrackEddiesObservations.load_file(filename_AVISO_CC)
RIOPS_v2_AC = TrackEddiesObservations.load_file(filename_RIOPS_v2_AC)
RIOPS_v2_CC = TrackEddiesObservations.load_file(filename_RIOPS_v2_CC)
RIOPS_v4_AC = TrackEddiesObservations.load_file(filename_RIOPS_v4_AC)
RIOPS_v4_CC = TrackEddiesObservations.load_file(filename_RIOPS_v4_CC)

dates = np.arange(RIOPS_v2_AC.obs['time'].min(),RIOPS_v2_AC.obs['time'].max()) # dates since 1950-01-01 (01/01/20 to 29/12/2021)

# Hits, Misses, False Alarm
## RIOPS_v2
AG_hits   = np.ma.masked_array(np.zeros((len(dates))),mask=True)
AG_misses = np.ma.masked_array(np.zeros((len(dates))),mask=True)
AG_falarm = np.ma.masked_array(np.zeros((len(dates))),mask=True)
## RIOPS_v4
AR_hits   = np.ma.masked_array(np.zeros((len(dates))),mask=True)
AR_misses = np.ma.masked_array(np.zeros((len(dates))),mask=True)
AR_falarm = np.ma.masked_array(np.zeros((len(dates))),mask=True)

# Costs (Hits)
## RIOPS_v2
AG_costs           = []
AG_cost_radius     = []
AG_cost_amplitudes = []
AG_cost_distances  = []
## RIOPS_v4
AR_costs           = []
AR_cost_radius     = []
AR_cost_amplitudes = []
AR_cost_distances  = []

# Radius & Amplitudes of Hits, Misses, & False Alarm
## RIOPS_v2
### Radius
radius_AG_hits       = []
radius_AG_falarm     = []
radius_AG_misses     = []
### Amplitude
amplitude_AG_hits    = []
amplitude_AG_falarm  = []
amplitude_AG_misses  = []
## RIOPS_v4
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
    aviso_ac = AVISO_AC.extract_with_period((round(date),round(date)))
    riops_v2_ac = RIOPS_v2_AC.extract_with_period((date,date))
    riops_v4_ac = RIOPS_v4_AC.extract_with_period((date,date))
    ## Cyclonic
    aviso_cc = AVISO_CC.extract_with_period((round(date),round(date)))
    riops_v2_cc = RIOPS_v2_CC.extract_with_period((date,date))
    riops_v4_cc = RIOPS_v4_CC.extract_with_period((date,date))
    
    # Fill arrays
    ## RIOPS_v2
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
        
    ## RIOPS_v4
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

######################################################

# Radius

bins = np.arange(0,250,10)

plt.subplots(figsize=(12,9))

# RIOPS_v2 Hits, Misses, False Alarm
plt.subplot(221)
plt.hist(radius_AG_hits,  bins=bins,fill=False,linewidth=4,histtype='step',label='hits')
plt.hist(radius_AG_misses,bins=bins,fill=False,linewidth=3,histtype='step',label='misses')
plt.hist(radius_AG_falarm,bins=bins,fill=False,linewidth=2,histtype='step',label='false alarm')
plt.gca().set(title='GLORYS2V4',ylim=(0,50000),xlim=(20,180),xlabel='Radius (km)')
plt.xticks(np.arange(20,200,20))
plt.legend()
# RIOPS_v4 Hits, Misses, False Alarm
plt.subplot(222)
plt.hist(radius_AR_hits,  bins=bins,fill=False,linewidth=4,histtype='step',label='hits')
plt.hist(radius_AR_misses,bins=bins,fill=False,linewidth=3,histtype='step',label='misses')
plt.hist(radius_AR_falarm,bins=bins,fill=False,linewidth=2,histtype='step',label='false alarm')
plt.gca().set(title='GLORYS12V1',ylim=(0,50000),xlim=(20,180),xlabel='Radius (km)')
plt.xticks(np.arange(20,200,20))
plt.legend()

# Compute BIAS & TS for RIOPS_v2
hist_hits,   bins_hits   = np.histogram(radius_AG_hits,bins=bins)
hist_misses, bins_misses = np.histogram(radius_AG_misses,bins=bins)
hist_falarm, bins_falarm = np.histogram(radius_AG_falarm,bins=bins)
TS_AG = hist_hits   / (hist_hits + hist_misses + hist_falarm)
BIAS_AG = (hist_hits + hist_falarm) / (hist_hits + hist_misses)

# Compute BIAS & TS for RIOPS_v4
hist_hits,   bins_hits   = np.histogram(radius_AR_hits,bins=bins)
hist_misses, bins_misses = np.histogram(radius_AR_misses,bins=bins)
hist_falarm, bins_falarm = np.histogram(radius_AR_falarm,bins=bins)
TS_AR = hist_hits   / (hist_hits + hist_misses + hist_falarm)
BIAS_AR = (hist_hits + hist_falarm) / (hist_hits + hist_misses)

# Compute POD & FAR for RIOPS_v2
hist_hits, _ = np.histogram(radius_AG_hits, bins=bins)
hist_misses, _ = np.histogram(radius_AG_misses, bins=bins)
hist_falarm, _ = np.histogram(radius_AG_falarm, bins=bins)
POD_AG = hist_hits / (hist_hits + hist_misses + 1e-10)
FAR_AG = hist_falarm / (hist_hits + hist_falarm + 1e-10)

# Compute POD & FAR for RIOPS_v4
hist_hits, _ = np.histogram(radius_AR_hits, bins=bins)
hist_misses, _ = np.histogram(radius_AR_misses, bins=bins)
hist_falarm, _ = np.histogram(radius_AR_falarm, bins=bins)
POD_AR = hist_hits / (hist_hits + hist_misses + 1e-10)
FAR_AR = hist_falarm / (hist_hits + hist_falarm + 1e-10)


# Threat Score
# POD
plt.subplot(223)
plt.step(bins[:-1], POD_AG, label='GLORYS2V4', color='black', linewidth=3)
plt.step(bins[:-1], POD_AR, label='GLORYS12V1', color='green', linewidth=3)
plt.gca().set(title='Probability of Detection - SWOT Comparison', ylim=(0,1.1), ylabel='POD', xlim=(20,180), xlabel='Radius (km)')
plt.xticks(np.arange(20,200,20))
plt.legend(loc='lower left'); plt.grid()

# FAR
plt.subplot(224)
plt.step(bins[:-1], FAR_AG, label='GLORYS2V4', color='black', linewidth=3)
plt.step(bins[:-1], FAR_AR, label='GLORYS12V1', color='green', linewidth=3)
plt.gca().set(title='False Alarm Ratio - SWOT Comparison', ylim=(0,1.1), ylabel='FAR', xlim=(20,180), xlabel='Radius (km)')
plt.xticks(np.arange(20,200,20))
plt.legend(loc='upper left'); plt.grid()
plt.suptitle("", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("Histogram_FAR_POD_Radius_swot_4days.png", dpi=300)
#plt.savefig("Histogram_False_Alarm_Ratio_Radius.pdf", dpi=300)
plt.show()

def cumulative_fraction(data, bins):
    hist, bin_edges = np.histogram(data, bins=bins, density=False)
    cum = np.cumsum(hist) / np.sum(hist)
    return bin_edges[1:], cum

bins = np.arange(0, 200, 5)

#CDF calculation

bins_hits_aviso, cdf_hits_aviso     = cumulative_fraction(radius_AR_hits, bins)
bins_misses_aviso, cdf_misses_aviso = cumulative_fraction(radius_AR_misses, bins)
bins_fa_aviso, cdf_fa_aviso         = cumulative_fraction(radius_AR_falarm, bins)

bins_hits_merc, cdf_hits_merc     = cumulative_fraction(radius_AG_hits, bins)
bins_misses_merc, cdf_misses_merc = cumulative_fraction(radius_AG_misses, bins)
bins_fa_merc, cdf_fa_merc         = cumulative_fraction(radius_AG_falarm, bins)

# Plot
plt.figure(figsize=(10,6))

# Hits
plt.plot(bins_hits_aviso, cdf_hits_aviso, label='Hits GLORYS12V1', color='dodgerblue', linewidth=2)
plt.plot(bins_hits_merc, cdf_hits_merc, linestyle='dotted', label='Hits GLORYS2V4', color='dodgerblue')

# Misses
plt.plot(bins_misses_aviso, cdf_misses_aviso, label='Misses GLORYS12V1', color='orange', linewidth=2)
plt.plot(bins_misses_merc, cdf_misses_merc, linestyle='dotted', label='Misses GLORYS2V4', color='orange')

# False Alarms
plt.plot(bins_fa_aviso, cdf_fa_aviso, label='False alarm GLORYS12V1', color='green', linewidth=2)
plt.plot(bins_fa_merc, cdf_fa_merc, linestyle='dotted', label='False Alarms GLORYS2V4', color='green')

import matplotlib.ticker as ticker

TITLE_FS  = 22
LABEL_FS  = 20
TICK_FS   = 16
LEGEND_FS = 14

ax = plt.gca()

ax.set_xlabel("Radius (km)", fontsize=LABEL_FS, fontweight="bold", labelpad=10)
ax.set_ylabel("", fontsize=LABEL_FS, fontweight="bold", labelpad=10)

ax.set_title("Cumulative Fraction - SWOT Comparison",
             fontsize=TITLE_FS, fontweight="bold", pad=12)

ax.set_xlim(0, 160)
ax.set_ylim(0, 1.05)

ax.tick_params(axis="both", which="major", labelsize=TICK_FS, width=1.8, length=8, direction="out")
ax.tick_params(axis="both", which="minor", width=1.2, length=4, direction="out")

ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

for s in ax.spines.values():
    s.set_linewidth(1.6)

ax.grid(True, which="major", linewidth=0.8, alpha=0.6)
ax.grid(True, which="minor", linestyle="--", alpha=0.25)

# Legenda
ax.legend(fontsize=LEGEND_FS, frameon=True, framealpha=0.95, loc="lower right")

plt.tight_layout()
plt.savefig(
    "Cumulative_Fraction_radius_GLORYS2V4_vs_GLORYS12V1_4days_swot.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)
plt.show()

def print_quartiles(data, label):
    q25, q50, q75, q90 = np.percentile(data, [25, 50, 75, 90])
    print(f"{label}\n Quartile (25%):  {q25:.2f} (km)\n Quartile (50%):  {q50:.2f} (km)\n Quartile (75%):  {q75:.2f} (km) \n Quartile (90%): {q90:.2f} (Km)")

print("\n📊 Quartiles for GLORYS2V4")
print_quartiles(radius_AR_hits, "Hits")
print_quartiles(radius_AR_misses, "Misses")
print_quartiles(radius_AR_falarm, "False Alarm")

print("\n📊 Quartiles for GLORYS12V1")
print_quartiles(radius_AG_hits, "Hits")
print_quartiles(radius_AG_misses, "Misses")
print_quartiles(radius_AG_falarm, "False Alarm")
