from py_eddy_tracker.observations.tracking import TrackEddiesObservations
from py_eddy_tracker.generic import distance_grid, reverse_index
from netCDF4 import num2date
import numpy as np
firstdataset = "GLORYS12V1"
seconddataset = "GLORYS2V4"

# Tracking data loading
#filename_AVISO_AC = '/ec/res4/scratch/ita6648/AVISO_DUACS/eddy_output_north_atlantic/eddy_tracking/Anticyclonic_4days.nc'
#filename_AVISO_CC = '/ec/res4/scratch/ita6648/AVISO_DUACS/eddy_output_north_atlantic/eddy_tracking/Cyclonic_4days.nc'
filename_AVISO_AC = '/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/Anticyclonic_4days.nc'
filename_AVISO_CC= '/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/Cyclonic_4days.nc'
#/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking
filename_RIOPS_v2_AC = '/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/Anticyclonic_cmems_4days.nc'
filename_RIOPS_v2_CC = '/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_1_8/Cyclonic_cmems_4days.nc'
#/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/eddy_tracking
filename_RIOPS_v4_AC = '/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/Anticyclonic_cmems_4days.nc'
filename_RIOPS_v4_CC = '/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/Cyclonic_cmems_4days.nc'
# Caricamento file
AVISO_AC = TrackEddiesObservations.load_file(filename_AVISO_AC)
AVISO_CC = TrackEddiesObservations.load_file(filename_AVISO_CC)
RIOPS_v2_AC = TrackEddiesObservations.load_file(filename_RIOPS_v2_AC)
RIOPS_v2_CC = TrackEddiesObservations.load_file(filename_RIOPS_v2_CC)
RIOPS_v4_AC = TrackEddiesObservations.load_file(filename_RIOPS_v4_AC)
RIOPS_v4_CC = TrackEddiesObservations.load_file(filename_RIOPS_v4_CC)

# Giorni da processare
ndays = 640

# Risultati giornalieri
results = []

for inc in range(ndays):
    date = RIOPS_v2_AC.obs['time'].min() + inc
    date1 = date
    date2 = AVISO_AC.obs['time'].min() + inc
    date_str = str(num2date(date, 'days since 1950-01-01', 'standard'))[:10]

    # Estrai le tracce del giorno
    aviso_ac = AVISO_AC.extract_with_period((date2, date2))
    aviso_cc = AVISO_CC.extract_with_period((date2, date2))
    riops_ac = RIOPS_v4_AC.extract_with_period((date1, date1))
    riops_cc = RIOPS_v4_CC.extract_with_period((date1, date1))
    giops_ac = RIOPS_v2_AC.extract_with_period((date1, date1))
    giops_cc = RIOPS_v2_CC.extract_with_period((date1, date1))

    def compute_scores(aviso, model):
        aviso_id, model_id, _ = aviso.tracking(model)
        aviso_miss = reverse_index(aviso_id, len(aviso))
        model_fa = reverse_index(model_id, len(model))
        hits = len(aviso_id)
        misses = len(aviso_miss)
        false_alarms = len(model_fa)
        return hits, misses, false_alarms

    # RIOPS vs AVISO
    h1_ac, m1_ac, f1_ac = compute_scores(aviso_ac, riops_ac)
    h1_cc, m1_cc, f1_cc = compute_scores(aviso_cc, riops_cc)

    # GIOPS vs AVISO
    h2_ac, m2_ac, f2_ac = compute_scores(aviso_ac, giops_ac)
    h2_cc, m2_cc, f2_cc = compute_scores(aviso_cc, giops_cc)

    # Totali per il giorno
    hits_riops = h1_ac + h1_cc
    misses_riops = m1_ac + m1_cc
    fa_riops = f1_ac + f1_cc

    hits_giops = h2_ac + h2_cc
    misses_giops = m2_ac + m2_cc
    fa_giops = f2_ac + f2_cc

    # POD e FAR
    pod_riops = hits_riops / (hits_riops + misses_riops) if (hits_riops + misses_riops) > 0 else np.nan
    far_riops = fa_riops / (fa_riops + hits_riops) if (fa_riops + hits_riops) > 0 else np.nan

    pod_giops = hits_giops / (hits_giops + misses_giops) if (hits_giops + misses_giops) > 0 else np.nan
    far_giops = fa_giops / (fa_giops + hits_giops) if (fa_giops + hits_giops) > 0 else np.nan

    # Salva risultati
    results.append({
        'date': date_str,
        'POD_RIOPS': pod_riops,
        'FAR_RIOPS': far_riops,
        'POD_GIOPS': pod_giops,
        'FAR_GIOPS': far_giops
    })

# Visualizza i risultati
import pandas as pd
df = pd.DataFrame(results)
print(df)



import matplotlib.pyplot as plt

# Assicurati che 'date' sia in formato datetime
df['date'] = pd.to_datetime(df['date'])
# Calcolo dei valori medi
pod_riops_mean = df["POD_RIOPS"].mean()
far_riops_mean = df["FAR_RIOPS"].mean()
pod_giops_mean = df["POD_GIOPS"].mean()
far_giops_mean = df["FAR_GIOPS"].mean()

# Crea figura con due subplot (POD sopra, FAR sotto)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Subplot POD
ax1.plot(df['date'], df['POD_RIOPS'], label=f"POD {firstdataset}  ({pod_riops_mean:.5f})", color='blue')
ax1.plot(df['date'], df['POD_GIOPS'], label=f"POD {seconddataset} ({pod_giops_mean:.5f})", color='green')
ax1.set_ylabel("POD")
ax1.set_title(f"POD - 4 days lifetime threshold - DUACS 1/8°")
ax1.grid(True)
ax1.legend()

# Subplot FAR

ax2.plot(df['date'], df['FAR_RIOPS'], label=f"FAR {firstdataset} ({far_riops_mean:.5f})", color='blue')
ax2.plot(df['date'], df['FAR_GIOPS'], label=f"FAR {seconddataset} ({far_giops_mean:.5f})", color='green')
ax2.set_ylabel("FAR")
ax2.set_title(f"FAR - 4 days lifetime threshold - DUACS 1/8°")
ax2.set_xlabel("Data")
ax2.grid(True)
ax2.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'POD_FAR_TimeSeries_remapped_duacs_{len(df)}.png', dpi=300)
plt.show()
print(f'\n\nPOD_FAR_TimeSeries_remapped_duacs_{len(df)}.png\n')


