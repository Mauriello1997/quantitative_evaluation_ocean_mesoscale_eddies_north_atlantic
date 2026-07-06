from py_eddy_tracker.observations.tracking import TrackEddiesObservations
from py_eddy_tracker.generic import distance_grid

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'serif', 'weight': 'bold', 'size': 14}
matplotlib.rc('font', **font)


def reverse_index(index, nb):
    """Return indices that are NOT present in index."""
    m = np.ones(nb, dtype="bool")
    m[index] = False
    return np.where(m)[0]


def eddy_center_from_contour(obs, ids):
    """Compute an approximate eddy centre as the mean of the effective contour coordinates."""
    lon = np.mean(obs.obs['contour_lon_e'][ids, :], axis=1)
    lat = np.mean(obs.obs['contour_lat_e'][ids, :], axis=1)
    return lon, lat


def append_all(target_list, values):
    """Append all values from a numpy array/list into a Python list."""
    target_list.extend(np.asarray(values).tolist())


def compare_one_day(model_obs, ref_obs):
    """
    Compare one model eddy set against one reference eddy set for one day.

    IMPORTANT DEFINITIONS
    ---------------------
    model_obs: GLORYS product to be evaluated.
    ref_obs:   reference product, here SWOT/MIOST.

    hits        = matched model-reference eddies.
    misses      = reference eddies not detected by the model.
    false alarms= model eddies not present in the reference.

    The returned errors are computed ONLY for matched eddies.
    """
    model_id, ref_id, cost = model_obs.tracking(ref_obs)

    # Correct verification logic when ref_obs is the reference dataset.
    model_id_junk = reverse_index(model_id, len(model_obs))  # false alarms
    ref_id_junk = reverse_index(ref_id, len(ref_obs))        # misses

    # Errors on matched eddies only.
    radius_error = np.abs(model_obs.obs['radius_e'][model_id] - ref_obs.obs['radius_e'][ref_id]) / 1000.0
    amplitude_error = np.abs(model_obs.obs['amplitude'][model_id] - ref_obs.obs['amplitude'][ref_id]) * 100.0

    model_lon, model_lat = eddy_center_from_contour(model_obs, model_id)
    ref_lon, ref_lat = eddy_center_from_contour(ref_obs, ref_id)
    dist_matrix = distance_grid(model_lon, model_lat, ref_lon, ref_lat)
    distance_error = np.array([dist_matrix[i, i] for i in range(len(model_id))])

    # For scale-dependent diagnostics.
    # Hits and misses are stored using reference eddy properties.
    # False alarms are stored using model eddy properties.
    radius_hits = ref_obs.obs['radius_e'][ref_id] / 1000.0
    radius_misses = ref_obs.obs['radius_e'][ref_id_junk] / 1000.0
    radius_falarm = model_obs.obs['radius_e'][model_id_junk] / 1000.0

    amplitude_hits = ref_obs.obs['amplitude'][ref_id] * 100.0
    amplitude_misses = ref_obs.obs['amplitude'][ref_id_junk] * 100.0
    amplitude_falarm = model_obs.obs['amplitude'][model_id_junk] * 100.0

    return {
        'hits': len(model_id),
        'misses': len(ref_id_junk),
        'false_alarms': len(model_id_junk),
        'cost': cost,
        'radius_error': radius_error,
        'amplitude_error': amplitude_error,
        'distance_error': distance_error,
        'radius_hits': radius_hits,
        'radius_misses': radius_misses,
        'radius_falarm': radius_falarm,
        'amplitude_hits': amplitude_hits,
        'amplitude_misses': amplitude_misses,
        'amplitude_falarm': amplitude_falarm,
    }


def merge_day_result(storage, result, index):
    """Add one AC/CC daily comparison result to cumulative arrays/lists."""
    storage['hits'][index] += result['hits']
    storage['misses'][index] += result['misses']
    storage['false_alarms'][index] += result['false_alarms']

    append_all(storage['costs'], result['cost'])
    append_all(storage['radius_errors'], result['radius_error'])
    append_all(storage['amplitude_errors'], result['amplitude_error'])
    append_all(storage['distance_errors'], result['distance_error'])

    append_all(storage['radius_hits'], result['radius_hits'])
    append_all(storage['radius_misses'], result['radius_misses'])
    append_all(storage['radius_falarm'], result['radius_falarm'])
    append_all(storage['amplitude_hits'], result['amplitude_hits'])
    append_all(storage['amplitude_misses'], result['amplitude_misses'])
    append_all(storage['amplitude_falarm'], result['amplitude_falarm'])


def new_storage(n_dates):
    return {
        'hits': np.zeros(n_dates, dtype=float),
        'misses': np.zeros(n_dates, dtype=float),
        'false_alarms': np.zeros(n_dates, dtype=float),
        'costs': [],
        'radius_errors': [],
        'amplitude_errors': [],
        'distance_errors': [],
        'radius_hits': [],
        'radius_misses': [],
        'radius_falarm': [],
        'amplitude_hits': [],
        'amplitude_misses': [],
        'amplitude_falarm': [],
    }


def compute_scores(storage):
    h = storage['hits']
    m = storage['misses']
    f = storage['false_alarms']

    total_hits = np.nansum(h)
    total_misses = np.nansum(m)
    total_false_alarms = np.nansum(f)

    aggregated_pod = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else np.nan
    aggregated_far = total_false_alarms / (total_hits + total_false_alarms) if (total_hits + total_false_alarms) > 0 else np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        daily_pod = np.where((h + m) > 0, h / (h + m), np.nan)
        daily_far = np.where((h + f) > 0, f / (h + f), np.nan)

    return {
        'total_hits': total_hits,
        'total_misses': total_misses,
        'total_false_alarms': total_false_alarms,
        'aggregated_pod': aggregated_pod,
        'aggregated_far': aggregated_far,
        'daily_pod': daily_pod,
        'daily_far': daily_far,
        'daily_pod_mean': np.nanmean(daily_pod),
        'daily_pod_std': np.nanstd(daily_pod),
        'daily_far_mean': np.nanmean(daily_far),
        'daily_far_std': np.nanstd(daily_far),
    }


def mean_std(values):
    values = np.asarray(values)
    return np.nanmean(values), np.nanstd(values)


def hist_weights(values):
    values = np.asarray(values)
    if len(values) == 0:
        return None
    return np.ones_like(values, dtype=float) / float(len(values))


def count_outside(values, xmin, xmax):
    values = np.asarray(values)
    return int(np.sum((values < xmin) | (values > xmax)))


# -----------------------------------------------------------------------------
# Input files
# -----------------------------------------------------------------------------
# Reference: SWOT/MIOST
filename_REF_AC = '/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/Anticyclonic_4days.nc'
filename_REF_CC = '/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/Cyclonic_4days.nc'

# Model 1: GLORYS2V4
filename_GLORYS2V4_AC = '/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/Anticyclonic_cmems_4days.nc'
filename_GLORYS2V4_CC = '/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/eddy_output_north_atlantic_remapped_1_8/Cyclonic_cmems_4days.nc'

# Model 2: GLORYS12V1
filename_GLORYS12V1_AC = '/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/Anticyclonic_cmems_4days.nc'
filename_GLORYS12V1_CC = '/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/Cyclonic_cmems_4days.nc'

REF_AC = TrackEddiesObservations.load_file(filename_REF_AC)
REF_CC = TrackEddiesObservations.load_file(filename_REF_CC)
GLORYS2V4_AC = TrackEddiesObservations.load_file(filename_GLORYS2V4_AC)
GLORYS2V4_CC = TrackEddiesObservations.load_file(filename_GLORYS2V4_CC)
GLORYS12V1_AC = TrackEddiesObservations.load_file(filename_GLORYS12V1_AC)
GLORYS12V1_CC = TrackEddiesObservations.load_file(filename_GLORYS12V1_CC)

# Use the common model period and include the last day.
time_min = int(max(GLORYS2V4_AC.obs['time'].min(), GLORYS12V1_AC.obs['time'].min(), REF_AC.obs['time'].min()))
time_max = int(min(GLORYS2V4_AC.obs['time'].max(), GLORYS12V1_AC.obs['time'].max(), REF_AC.obs['time'].max()))
dates = np.arange(time_min, time_max + 1)

GLORYS2 = new_storage(len(dates))
GLORYS12 = new_storage(len(dates))

# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------
for index, date in enumerate(dates):
    # Extract reference eddies for this day
    ref_ac = REF_AC.extract_with_period((date, date))
    ref_cc = REF_CC.extract_with_period((date, date))

    # Extract model eddies for this day
    g2_ac = GLORYS2V4_AC.extract_with_period((date, date))
    g2_cc = GLORYS2V4_CC.extract_with_period((date, date))
    g12_ac = GLORYS12V1_AC.extract_with_period((date, date))
    g12_cc = GLORYS12V1_CC.extract_with_period((date, date))

    # GLORYS2V4 vs SWOT/MIOST
    merge_day_result(GLORYS2, compare_one_day(g2_ac, ref_ac), index)
    merge_day_result(GLORYS2, compare_one_day(g2_cc, ref_cc), index)

    # GLORYS12V1 vs SWOT/MIOST
    # This is the main correction: compare GLORYS12V1 directly against the reference,
    # not against GLORYS2V4.
    merge_day_result(GLORYS12, compare_one_day(g12_ac, ref_ac), index)
    merge_day_result(GLORYS12, compare_one_day(g12_cc, ref_cc), index)

# -----------------------------------------------------------------------------
# POD/FAR
# -----------------------------------------------------------------------------
score_G2 = compute_scores(GLORYS2)
score_G12 = compute_scores(GLORYS12)

with open('POD_FAR_results_FIXED.txt', 'w') as f:
    f.write('# POD and FAR Results - corrected definitions\n')
    f.write('# Reference product: SWOT/MIOST\n')
    f.write('# misses = reference eddies not detected by the model\n')
    f.write('# false alarms = model eddies not present in the reference\n\n')

    f.write(f"Aggregated POD GLORYS2V4: {score_G2['aggregated_pod']:.3f}\n")
    f.write(f"Aggregated FAR GLORYS2V4: {score_G2['aggregated_far']:.3f}\n")
    f.write(f"Aggregated POD GLORYS12V1: {score_G12['aggregated_pod']:.3f}\n")
    f.write(f"Aggregated FAR GLORYS12V1: {score_G12['aggregated_far']:.3f}\n\n")

    f.write(f"Daily POD GLORYS2V4: {score_G2['daily_pod_mean']:.3f} ± {score_G2['daily_pod_std']:.3f}\n")
    f.write(f"Daily FAR GLORYS2V4: {score_G2['daily_far_mean']:.3f} ± {score_G2['daily_far_std']:.3f}\n")
    f.write(f"Daily POD GLORYS12V1: {score_G12['daily_pod_mean']:.3f} ± {score_G12['daily_pod_std']:.3f}\n")
    f.write(f"Daily FAR GLORYS12V1: {score_G12['daily_far_mean']:.3f} ± {score_G12['daily_far_std']:.3f}\n")

print('\nPOD/FAR esportati in: POD_FAR_results_FIXED.txt')
print('\nPOD e FAR aggregati:')
print(f"GLORYS2V4  -> POD: {score_G2['aggregated_pod']:.3f} | FAR: {score_G2['aggregated_far']:.3f}")
print(f"GLORYS12V1 -> POD: {score_G12['aggregated_pod']:.3f} | FAR: {score_G12['aggregated_far']:.3f}")

print('\nPOD/FAR giornaliero medio e deviazione standard:')
print(f"GLORYS2V4  -> POD: {score_G2['daily_pod_mean']:.3f} ± {score_G2['daily_pod_std']:.3f} | FAR: {score_G2['daily_far_mean']:.3f} ± {score_G2['daily_far_std']:.3f}")
print(f"GLORYS12V1 -> POD: {score_G12['daily_pod_mean']:.3f} ± {score_G12['daily_pod_std']:.3f} | FAR: {score_G12['daily_far_mean']:.3f} ± {score_G12['daily_far_std']:.3f}")

# Daily POD/FAR plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(score_G2['daily_pod'], label='POD GLORYS2V4', color='black')
plt.plot(score_G12['daily_pod'], label='POD GLORYS12V1', color='green')
plt.title('Daily POD')
plt.xlabel('Days')
plt.ylabel('POD')
plt.ylim(0, 1.05)
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(score_G2['daily_far'], label='FAR GLORYS2V4', color='black')
plt.plot(score_G12['daily_far'], label='FAR GLORYS12V1', color='green')
plt.title('Daily FAR')
plt.xlabel('Days')
plt.ylabel('FAR')
plt.ylim(0, 1.05)
plt.grid()
plt.legend()

plt.tight_layout()
plt.savefig('daily_POD_FAR_trends_swot_FIXED.png', dpi=300)
plt.show()

# -----------------------------------------------------------------------------
# Figure 14: cost/error distributions for matched eddies
# -----------------------------------------------------------------------------
TITLE_FS = 16
plt.subplots(figsize=(12, 6))
plt.suptitle('Hits cost - DUACS Comparison', fontsize=TITLE_FS, fontweight='bold')

plt.subplot(221)
plt.hist(GLORYS2['costs'], bins=np.arange(0, 6, 0.1), fill=False, edgecolor='k', linewidth=3,
         histtype='step', label='GLORYS2V4', weights=hist_weights(GLORYS2['costs']))
plt.hist(GLORYS12['costs'], bins=np.arange(0, 6, 0.1), fill=False, edgecolor='g', linewidth=3,
         histtype='step', label='GLORYS12V1', weights=hist_weights(GLORYS12['costs']))
plt.gca().set(xlim=(0, 6), ylim=(0, 0.11), xlabel='Cost')
plt.legend()
plt.grid()

plt.subplot(222)
plt.hist(GLORYS2['radius_errors'], bins=np.arange(0, 100, 0.6), fill=False, edgecolor='k', linewidth=3,
         histtype='step', label='GLORYS2V4', weights=hist_weights(GLORYS2['radius_errors']))
plt.hist(GLORYS12['radius_errors'], bins=np.arange(0, 100, 0.6), fill=False, edgecolor='g', linewidth=3,
         histtype='step', label='GLORYS12V1', weights=hist_weights(GLORYS12['radius_errors']))
plt.gca().set(xlim=(0, 100), ylim=(0, 0.022), xlabel='Radius Error (km)')
plt.legend()
plt.grid()

plt.subplot(223)
plt.hist(GLORYS2['amplitude_errors'], bins=np.arange(0, 35, 0.133), fill=False, edgecolor='k', linewidth=3,
         histtype='step', label='GLORYS2V4', weights=hist_weights(GLORYS2['amplitude_errors']))
plt.hist(GLORYS12['amplitude_errors'], bins=np.arange(0, 35, 0.133), fill=False, edgecolor='g', linewidth=3,
         histtype='step', label='GLORYS12V1', weights=hist_weights(GLORYS12['amplitude_errors']))
plt.gca().set(xlim=(0, 35), ylim=(0, 0.04), xlabel='Amplitude Error (cm)')
plt.legend()
plt.grid()

plt.subplot(224)
plt.hist(GLORYS2['distance_errors'], bins=np.arange(0, 150, 1), fill=False, edgecolor='k', linewidth=3,
         histtype='step', label='GLORYS2V4', weights=hist_weights(GLORYS2['distance_errors']))
plt.hist(GLORYS12['distance_errors'], bins=np.arange(0, 150, 1), fill=False, edgecolor='g', linewidth=3,
         histtype='step', label='GLORYS12V1', weights=hist_weights(GLORYS12['distance_errors']))
plt.gca().set(xlim=(0, 150), ylim=(0, 0.013), xlabel='Distance Error (km)')
plt.legend()
plt.grid()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig('shape_and_size_representation.png', dpi=300)
plt.show()

metrics = {
    'Cost': ('costs', '', 0, 6),
    'Radius Error': ('radius_errors', 'km', 0, 100),
    'Amplitude Error': ('amplitude_errors', 'cm', 0, 35),
    'Distance Error': ('distance_errors', 'km', 0, 150),
}

with open('matched_error_statistics_FIXED.txt', 'w') as f:
    f.write('# Mean and standard deviation of matched-eddy errors\n')
    f.write('# Reference product: SWOT/MIOST\n')
    f.write('# These statistics are computed only over matched eddies/hits.\n')
    f.write('# Note: histograms may hide values outside the displayed x-axis limits; they are still included in the means below.\n\n')

    print('\nMean and Standard Deviation - matched eddies only')
    for metric_name, (key, unit, xmin, xmax) in metrics.items():
        m2, s2 = mean_std(GLORYS2[key])
        m12, s12 = mean_std(GLORYS12[key])
        outside2 = count_outside(GLORYS2[key], xmin, xmax)
        outside12 = count_outside(GLORYS12[key], xmin, xmax)

        line1 = f'{metric_name} ({unit})' if unit else metric_name
        line2 = f'GLORYS2V4:  {m2:.6f} ± {s2:.6f} | n={len(GLORYS2[key])} | outside plotted range={outside2}'
        line3 = f'GLORYS12V1: {m12:.6f} ± {s12:.6f} | n={len(GLORYS12[key])} | outside plotted range={outside12}'

        print(line1)
        print(line2)
        print(line3)
        print('')

        f.write(line1 + '\n')
        f.write(line2 + '\n')
        f.write(line3 + '\n\n')

print('Statistiche esportate in: matched_error_statistics_FIXED.txt')

