from py_eddy_tracker.observations.tracking import TrackEddiesObservations
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import scipy.stats as stats

# -------------------------------------------------
# Dataset
# -------------------------------------------------

dataset = "DUACS"

data_sources = {
    "DUACS": {
        "Anticyclonic": "/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/eddy_tracking/Anticyclonic_4days.nc",
        "Cyclonic": "/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/eddy_tracking/Cyclonic_4days.nc"
    }
}

# -------------------------------------------------
# Plot settings (paper style, more visible)
# -------------------------------------------------

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 16,
    "axes.labelsize": 22,
    "axes.labelweight": "bold",
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "axes.linewidth": 1.8,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 18
})

fig, ax = plt.subplots(figsize=(11, 7))

# colore unico per dataset
dataset_color = "#87CEEB"

# marker diversi
markers = {
    "Anticyclonic": "o",
    "Cyclonic": "^"
}

# -------------------------------------------------
# liste globali per statistiche
# -------------------------------------------------

all_amplitudes = []
all_radii = []

# -------------------------------------------------
# Loop dati
# -------------------------------------------------

for source_name, vortices in data_sources.items():
    for vortex_type, file_path in vortices.items():

        eddies = TrackEddiesObservations.load_file(file_path)

        amplitude = eddies.obs["amplitude"] * 100
        radius = eddies.obs["radius_e"] / 1000

        all_amplitudes.extend(amplitude)
        all_radii.extend(radius)

        ax.scatter(
            amplitude,
            radius,
            s=32,
            alpha=0.55,
            marker=markers[vortex_type],
            color=dataset_color,
            edgecolors="black",
            linewidths=0.35,
            label=vortex_type
        )

        print(f"{source_name} {vortex_type}: {len(amplitude)} eddies")

# -------------------------------------------------
# Statistiche
# -------------------------------------------------

amp_mean = np.mean(all_amplitudes)
rad_mean = np.mean(all_radii)

amp_std = np.std(all_amplitudes)
rad_std = np.std(all_radii)

amp_sem = stats.sem(all_amplitudes)
rad_sem = stats.sem(all_radii)

print(f"Amplitude mean: {amp_mean:.2f} ± {amp_std:.2f} cm")
print(f"Radius mean: {rad_mean:.2f} ± {rad_std:.2f} km")

# -------------------------------------------------
# Regressione lineare
# -------------------------------------------------

slope, intercept, r_value, p_value, std_err = linregress(all_amplitudes, all_radii)

x_vals = np.linspace(0, 205, 300)
y_vals = slope * x_vals + intercept

#fit_label = f"Linear fit (R² = {r_value**2:.3f})"
fit_label = (
    f"Linear fit: y = {slope:.2f}x + {intercept:.2f}\n"
    f"R² = {r_value**2:.3f}"
)

ax.plot(
    x_vals,
    y_vals,
    color="black",
    linewidth=2.4,
    linestyle="--",
    label=fit_label
)

print(f"Fit: Radius = {slope:.2f} × Amplitude + {intercept:.2f}")
print(f"R² = {r_value**2:.3f}")

# -------------------------------------------------
# legenda senza duplicati
# -------------------------------------------------

handles, labels = ax.get_legend_handles_labels()
unique = dict(zip(labels, handles))

legend = ax.legend(
    unique.values(),
    unique.keys(),
    loc="lower right",
    frameon=True,
    framealpha=1.0,
    fancybox=True,
    borderpad=0.8,
    handlelength=1.8,
    markerscale=1.2
)

legend.get_frame().set_linewidth(1.4)
legend.get_frame().set_edgecolor("black")

# -------------------------------------------------
# axis settings
# -------------------------------------------------

ax.set_xlabel("Amplitude (cm)", labelpad=12, fontweight="bold")
ax.set_ylabel("Radius (km)", labelpad=12, fontweight="bold")

ax.set_xlim(0, 150)
ax.set_ylim(0, 245)

ax.set_xticks(np.arange(0, 205, 25))
ax.set_yticks(np.arange(0, 245, 25))

ax.set_title(f"DUACS 1/8°", pad=10, fontweight="bold")

ax.tick_params(
    axis="both",
    which="major",
    direction="out",
    length=7,
    width=1.8,
    labelsize=18
)

# rende più marcati anche i numeri dei tick
for tick in ax.get_xticklabels() + ax.get_yticklabels():
    tick.set_fontweight("bold")

# bordi più visibili
for spine in ax.spines.values():
    spine.set_linewidth(1.8)

# griglia molto leggera
ax.grid(alpha=0.15, linewidth=0.8)

plt.tight_layout()

# -------------------------------------------------
# Save figure
# -------------------------------------------------

output_file = f"Amplitude_vs_Radius_4days_DUACS.png"

plt.savefig(
    output_file,
    dpi=600,
    bbox_inches="tight"
)

print(f"Figure saved as: {output_file}")

plt.show()
plt.close()
