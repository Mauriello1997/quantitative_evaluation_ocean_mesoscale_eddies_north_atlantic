from py_eddy_tracker.observations.tracking import TrackEddiesObservations
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import os

# =============================================================================
# CONFIGURAZIONE
# =============================================================================

data_sources = {
    "GLORYS12V1": {
        "Anticyclonic": "/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/"
                         "eddy_output_north_atlantic_remapped_swot_grid/eddy_tracking/Anticyclonic_4days.nc",
        "Cyclonic":     "/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/"
                         "eddy_output_north_atlantic_remapped_swot_grid/eddy_tracking/Cyclonic_4days.nc",
    },
    "GLORYS2V4": {
        "Anticyclonic": "/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/"
                         "eddy_output_north_atlantic_remapped_1_8/eddy_tracking/Anticyclonic_4days.nc",
        "Cyclonic":     "/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/"
                         "eddy_output_north_atlantic_remapped_1_8/eddy_tracking/Cyclonic_4days.nc",
    },
    "SWOT MIOST Science": {
        "Anticyclonic": "/ec/res4/scratch/ita6648/miost_science/eddy_output_north_atlantic/"
                         "eddy_tracking/Anticyclonic_4days.nc",
        "Cyclonic":     "/ec/res4/scratch/ita6648/miost_science/eddy_output_north_atlantic/"
                         "eddy_tracking/Cyclonic_4days.nc",
    },
    "DUACS 1/8°": {
        "Anticyclonic": "/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/"
                         "eddy_tracking/Anticyclonic_4days.nc",
        "Cyclonic":     "/ec/res4/scratch/ita6648/cmems/eddy_output_north_atlantic/"
                         "eddy_tracking/Cyclonic_4days.nc",
    },
}

colors = {
    "Anticyclonic": "#e41a1c",
    "Cyclonic": "#377eb8",
}

markers = {
    "Anticyclonic": "o",
    "Cyclonic": "^",
}

MIN_LIFETIME_DAYS = 4

# =============================================================================
# PARAMETRI GRAFICI (PAPER)
# =============================================================================

TITLE_FS = 28
SUBTITLE_FS = 22
LABEL_FS = 22
TICK_FS = 22
LEGEND_FS = 18

plt.rcParams.update({
    "font.family": "serif",
})

# =============================================================================
# FUNZIONI
# =============================================================================

def get_track_amplitude_and_lifetime(eddies, min_lifetime_days=0):
    obs = eddies.obs
    names = obs.dtype.names

    if "n" in names:
        obs_num = np.array(obs["n"])
    elif "observation_number" in names:
        obs_num = np.array(obs["observation_number"])
    else:
        raise RuntimeError("Indice temporale non trovato in obs")

    if "track" not in names:
        raise RuntimeError("Campo 'track' non trovato in obs")

    track_ids = np.array(obs["track"])

    if "amplitude" in names:
        amp_m = np.array(obs["amplitude"])
    elif "amp" in names:
        amp_m = np.array(obs["amp"])
    else:
        raise RuntimeError("Campo amplitude non trovato")

    amp_cm = amp_m * 100.0

    lifetimes = []
    amplitudes = []

    for tid in np.unique(track_ids):
        mask = track_ids == tid
        obs_track = obs_num[mask]
        amp_track = amp_cm[mask]

        lifetime_days = obs_track.max() + 1
        if lifetime_days < min_lifetime_days:
            continue

        lifetimes.append(lifetime_days)
        amplitudes.append(np.mean(amp_track))

    return np.array(lifetimes), np.array(amplitudes)


def sanitize_filename(name):
    for b, r in zip(["/", "\\", " ", "°"], ["_", "_", "_", "deg"]):
        name = name.replace(b, r)
    return name


# =============================================================================
# MAIN
# =============================================================================
def main():

    for dataset_name, vortices in data_sources.items():
        print(f"\n=== Dataset: {dataset_name} ===")

        fig, ax = plt.subplots(figsize=(16, 10))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        all_lifetimes = []
        all_amplitudes = []

        for vortex_type, file_path in vortices.items():
            if not os.path.exists(file_path):
                print(f"  File mancante: {file_path}")
                continue

            eddies = TrackEddiesObservations.load_file(file_path)

            lifetimes, amplitudes = get_track_amplitude_and_lifetime(
                eddies,
                min_lifetime_days=MIN_LIFETIME_DAYS,
            )

            if lifetimes.size == 0:
                continue

            all_lifetimes.append(lifetimes)
            all_amplitudes.append(amplitudes)

            ax.scatter(
                lifetimes,
                amplitudes,
                s=60,
                alpha=0.75,
                marker=markers[vortex_type],
                color=colors[vortex_type],
                edgecolor="black",
                linewidth=0.8,
                label=vortex_type,
            )

        # 🔴 QUESTO DEVE STARE DENTRO IL FOR
        if not all_lifetimes:
            plt.close(fig)
            continue

        # =============================
        # STILE ASSI (PAPER)
        # =============================
        ax.set_xlabel("Lifetime (days)", fontsize=26, fontweight="bold", labelpad=12)
        ax.set_ylabel("Amplitude (cm)", fontsize=26, fontweight="bold", labelpad=12)

        ax.tick_params(axis="both", which="major", labelsize=24, width=1.8, length=8, direction="out")
        ax.tick_params(axis="both", which="minor", width=1.2, length=4, direction="out")

        #ax.xaxis.set_minor_locator(ticker.MultipleLocator(5))
         #ax.yaxis.set_minor_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
        # Tick bold
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # Assi spessi
        for spine in ax.spines.values():
            spine.set_linewidth(1.8)

        # Griglia
        ax.grid(True, which="major", linewidth=0.8, alpha=0.6)
         #ax.grid(True, which="minor", linestyle="--", alpha=0.3)

        # =============================
        # TITOLO + LEGENDA
        # =============================
        ax.set_title(f"{dataset_name}", fontsize=32, fontweight="bold", pad=14)

        ax.legend(
            fontsize=22,
            frameon=True,
            fancybox=True,
            framealpha=1.0,
            edgecolor="black",
            markerscale=1.8
        )

        # =============================
        # SALVATAGGIO
        # =============================
        fig.tight_layout()

        safe_name = sanitize_filename(dataset_name)
        out_name = f"Amplitude_vs_Lifetime_{safe_name}_min{MIN_LIFETIME_DAYS}days_paper.png"

        fig.savefig(out_name, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        
        print(f"  -> Salvato: {out_name}")
if __name__ == "__main__":
    main()
