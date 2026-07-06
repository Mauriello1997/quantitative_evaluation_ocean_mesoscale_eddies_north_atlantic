#!/usr/bin/env python3

from datetime import datetime, timedelta
import os
import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from netCDF4 import Dataset as NetCDFDataset
from py_eddy_tracker import start_logger
from py_eddy_tracker.dataset.grid import RegularGridDataset


# =============================================================================
# LOGGER
# =============================================================================
start_logger().setLevel("DEBUG")


# =============================================================================
# CONFIG
# =============================================================================
DATASET = "SWOT MIOST Science"
dataset = "swot"

input_path_template = (
    "/ec/res4/scratch/ita6648/miost_science/eddy_output_north_atlantic/"
    "dt_global_allsat_phy_l4_{date}_20250826.nc"
)

output_dir = "/ec/res4/scratch/ita6648/miost_science/eddy_output_north_atlantic/"
os.makedirs(output_dir, exist_ok=True)

output_eddy_dir = os.path.join(output_dir, "eddy_detected")
os.makedirs(output_eddy_dir, exist_ok=True)


# =============================================================================
# PARAMETRI DETECTION / AREA
# =============================================================================
indexs = dict(
    longitude=slice(799, 1200),   # stop escluso: indici 799...1199
    latitude=slice(958, 1199),    # stop escluso: indici 958...1198
)

# Ricrea sempre il file temporaneo quando si esegue il codice.
# Evita di riutilizzare un sottoinsieme creato con indici precedenti.
FORCE_RECREATE_TEMP = True

LAT_MIN, LAT_MAX = 30, 60
LON_MIN, LON_MAX = 280, 330

start_date = datetime(2025, 1, 1)
end_date   = datetime(2025, 1, 1)

delta = timedelta(days=1)


# =============================================================================
# PARAMETRI ADT
# =============================================================================
ADT_VMIN = -2.0
ADT_VMAX = 2.0


# =============================================================================
# CREA FILE TEMPORANEO CON LA STESSA AREA DI DETECTION E LONGITUDE 0–360
# =============================================================================
def create_detection_area_file(input_file, current_date):
    """
    Crea un file temporaneo solo per il plot/detection.
    Usa esattamente gli stessi indexs della detection:
        longitude=slice(799,1200)
        latitude=slice(958,1199)

    Poi converte le longitude da -80/-30 a 280/330,
    così ADT ed eddies sono nello stesso sistema di coordinate.
    """

    tmp_file = os.path.join(
        output_dir,
        f"tmp_{dataset}_adt_detection_area_{current_date:%Y%m%d}_lon360.nc"
    )

    if os.path.exists(tmp_file):
        if FORCE_RECREATE_TEMP:
            print(f"Removing old temporary file: {tmp_file}")
            os.remove(tmp_file)
        else:
            print(f"Temporary file already exists: {tmp_file}")
            return tmp_file

    print(f"Creating temporary detection-area file: {tmp_file}")

    with xr.open_dataset(input_file, decode_cf=True) as ds_full:

        print(
            "Full grid dimensions:",
            f"latitude={ds_full.sizes['latitude']},",
            f"longitude={ds_full.sizes['longitude']}"
        )

        # isel usa indici posizionali e il limite finale non è incluso.
        ds = ds_full.isel(
            longitude=indexs["longitude"],
            latitude=indexs["latitude"],
        ).load()

    # Converte le longitudini selezionate da -180/180 a 0/360.
    ds = ds.assign_coords(
        longitude=xr.where(
            ds["longitude"] < 0,
            ds["longitude"] + 360,
            ds["longitude"]
        )
    )

    # Garantisce coordinate crescenti, richieste dalla griglia regolare.
    ds = ds.sortby("longitude").sortby("latitude")

    print(
        "Selected detection area:",
        f"longitude={float(ds.longitude.min()):.4f} ... "
        f"{float(ds.longitude.max()):.4f};",
        f"latitude={float(ds.latitude.min()):.4f} ... "
        f"{float(ds.latitude.max()):.4f}"
    )
    print(
        "Selected grid dimensions:",
        f"latitude={ds.sizes['latitude']},",
        f"longitude={ds.sizes['longitude']}"
    )

    # Tiene solo le variabili necessarie.
    ds = ds[["adt", "ugos", "vgos"]]

    ds.to_netcdf(tmp_file)
    ds.close()

    return tmp_file


# =============================================================================
# PLOT ADT + DETECTED EDDIES
# =============================================================================
def plot_adt_and_eddies(g_plot, a, c, current_date):

    TITLE_FONTSIZE = 22
    TICK_FONTSIZE = 20
    CBAR_LABEL_FONTSIZE = 16
    CBAR_TICK_FONTSIZE = 22
    LEGEND_FONTSIZE = 15

    fig, ax = plt.subplots(figsize=(14, 9))

    # -------------------------------------------------------------------------
    # LIMITI MAPPA IN 0–360
    # -------------------------------------------------------------------------
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_color("black")

    # -------------------------------------------------------------------------
    # TICKS
    # -------------------------------------------------------------------------
    xticks = np.arange(279.6, 331.5, 10)
    yticks = np.arange(30, 61, 10)

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    ax.set_xticklabels(
        [f"{int(360 - x)}°W" for x in xticks],
        fontsize=TICK_FONTSIZE,
        fontweight="bold",
    )

    ax.set_yticklabels(
        [f"{int(y)}°N" for y in yticks],
        fontsize=TICK_FONTSIZE,
        fontweight="bold",
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=TICK_FONTSIZE,
        width=2.2,
        length=8,
        direction="out",
    )

    ax.grid(True, linewidth=0.5, alpha=0.5)

    # -------------------------------------------------------------------------
    # 1. ADT BACKGROUND
    # -------------------------------------------------------------------------
    m = g_plot.display(
        ax,
        "adt",
        vmin=ADT_VMIN,
        vmax=ADT_VMAX,
        cmap="RdBu_r",
    )

    # -------------------------------------------------------------------------
    # 2. DETECTED EDDIES SOPRA ADT
    # Anticyclonic = red
    # Cyclonic = blue
    # -------------------------------------------------------------------------
    a.display(
        ax,
        color="red",
        linewidth=1,
        label="Anticyclonic ({nb_obs})",
        ref=-10,
    )

    c.display(
        ax,
        color="blue",
        linewidth=1,
        label="Cyclonic ({nb_obs})",
        ref=-10,
    )

    # -------------------------------------------------------------------------
    # TITOLO
    # -------------------------------------------------------------------------
    ax.set_title(
        f"{DATASET}",
        fontsize=TITLE_FONTSIZE,
        weight="bold",
        pad=12,
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    # -------------------------------------------------------------------------
    # LEGEND
    # -------------------------------------------------------------------------
    legend = ax.legend(
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        framealpha=1.0,
        loc="upper right",
    )

    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.3)
    legend.get_frame().set_facecolor("white")

    for text in legend.get_texts():
        text.set_fontweight("bold")

    for line in legend.get_lines():
        line.set_linewidth(1.5)

    # -------------------------------------------------------------------------
    # COLORBAR
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # SALVATAGGIO
    # -------------------------------------------------------------------------
    out_png = os.path.join(
        output_dir,
        f"adt_detected_eddies_{dataset}_{current_date:%Y%m%d}.png"
    )

    fig.tight_layout()

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="w",
        edgecolor="black",
    )

    plt.close(fig)

    print(f"Figure saved: {out_png}")


# =============================================================================
# MAIN
# =============================================================================
def main():

    current_date = start_date

    while current_date <= end_date:

        date_str = current_date.strftime("%Y%m%d")
        print("Processing:", date_str)

        input_file = input_path_template.format(date=date_str)

        if not os.path.exists(input_file):
            print(f"File not found: {input_file}")
            current_date += delta
            continue

        # ---------------------------------------------------------------------
        # File con stessa area degli indexs + longitude 0–360
        # ---------------------------------------------------------------------
        detection_area_file = create_detection_area_file(
            input_file,
            current_date
        )

        # ---------------------------------------------------------------------
        # Dataset per ADT background
        # ---------------------------------------------------------------------
        g_plot = RegularGridDataset(
            detection_area_file,
            "longitude",
            "latitude",
        )

        # ---------------------------------------------------------------------
        # Dataset per detection
        # ---------------------------------------------------------------------
        h = RegularGridDataset(
            detection_area_file,
            "longitude",
            "latitude",
        )

        # ---------------------------------------------------------------------
        # EDDY IDENTIFICATION
        # ---------------------------------------------------------------------
        h.bessel_high_filter("adt", 500, order=3)

        a, c = h.eddy_identification(
            "adt",
            "ugos",
            "vgos",
            current_date,
            0.002,
            pixel_limit=(5, 2000),
            shape_error=55,
        )

        print(f"Anticyclonic eddies detected: {len(a)}")
        print(f"Cyclonic eddies detected: {len(c)}")

        # ---------------------------------------------------------------------
        # SALVA EDDIES
        # ---------------------------------------------------------------------
        out_a_path = os.path.join(
            output_eddy_dir,
            f"Anticyclonic_{current_date:%Y%m%d}.nc"
        )

        out_c_path = os.path.join(
            output_eddy_dir,
            f"Cyclonic_{current_date:%Y%m%d}.nc"
        )

        with NetCDFDataset(out_a_path, "w") as out_a:
            a.to_netcdf(out_a)

        with NetCDFDataset(out_c_path, "w") as out_c:
            c.to_netcdf(out_c)

        print(f"Saved: {out_a_path}")
        print(f"Saved: {out_c_path}")

        # ---------------------------------------------------------------------
        # PLOT ADT + EDDIES
        # ---------------------------------------------------------------------
        plot_adt_and_eddies(
            g_plot,
            a,
            c,
            current_date,
        )

        current_date += delta


if __name__ == "__main__":
    main()

