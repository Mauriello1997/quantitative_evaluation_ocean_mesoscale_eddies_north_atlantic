#!/usr/bin/env python

import os
import re
from datetime import datetime
from netCDF4 import Dataset
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("Agg")

from py_eddy_tracker.dataset.grid import RegularGridDataset


dataset = "GLORYS12V1 remapped 1/8°"


# ==========================================================
# Periodo da processare
# ==========================================================
START_DATE = datetime(2023, 8, 1)
END_DATE   = datetime(2025, 5, 1)


# ==========================================================
# Area Nord Atlantico
#
# File remappati:
# longitude = 2880, da -180 a 179.9167, passo 0.125°
# latitude  = 1440, da -80 a 90, passo circa 0.118°/0.125° a seconda della griglia
#
# Target plot:
# lon: 280–330°E = -80°W / -30°W
# lat: 29.87–59.87°N
# ==========================================================
indexs = dict(
    longitude=slice(799, 1200),   # circa -80°W to -30°W
    latitude=slice(958, 1199),    # circa 29.9°N to 59.9°N
)


# ==========================================================
# Cartelle
# ==========================================================
input_dir = "/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES"
output_dir = "/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/eddy_output_north_atlantic_remapped_swot_grid/"

os.makedirs(output_dir, exist_ok=True)


# ==========================================================
# Selezione dei file remap*_depth0.nc
# Esempio:
# remap_mercatorglorys12v1_gl12_mean_20250405_R20250409_depth0.nc
# ==========================================================
nc_files = sorted(
    f for f in os.listdir(input_dir)
    if f.startswith("remap_")
    and f.endswith("_depth0.nc")
)


print("Numero file remap*_depth0.nc trovati:", len(nc_files))


processed = 0
skipped_date = 0
skipped_existing = 0
errors = 0


for nc_file in nc_files:

    # Cerca la data principale nel nome file.
    # Nel nome:
    # remap_mercatorglorys12v1_gl12_mean_20250405_R20250409_depth0.nc
    # prende 20250405, non la data R20250409.
    match = re.search(r"mean_(\d{8})_R\d{8}_depth0\.nc$", nc_file)

    if not match:
        print(f"❌ Data non trovata nel nome file: {nc_file}")
        errors += 1
        continue

    date_str = match.group(1)
    current_date = datetime.strptime(date_str, "%Y%m%d")

    # Limita il periodo richiesto
    if current_date < START_DATE or current_date > END_DATE:
        skipped_date += 1
        continue

    file_path = os.path.join(input_dir, nc_file)

    out_a_path = os.path.join(output_dir, f"Anticyclonic_GLORYS12V1_{date_str}.nc")
    out_c_path = os.path.join(output_dir, f"Cyclonic_GLORYS12V1_{date_str}.nc")
    out_png_path = os.path.join(output_dir, f"eddies_GLORYS12V1_{date_str}.png")

    # Se entrambi i file NetCDF esistono già, salta il giorno
    if os.path.exists(out_a_path) and os.path.exists(out_c_path):
        print(f"⏭️  Già processato: {date_str}")
        skipped_existing += 1
        continue

    print("Processing:", file_path)

    try:
        # Carica il dataset
        h = RegularGridDataset(
            file_path,
            "longitude",
            "latitude",
            indexs=indexs,
        )

        # Filtro sulla variabile zos
        h.bessel_high_filter("zos", 500, order=3)

        # Eddy detection
        a, c = h.eddy_identification(
            "zos",
            "uo",
            "vo",
            current_date,
            0.002,
            pixel_limit=(5, 2000),
            shape_error=55,
        )

        # ==================================================
        # Plot
        # ==================================================
        fig = plt.figure(figsize=(15, 7))
        ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])

        ax.set_title(
            f"Eddies detected -- {date_str} -- {dataset} -- "
            "Anticyclonic (red) and Cyclonic (blue)"
        )

        ax.set_ylim(29.87, 59.87)
        ax.set_xlim(280, 330)
        ax.set_aspect("equal")
        ax.grid()

        a.display(
            ax,
            color="r",
            linewidth=0.5,
            label="Anticyclonic ({nb_obs} eddies)",
            ref=-10,
        )

        c.display(
            ax,
            color="b",
            linewidth=0.5,
            label="Cyclonic ({nb_obs} eddies)",
            ref=-10,
        )

        ax.legend()

        fig.savefig(out_png_path, dpi=300)
        plt.close(fig)

        # ==================================================
        # Salvataggio NetCDF
        # ==================================================
        with Dataset(out_a_path, "w") as out_a:
            a.to_netcdf(out_a)

        with Dataset(out_c_path, "w") as out_c:
            c.to_netcdf(out_c)

        print(f"✔ Completato {date_str}")

        processed += 1

    except Exception as e:
        print(f"❌ Errore durante il processing di {nc_file}: {e}")
        errors += 1
        continue


print("====================================")
print("Detection terminata")
print("Periodo richiesto:", START_DATE.strftime("%Y-%m-%d"), "→", END_DATE.strftime("%Y-%m-%d"))
print("File processati:", processed)
print("File saltati perché fuori periodo:", skipped_date)
print("File saltati perché già esistenti:", skipped_existing)
print("Errori:", errors)
print("Output directory:", output_dir)
print("====================================")
