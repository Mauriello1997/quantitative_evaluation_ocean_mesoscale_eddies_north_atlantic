#!/bin/bash
# Remapping dei file cmems_mod_glo_phy-all_my_0.25deg_P1D-m-2024*_depth0.nc
# con griglia aviso_grid.txt → output in /remapped

set -euo pipefail

# === Attiva ambiente ===
module load conda
conda activate copernicus_py39

# === Percorsi ===
GRID="aviso_grid.txt"   # deve essere nel cwd o usa un path assoluto
IN_DIR="/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES"
OUT_DIR="/ec/res4/scratch/ita6648/GLORYS2V4/ALL_FILES/remapped"
mkdir -p "$OUT_DIR"

cd "$IN_DIR"

echo "=== INIZIO REMAP ==="
date

# === Loop sui file ===
for file in cmems_mod_glo_phy-all_my_0.25deg_P1D-m-20231*_depth0.nc; do
  [ -e "$file" ] || continue
  base=$(basename "$file")
  output="$OUT_DIR/remap_${base}"

  if [ -f "$output" ]; then
    echo "⏭️  Il file $output esiste già, salto."
    continue
  fi

  echo "👉 Remapping $file → $output"
  cdo -L -P 4 remapbil,"$GRID" "$file" "$output"
done

echo "=== FINE REMAP ==="
date

