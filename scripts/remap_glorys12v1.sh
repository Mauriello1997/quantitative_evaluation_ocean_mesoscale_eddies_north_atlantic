#!/bin/bash
# Rimappa i file mercatorglorys12v1_gl12_mean_20231*_depth0.nc su griglia aviso_grid.txt
# Output -> /remapped
# Eseguibile direttamente con ./remap_mercator_9.sh o con nohup

set -euo pipefail

# === Attiva ambiente conda ===
module load conda
conda activate copernicus_py39

# === Percorsi ===
GRID="aviso_grid.txt"   # usa path assoluto se non è nel cwd
IN_DIR="/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES"
OUT_DIR="/ec/res4/scratch/ita6648/GLORYS12V1/ALL_FILES/remapped"
mkdir -p "$OUT_DIR"

cd "$IN_DIR"

echo "=== INIZIO REMAP MERCATOR 20231* ==="
date

shopt -s nullglob
count=0

for file in mercatorglorys12v1_gl12_mean_20231*_depth0.nc; do
  [ -e "$file" ] || continue
  base=$(basename "$file")
  output="$OUT_DIR/remap_${base}"

  if [ -f "$output" ]; then
    echo "⏭️  Esiste già: $output — salto."
    continue
  fi

  echo "👉 Remapping $file → $output"
  cdo -L -P 4 remapbil,"$GRID" "$file" "$output"
  ((count++)) || true
done

echo "=== FINE REMAP MERCATOR 20231* (totale: $count file) ==="
date

