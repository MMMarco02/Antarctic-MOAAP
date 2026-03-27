# This bash script computes seasonal 99th and 99.9th percentiles for ERA5 precipitation data, applying a threshold to filter out low values. 
# It processes each year individually for speed, merges them into seasonal stacks, and then computes the desired percentiles. 
# The final percentile files are saved in the specified output directory.

#!/bin/bash
INDIR="ERA5/"
OUTDIR="ERA5/pr_percentile_computation/seasonal_percentiles_thr0.1_ERA5"
TMPDIR="${OUTDIR}/tmp"
mkdir -p "$OUTDIR" "$TMPDIR"
echo "Directory created. Starting percentile computation for ERA5 precipitation data..."

SEASONS=("DJF" "MAM" "JJA" "SON")
PCTS=("99" "99.9")
THRESHOLD=0.0001 # 0.1 mm/h threshold to filter out low precipitation values in the interior
export OMP_NUM_THREADS=1
CDO_PAR=8   # or 4

for SEASON in "${SEASONS[@]}"; do
    echo "--- Processing Season: $SEASON ---"
    
    YEAR_FILES=()
    # 1. Process each year individually (FAST & PARALLELIZABLE)
    for YEAR in {2001..2020}; do
        FILE="${INDIR}/${YEAR}/remapped/ANT_pr_${YEAR}_remapped.nc"
        YEAR_OUT="${TMPDIR}/pr_${SEASON}_${YEAR}_processed.nc"
        
        if [ ! -f "$YEAR_OUT" ]; then
            echo "Processing $YEAR..."
            # -selseason FIRST to reduce data, then threshold
            cdo -P 4 -z zip_1 -setrtoc,-inf,$THRESHOLD,0 -selseason,${SEASON} "$FILE" "$YEAR_OUT"
        fi
        YEAR_FILES+=("$YEAR_OUT")
    done

    # 2. Merge the small processed years into one seasonal stack
    SEASON_STACK="${TMPDIR}/precip_${SEASON}_total_stack.nc"
    echo "Merging yearly files into $SEASON_STACK..."
    cdo -P $CDO_PAR -mergetime "${YEAR_FILES[@]}" "$SEASON_STACK"

    # 3. Compute Min/Max
    echo "Computing Min/Max..."
    cdo -P $CDO_PAR timmin "$SEASON_STACK" "${TMPDIR}/${SEASON}_min.nc"
    cdo -P $CDO_PAR timmax "$SEASON_STACK" "${TMPDIR}/${SEASON}_max.nc"

    # 4. Compute Percentiles
    for P in "${PCTS[@]}"; do
        echo "Computing Percentile $P..."
        cdo -P $CDO_PAR -L -f nc4 -z zip_1 timpctl,$P "$SEASON_STACK" "${TMPDIR}/${SEASON}_min.nc" "${TMPDIR}/${SEASON}_max.nc" "${OUTDIR}/ANT_ERA5_pr_${SEASON}_p${P}.nc"
    done
    
    # Cleanup
    rm "${YEAR_FILES[@]}" "$SEASON_STACK" "${TMPDIR}/${SEASON}_min.nc" "${TMPDIR}/${SEASON}_max.nc"
done

echo " All done! Percentile files are in $OUTDIR"