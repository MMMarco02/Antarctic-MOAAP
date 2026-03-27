# This bash script performs the remapping of PolarRES model output files to the commond PolarCORDEX grid using CDO. 
# It iterates over specified years and variables, applies the chosen remapping method, and saves the remapped files in a structured directory. 
# This represents the second step in the nesting process, since embedding into the ERA5 background is in the embed_model_data.py script.


# --- Paths ---
# Point this to where your '2001' folder lives
model_name="RACMO2" # HCLIM or MetUM or RACMO2
SRC_ROOT="PolarRES/${model_name}/raw"
GRID_FILE="PolarRES_WP3_Antarctic_domain.nc"

# Updated list based on your 'ls' output (THERE IS NO 750hPa LEVEL)
#VAR_LIST=("ua200" "ua300" "ua400" "ua500" "ua600" "ua700" "ua850" "ua925" "ua1000" \
#          "va200" "va300" "va400" "va500" "va600" "va700" "va850" "va925" "va1000" \
#          "ta850" "zg300" "zg400" "zg500" "zg600" "zg700" "zg850" "zg925" "zg1000" \
#          "hus300" "hus400" "hus500" "hus600" "hus700" "hus850" "hus925" "hus1000" "psl")
#VAR_LIST=("ua300" "ua850" \
#          "va300" "va850"  \
#          "ta850" \
#          "zg500" \
#          "hus850" \
#          "ivte_computed" "ivtn_computed" )
VAR_LIST=("pr") # hourly precipitation
#VAR_LIST=("va300")

# Check cdo available
command -v cdo >/dev/null 2>&1 || { echo "cdo not found in PATH"; exit 1; }

for YEAR in {2002..2020}; do
    ORIG_DIR="${SRC_ROOT}/${YEAR}"
    DST_DIR="PolarRES/${model_name}/remapped/${YEAR}"

    # Ensure output directory exists
    mkdir -p "${DST_DIR}"

    echo ">>> Starting remapping for year ${YEAR}..."
    echo ">>> Logging to ${DST_DIR}/remap_${YEAR}.log"

    for VAR in "${VAR_LIST[@]}"; do
        echo "--- Processing ${VAR} ---"

        # Find the file: Matches variable name at start, ends in .nc, IGNORES .raw.nc
        # Example: zg500_ANT-12_ERA5...nc
        INFILE=$(find "${ORIG_DIR}" -maxdepth 1 -type f -name "${VAR}_*.nc" ! -name "*.raw.nc" -print -quit)

        if [ -z "${INFILE}" ]; then
            echo "Skipping ${VAR}: No matching file found in ${ORIG_DIR}"
            continue
        fi

        OUTFILE="${DST_DIR}/ANT_${model_name}_${VAR}_${YEAR}_remapped.nc"

        # --- Select Remapping Method (manually choose one) ---
        #REMAP_OP="remapbil"
        #REMAP_OP="remapdis"
        REMAP_OP="remapcon" 
        #echo "Method: Distance-weighted (Continuous variable)"
        #echo "Method: Bilinear interpolation (Continuous variable)"
        echo "Method: Conservative remapping (Best for precipitation, conserves totals)"
        
        start_time=$(date +%s)
        
        # Execute CDO with the selected operator
        cdo ${REMAP_OP},"${GRID_FILE}" "${INFILE}" "${OUTFILE}"
        
        rc=$?
        end_time=$(date +%s)
        elapsed=$(( end_time - start_time ))

        if [ $rc -eq 0 ]; then
            echo "✓ Success: ${VAR} (${elapsed}s)"
        else
            echo "✗ FAILED: ${VAR} (CDO exit code ${rc})"
        fi
    done
done

echo ">>> Remapping complete for ${YEAR}."