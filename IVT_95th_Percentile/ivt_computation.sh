# This bash script computes the IVT magnitude from the ivte and ivtn components for each year and 
# saves the output as a new NetCDF file. It checks for the existence of input files before running the CDO command to avoid errors.
# It represent the first step for computing IVT 95th percentile, used in AR detection.

data_type="RACMO2" # HCLIM or MetUM or RACMO2
BASE_DIR="PolarRES/${data_type}/nested"

for year in {2001..2020}; do
    echo "---------------------------"
    echo "Processing Year: $year"
    
    YEAR_DIR="${BASE_DIR}/${year}"
    
    E_FILE="${YEAR_DIR}/ANT_${data_type}_ivte_computed_${year}_nested.nc"
    N_FILE="${YEAR_DIR}/ANT_${data_type}_ivtn_computed_${year}_nested.nc"
    # Changed name slightly to be safe
    OUT_FILE="${YEAR_DIR}/ANT_${data_type}_ivt_mag_${year}_nested.nc" 
    
    # Check if input files actually exist before running CDO
    if [[ -f "$E_FILE" && -f "$N_FILE" ]]; then
        cdo -sqrt -add -sqr "$E_FILE" -sqr "$N_FILE" "$OUT_FILE"
        echo "Successfully created $OUT_FILE"
    else
        echo "SKIPPING $year: One or both input files missing."
        [[ ! -f "$E_FILE" ]] && echo "Missing: $E_FILE"
        [[ ! -f "$N_FILE" ]] && echo "Missing: $N_FILE"
    fi
done