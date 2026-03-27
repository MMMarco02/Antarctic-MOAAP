# This script merges the IVT magnitude files for the years 2001-2020 into a single file for each data type (HCLIM, MetUM, RACMO2). 
# It sets the varibale name to "ivt" and saves the combined file as "ivt_computed_combined_2001_2020_nested_{data_type}.nc".
# It represents the second step in the IVT percentile calculation process, following the computation of IVT magnitude.

data_type="RACMO2" # HCLIM or MetUM or RACMO2
BASE="PolarRES/${data_type}/nested"

# We quote the BASE to handle spaces, but leave the {range} unquoted so Bash expands it
cdo -setname,ivt -cat ${BASE}/{2001..2020}/ANT_${data_type}_ivt_mag_*_nested.nc ivt_computed_combined_2001_2020_nested_${data_type}.nc