# This bash script computes the 95th percentile of IVT for the period 2001-2020. 
# It first computes the monthly minimum and maximum values, then uses these bounds to calculate the 95th percentile. 
# This is the final step for computing IVT 95th percentile.

# Configuration
data_type="RACMO2" # HCLIM or MetUM or RACMO2
BASE_DIR="ERA5"
COMBINED_FILE="ivt_computed_combined_2001_2020_nested_${data_type}.nc"
MIN_FILE="ivt_computed_ymonmin_${data_type}.nc"
MAX_FILE="ivt_computed_ymonmax_${data_type}.nc"
FINAL_FILE="ivt_computed_p95_12mon_nested_${data_type}.nc"
N_THREADS=4  # Adjust based on your Betzy allocation

echo "------------------------------------------------"
echo "Starting IVT 95th Percentile Computation"
echo "Period: 2001 - 2020"
echo "Time: $(date)"
echo "------------------------------------------------"


# Step 1: Compute Bounds (Min/Max)
echo "[1/2] Computing monthly minimums and maximums (bounds)..."
cdo -P $N_THREADS ymonmin $COMBINED_FILE $MIN_FILE
echo "Monthly minimums completed."
cdo -P $N_THREADS ymonmax $COMBINED_FILE $MAX_FILE
echo "Monthly maximums completed."

# Step 2: Compute 95th Percentile
echo "[2/2] Computing 95th percentile (this may take a while)..."
cdo -P $N_THREADS ymonpctl,95 -setname,ivt $COMBINED_FILE $MIN_FILE $MAX_FILE $FINAL_FILE
echo "Done. Final file created: $FINAL_FILE"

echo "------------------------------------------------"
echo "Process Finished Successfully at $(date)"
echo "------------------------------------------------"