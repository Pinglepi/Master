#!/bin/bash
start=`date +%s`
# Array of species to process
species=("General" "Bacillales" "Corynebacteriales" "Burkholderiales" "Lactobacillales" "Enterobacterales" "Pfam_1" "Pfam_2" "Pfam_3" "Pfam_4" "Pfam_5")
#  
#  
# Loop through each species
for spec in "${species[@]}"; do
    echo "Starting processing for $spec"
    python Train_models.py "$spec"
    echo "Finished processing for $spec"
done

end=`date +%s`
runtime=$((end-start))
echo "Total time taken: $runtime seconds"