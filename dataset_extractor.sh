#!/bin/bash

# Directory containing the zip files
zip_dir="Datasets"
extr_dir="Datasets_Extracted"

mkdir -p "$extr_dir"
# Change to the zip directory
cd "$zip_dir" || exit

# Loop through the zip files
for zip_file in *.zip; do
  if [ "$zip_file" = "dias_datasets.zip" ]; then
    # Perform a different action
    unzip -q "$zip_file" -d "../$extr_dir"
    for f in "../$extr_dir/dias-datasets/*"; do
      mv -f $f "../$extr_dir/"
    done
    rm "../$extr_dir/copier.sh"
    rm -rf "../$extr_dir/dias-datasets"
  else
    # Extract the <ABC> and <xyz> parts from the filename
    file_name="${zip_file%.*}"
    if [[ ! $zip_file =~ .*__.*\.zip ]]; then
      echo "Skipped: $zip_file (Invalid name)"
      continue
    fi

    ABC_part="${file_name%%__*}"
    xyz_part="${file_name#*__}"

    # Create the directory structure
    mkdir -p "../$extr_dir/$ABC_part/$xyz_part"

    # Extract the zip file into the directory
    unzip -q "$zip_file" -d "../$extr_dir/$ABC_part/$xyz_part"

    #echo "Extracted $zip_file to ../$extr_dir/$ABC_part/$xyz_part"
  fi
done

echo "Extracted datasets and moved to the $extr_dir directory"
