# Directory to start the search
BASE_DIR="./notebooks"

# Find all directories 2 levels deep
SUBDIRS=$(find "$BASE_DIR" -mindepth 2 -maxdepth 2 -type d)

# Check each subdirectory for the 'input' directory
for dir in $SUBDIRS; do
  if ! [ -d "$dir/input" ]; then
    echo "input/ was not found in $dir. This means that the dataset was not copied successfully."
    exit 1
  fi
done