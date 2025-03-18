#! /bin/bash
#
# Download Lupa Freedman 2021 opacity data
FILE="lupa_freedman_2021"
DATA_DIR=$1/

# Download link
DROPBOX_LINK="https://zenodo.org/api/records/7542068/files-archive"

# Read expected SHA256 from the file
EXPECTED_SHA256=$(grep "$FILE" ${DATA_DIR}checksums.txt | awk '{print $1}')

# Check if the SHA256 has been found
if [ -z "$EXPECTED_SHA256" ]; then
    echo "Cannot find SHA256 for $FILE in ${DATA_DIR}checksums.txt"
    exit 1
fi

# Check if the folder exists
if [ -d "${DATA_DIR}$FILE" ]; then
    echo "$FILE exists. Checking SHA256..."
    
    # Check the operating system and calculate SHA256 of the existing file
    SHA256=$(checksumdir -a sha256 ${DATA_DIR}$FILE)

    # Compare the calculated SHA256 with the expected one
    if [ "$SHA256" == "$EXPECTED_SHA256" ]; then
        echo "SHA256 is correct. No need to download the file."
    else
        echo "SHA256 is not correct. Need to download the file..."
    fi
else
    echo "$FILE does not exist. Need to download the file..."
fi

# Download and extract the file if it does not exist or its SHA256 is incorrect
if [ ! -d "${DATA_DIR}$FILE" ] || [ "$SHA256" != "$EXPECTED_SHA256" ]; then
    wget -q --show-progress -O ${DATA_DIR}${FILE}.tar.gz $DROPBOX_LINK
    if tar -xzvf ${DATA_DIR}${FILE}.tar.gz -C ${DATA_DIR}; then
        echo "Successfully extracted ${FILE}.tar.gz"
        rm ${DATA_DIR}${FILE}.tar.gz
        echo "Removed ${FILE}.tar.gz"
    else
        echo "Failed to extract ${FILE}.tar.gz"
        exit 1
    fi
fi
