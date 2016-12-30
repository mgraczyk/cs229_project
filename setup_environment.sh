#!/bin/bash

# Get beautiful soup for html parsing.
pip3 install lxml
pip3 install beautifulsoup4

# Ubuntu needs a special python3 bs4 package
if [ -f /etc/debian_version ]; then
  sudo apt-get install python3-bs4
fi


# Get an example data object for testing
mkdir -p ./testdata
wget https://3f12a5f8d5dc66fdb22bb6e1d4d68ebe530af91c.googledrive.com/host/0B-2un9FOvIMCS2d6bjMyTTBXY2s/2014-01-02.tar.xz \
    -O ./testdata/2014-01-02.tar.xz

# We use ipython for interactive data manipulation.
# MATLAB is not good for parsing HTML
python3 -m pip install --upgrade ipython sklearn scipy numpy
