#!/bin/bash

sudo mkdir /kaggle
sudo mkdir /kaggle/working
sudo chmod 777 /kaggle/working

# These are one-time downloads so that they're ready when the notebooks try to use them. Otherwise, one run may incur the latency of downloading them and another not.
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"

mkdir stats