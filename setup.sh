#!/usr/bin/env bash
mkdir saved_models
mkdir logs
mkdir data/murad-data
gdown <URL HERE>
mv murad-dataset-rev.zip data/murad-data
unzip data/murad-data/murad-dataset-rev.zip
rm data/murad-data/murad-dataset-rev.zip
virtualenv venv -p python3
source venv/bin/activate
pip install numpy==1.18.5