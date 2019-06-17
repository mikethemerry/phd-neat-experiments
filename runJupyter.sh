#! /bin/bash

cd /src/requirements/phd-neat-experiments

git pull

python setup.py develop

cd ~

jupyter notebook --ip=0.0.0.0 --port=9999 --allow-root