#! /bin/bash

python setup.py develop

jupyter notebook --ip=0.0.0.0 --port=9999 --allow-root