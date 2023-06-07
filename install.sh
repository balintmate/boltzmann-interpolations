#!/bin/bash
rm -rf env
virtualenv env
source env/bin/activate
pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install -r requirements.txt
deactivate