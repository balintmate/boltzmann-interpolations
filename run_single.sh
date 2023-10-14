#!/bin/bash
source env/bin/activate
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd src
export PYTHONPATH=${PWD}:${PYTHONPATH}

cd ../experiments
python3 exp.py "$@" 1>out 2>error