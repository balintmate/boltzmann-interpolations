#!/bin/bash
source env/bin/activate
export PYTHONPATH=${PWD}:${PYTHONPATH}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd experiments
python3 exp.py "$@" 1>out 2>error
tmux kill-session