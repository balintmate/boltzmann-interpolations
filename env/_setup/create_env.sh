#!/bin/bash
cd "${0%/*}"
cd ../..
mv ./env/_setup .
rm -rf env
virtualenv env
mv _setup/ env/_setup/
source env/bin/activate
pip3 install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip3 install -r env/_setup/requirements.txt
deactivate
echo '/*' > ./env/.gitignore
echo !/_setup >> ./env/.gitignore