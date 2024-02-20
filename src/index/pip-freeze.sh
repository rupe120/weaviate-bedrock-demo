#!/bin/bash

set -e

pyenv install 3.11 -s
pyenv local 3.11
python -m venv .venv
source ./.venv/bin/activate
python.exe -m pip install --upgrade pip
pip install -r requirements.txt
pip list --format=freeze > requirements-frozen.txt
deactivate