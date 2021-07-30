#/bin/bash
set -eux
./regression.py --check
./regression.py --drop
./regression.py --bn
