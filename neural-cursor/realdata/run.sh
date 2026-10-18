#!/bin/sh
# Run a realdata module with no cwd or PYTHONPATH on sys.path (-E -P), only the repo root added.
# Usage: realdata/run.sh load --data D --cache C   |   realdata/run.sh experiments within --feat F
ROOT=$(cd "$(dirname "$0")/.." && pwd)
MOD=$1; shift
exec python3 -E -P -c "import sys, runpy; sys.path.insert(0, '$ROOT'); sys.argv = ['$MOD'] + sys.argv[1:]; runpy.run_module('realdata.$MOD', run_name='__main__')" "$@"
