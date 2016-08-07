#!/bin/bash
. scripts/giza/run_common.sh

# This script produces only Model 1 alignment

export BIN_DIR="$giza_dir/GIZA++-v2/"
echo "Giza bin dir: $BIN_DIR"
rm -f *.vcb *.snt *.cooc *.gizacfg  *.classes output.*
rm -f *.vcb *.snt *.cooc *.gizacfg  *.classes output.*
"$BIN_DIR/plain2snt.out" source target
check plain2snt.out

