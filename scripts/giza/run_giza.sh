#!/bin/bash
. scripts/giza/run_common.sh

# This script produces only Model 1 alignment

export BIN_DIR="$giza_dir/GIZA++-v2/"
echo "Giza bin dir: $BIN_DIR"
rm -f *.vcb *.snt *.cooc *.gizacfg  *.classes output.*
"$BIN_DIR/plain2snt.out" source target
check plain2snt.out

"$BIN_DIR/snt2cooc.out" source.vcb target.vcb source_target.snt > source_target.cooc 
check snt2cooc

"$BIN_DIR/GIZA++" -s source.vcb -t target.vcb -o output -outputpath . -c source_target.snt -coocurrencefile source_target.cooc -m1 $num_iter -m2 0 -mh 0 -m3 0 -m4 0 -m5 0 -m6 0 -t1 1
check mgiza

