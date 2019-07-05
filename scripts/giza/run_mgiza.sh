#!/bin/bash
. scripts/giza/run_common.sh

# This script produces only Model 1 alignment

num_threads=0

export BIN_DIR="$giza_dir/mgizapp/bin/"
echo "M-giza bin dir: $BIN_DIR"
rm -f *.vcb *.snt *.cooc *.gizacfg  *.classes output.*
"$BIN_DIR/plain2snt" source target
check plain2snt

"$BIN_DIR/snt2cooc" source_target.cooc source.vcb target.vcb source_target.snt 
check snt2cooc

"$BIN_DIR/mgiza" -s source.vcb -t target.vcb -o output -outputpath . -c source_target.snt -coocurrencefile source_target.cooc   -m1 $num_iter -m2 0 -mh 0 -m3 0 -m4 0 -m5 0 -m6 0 -ncpus $num_threads -t1 1
check mgiza

