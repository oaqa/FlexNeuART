INDEX_METHOD_PREFIX="sw-graph"
NMSLIB_METHOD="sw-graph"
NMSLIB_HEADER_NAME="header_exper1_bm25_symm_text_hash_payload"

PARAMS=( \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=5" \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=10" \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=25" \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=50" \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=100" \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=250" \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=500" \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=1000" \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=1500" \
"NN=50,efConstruction=100,useProxyDist=1" "efSearch=2000" \
)

if [ "$collect" = "squad" ] ; then
NMSLIB_FIELDS="text,text_alias1"
else
NMSLIB_FIELDS="text"
fi

