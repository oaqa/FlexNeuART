# Here we have a variable $collect available.
# If needed we can make a collection-specific choice

export N_TRAIN=100 # >=20 for NDCG@20 and ERR@20

#export NUM_RET_LIST="10,25,50,75,100,250,500,1000"
if [ "$collect" = "squad" -o "$collect" = "clueweb09" ] ; then
  export NUM_RET_LIST="10,50,100,250,500,1000"
else
  export NUM_RET_LIST="10,50,100,250"
fi
