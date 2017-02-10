if [ "$collect" = "manner" ] ; then
  export NUM_RET_LIST="10,15,17,36,72"
  export N_TRAIN=15
elif [ "$collect" = "compr" ] ; then
  #NUM_RET_LIST="50,100,200,400,600,1000,1500,2500"
  # Not much value beoynd 400
  #NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,100,200,400"
  export NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
  export N_TRAIN=15
elif [ "$collect" = "stackoverflow" ] ; then
  export NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
  export N_TRAIN=15
elif [ "$collect" = "squad" ] ; then
  export NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100"
  #export NUM_RET_LIST="1,2,3,4,5,10,15,20,25,30,35,45,50,60,70,80,90,100,250,500,750,1000"
  export N_TRAIN=20
elif [ "$collect" = "squad" ] ; then
  export NUM_RET_LIST="10,20,30,50,60,70,80,90,100,200,300,400,500"
  export N_TRAIN=20
else
  echo "Unsupported collection: $collect"
  exit 1
fi
