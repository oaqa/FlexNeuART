#!/bin/bash
DATA_ROOT=$1
if [ "$DATA_ROOT" = "" ] ; then
  echo "Specify the data directory root (1st arg)!"
  exit 1
fi

if [ ! -d "$DATA_ROOT" ] ; then
  echo "'$DATA_ROOT' is not a directory (1st arg)!"
  exit 1
fi

collect=$2
if [ "$collect" = "" ] ; then
  echo "Specify sub-collection (2d arg): manner, compr"
  exit 1
fi

function check {
  f="$?"
  name=$1
  if [ "$f" != "0" ] ; then
    echo "**************************************"
    echo "* Failed: $name"
    echo "**************************************"
    exit 1
  fi
}

PIVOT_QTY=10000

if [ ! -d "$DATA_ROOT/nmslib/$collect/pivots/" ] ; then
  mkdir "$DATA_ROOT/nmslib/$collect/pivots/"
  check "mkdir "$DATA_ROOT/nmslib/$collect/pivots/""
fi

#for pivot_term_qty in 100 300 1000 3000 10000 ; do
#  for max_term_qty_K in 25 50 75 100 ; do
for pivot_term_qty in 1000 ; do
  for max_term_qty_K in 50 ; do
    max_term_qty=$((1000*$max_term_qty_K))
    if [ "$collect" = "squad" ] ; then
      pivot_file_name="pivots_2field_maxTermQty${max_term_qty_K}K_pivotTermQty${pivot_term_qty}"
      cmd="scripts/nmslib/gen_pivots_multifield.py -d memfwdindex/$collect -o  $DATA_ROOT/nmslib/$collect/pivots/$pivot_file_name -q $PIVOT_QTY -f text,text_alias1 -m $max_term_qty,$max_term_qty -t $pivot_term_qty,$pivot_term_qty"
      bash -c "$cmd"
      check "$cmd"
    fi

    pivot_file_name="pivots_text_field_maxTermQty${max_term_qty_K}K_pivotTermQty${pivot_term_qty}"
    scripts/nmslib/gen_pivots_multifield.py -d memfwdindex/$collect -o  $DATA_ROOT/nmslib/$collect/pivots/$pivot_file_name -q $PIVOT_QTY -f text -m $max_term_qty -t $pivot_term_qty
    check "scripts/nmslib/gen_pivots_multifield.py -d memfwdindex/$collect -o  $DATA_ROOT/nmslib/$collect/pivots/$pivot_file_name -q $PIVOT_QTY -f text -m $max_term_qty -t $pivot_term_qty"

  done
done

