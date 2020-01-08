#/bin/bash -e
source scripts/common_proc.sh
source scripts/config.sh

checkVarNonEmpty "FAKE_RUN_ID"
checkVarNonEmpty "SEP_DEBUG_LINE"

POS_ARGS=()

thread_qty=1

while [ $# -ne 0 ] ; do
  echo $1|grep "^-" >/dev/null 
  if [ $? = 0 ] ; then
    OPT_NAME="$1"
    OPT_VALUE="$2"
    OPT="$1 $2"
    if [ "$OPT_VALUE" = "" ] ; then  
      echo "Option $OPT_NAME requires an argument." >&2
      exit 1
    fi
    shift 2
    case $OPT_NAME in
      -thread_qty)
        thread_qty=$OPT_VALUE 
        ;;
      -max_num_query)
        max_num_query_param=$OPT
        ;;
      -extr_type_interm)
        extr_type_interm_param=$OPT
        extr_type_interm=$OPT_VALUE
        ;;
      -model_interm)
        model_interm_param=$OPT
        model_interm=$OPT_VALUE
        ;;
      -cand_qty)
        cand_qty_param=$OPT
        cand_qty=$OPT_VALUE
        ;;
      -out_pref)
        out_pref=$OPT_VALUE
        ;;
      -query_cache_file)
        query_cache_file_param=$OPT
        ;;
      *)
        echo "Invalid option: $OPT_NAME" >&2
        exit 1
        ;;
    esac
  else
    POS_ARGS=(${POS_ARGS[*]} $1)
    shift 1
  fi
done
if [ "$extr_type_interm_param" != "" ] ; then
  if [ "$model_interm_param" = "" -o "$cand_qty_param" = "" ] ; then
    echo "Option -extr_type_interm requires options: -model_interm and -cand_qty" >&2
    exit 1
  fi

  extr_type_interm_code="${extr_type_interm}:${model_interm}:cand_qty=${cand_qty}_"
fi

collect=${POS_ARGS[0]}
if [ "$collect" = "" ] ; then
  echo "Specify a sub-collection (1st positional arg): e.g., manner, compr"
  exit 1
fi

qrel_file=${POS_ARGS[1]}
if [ "$qrel_file" = "" ] ; then
  echo "Specify QREL file (2d positional arg): e.g., qrels_all_graded.txt, qrels_all_graded_same_score.txt"
  exit 1
fi

part=${POS_ARGS[2]}
if [ "$part" = "" ] ; then
  echo "Specify part (3d positional arg): e.g., dev1, dev2, train, test"
  exit 1
fi

cand_type=${POS_ARGS[3]}
if [ "$cand_type" = "" ] ; then
  echo "Specify the type of candidate provider (4th positional arg): e.g., nmslib, lucene"
  exit 1
fi

URI=${POS_ARGS[4]}
if [ "$URI" = "" ] ; then
  echo "Specify the index location: Lucene index dir, nmslib TCP/IP address, etc... (5th positional arg)"
  exit 1
fi

n=${POS_ARGS[5]}
if [ "$n" = "" ] ; then
  echo "Specify coma-separated numbers of candidate records (6th positional arg)"
  exit
fi

extr_type_final=${POS_ARGS[6]}
if [ "$extr_type_final" = "" ] ; then
  echo "Specify the type of the final feature extractor (7th positional arg) "
  exit 1
fi

out_dir=${POS_ARGS[7]}
if [ "$out_dir" = "" ] ; then
  echo "Specify the output directory (8th positional arg)"
  exit 1
fi

if [ ! -d "$out_dir" ] ; then
  echo "Not a directory: $out_dir"
  exit 1
fi

if [ "$out_pref" = "" ] ; then
  out_pref="out_${collect}_${part}_${extr_type_interm_code}${extr_type_final}"
fi
full_out_pref="$out_dir/$out_pref"

source scripts/set_common_resource_vars.sh

checkVarNonEmpty "inputDataDir" # set by set_common_resource_vars.sh
checkVarNonEmpty "commonResourceParams"  # set by set_common_resource_vars.sh

retVal=""
getIndexQueryDataInfo "$inputDataDir"
queryFileName=${retVal[3]}
if [ "$queryFileName" = "" ] ; then
  echo "Cannot guess the type of data, perhaps, your data uses different naming conventions."
  exit 1
fi

echo "$SEP_DEBUG_LINE"

echo "Parameter and settings review"

echo "$SEP_DEBUG_LINE"

echo "Data directory:          $inputDataDir"
echo "Data file name:          $queryFileName"

echo "$SEP_DEBUG_LINE"

echo "Thread qty:              $thread_qty"
echo "max_num_query_param:     $max_num_query_param"
echo "extr_type_interm:        $extr_type_interm_param"
echo "model_interm_param:      $model_interm_param"
echo "cand_qty_param:          $cand_qty_param"
echo "query_cache_file_param:  $query_cache_file_param"
echo "Positional arguments:    ${POS_ARGS[*]}"



echo "OUTPUT FILE PREFIX:"
echo "$out_pref"
echo "FULL OUTPUT FILE PREFIX:"
echo "$full_out_pref"

echo "$SEP_DEBUG_LINE"

# Do it only after argument parsing
set -eo pipefail

scripts/query/run_multhread_feat.sh \
-u "$URI" \
-run_id "$FAKE_RUN_ID" \
-cand_prov $cand_type \
-q "$inputDataDir/$part/$queryFileName" \
-qrel_file "$inputDataDir/$part/$qrel_file" \
 "$max_num_query_param" \
-n "$n" \
-f "$full_out_pref" \
$commonResourceParams \
-extr_type_final $extr_type_final \
$extr_type_interm_param $model_interm_param $cand_qty_param \
-thread_qty $thread_qty  \
$query_cache_file_param \
2>&1 | tee "${full_out_pref}_${n}.log"

