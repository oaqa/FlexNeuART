#/bin/bash
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
      -embed_files)
        embed_files_param=$OPT
        ;;
      -horder_files)
        horder_files_param=$OPT
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

part=${POS_ARGS[1]}
if [ "$part" = "" ] ; then
  echo "Specify part (2d positional arg): e.g., dev1, dev2, train, test"
  exit 1
fi

cand_type=${POS_ARGS[2]}
if [ "$cand_type" = "" ] ; then
  echo "Specify the type of candidate provider (3d positional arg): e.g., nmslib, lucene"
  exit 1
fi

URI=${POS_ARGS[3]}
if [ "$URI" = "" ] ; then
  echo "Specify the index location: Lucene index dir, nmslib TCP/IP address, etc... (4th positional arg)"
  exit 1
fi

n=${POS_ARGS[4]}
if [ "$n" = "" ] ; then
  echo "Specify coma-separated numbers of candidate records (5th positional arg)"
  exit
fi

extr_type_final=${POS_ARGS[5]}
if [ "$extr_type_final" = "" ] ; then
  echo "Specify the type of the final feature extractor (6th positional arg) "
  exit 1
fi

out_dir=${POS_ARGS[6]}
if [ "$out_dir" = "" ] ; then
  echo "Specify the output directory (7th positional arg)"
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
echo "==============================================="
echo "PARAMETER REVIEW"
echo "==============================================="
echo "Thread qty:           $thread_qty"
echo "max_num_query_param:  $max_num_query_param"
echo "embed_files_param:    $embed_files_param"
echo "horder_files_param:   $horder_files_param"
echo "extr_type_interm:     $extr_type_interm_param"
echo "model_interm_param:   $model_interm_param"
echo "cand_qty_param:       $cand_qty_param"
echo "Positional arguments: ${POS_ARGS[*]}"
echo "==============================================="
echo "OUTPUT FILE PREFIX:"
echo "$out_pref"
echo "FULL OUTPUT FILE PREFIX:"
echo "$full_out_pref"
echo "==============================================="

scripts/query/run_multhread_feat.sh \
-u "$URI" \
-cand_prov $cand_type \
-q output/$collect/$part/SolrQuestionFile.txt -qrel_file output/$collect/$part/qrels.txt "$max_num_query_param" \
-n "$n" \
-f "$full_out_pref" \
-memindex_dir "memfwdindex/$collect" \
-giza_root_dir tran/$collect/ -giza_iter_qty 5 \
-extr_type_final $extr_type_final \
"$extr_type_interm_param" "$model_interm_param" "$cand_qty_param" \
-thread_qty $thread_qty  \
-embed_dir WordEmbeddings/$collect  "$embed_files_param" "$horder_files_param" \
2>&1 | tee "${full_out_pref}_${n}.log"
if [ "${PIPESTATUS[0]}" != "0" ] ; then
  exit 1
fi
