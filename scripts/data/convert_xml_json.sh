#!/bin/bash -e
source scripts/config.sh

outDir=$1

if [ "$outDir" = "" ] ; then
  echo "Specify the data input directory, e.g., squad/input_data"
  exit 1
fi
if [ ! -d "$outDir" ] ; then
  echo "Not a directory: $outDir"
  exit 1
fi

oldFilesArr=(SolrQuestionFile.txt SolrAnswerFile.txt)
newFilesArr=(QuestionFields.jsonl AnswerFields.jsonl)

checkVarNonEmpty "COLLECT_ROOT"
cd "$COLLECT_ROOT"

for subDir in "$outDir"/* ; do
  if [ -d "$subDir" ] ; then
    for i in 0 1 ; do
      oldFile="$subDir"/${oldFilesArr[$i]}
      newFile="$subDir"/${newFilesArr[$i]}
      if [ -f "$oldFile" ] ; then 
        echo "Preparing to do conversion: $oldFile -> $newFile"
        scripts/data/run_convert_xml_json.sh -input "$oldFile" -output "$newFile"
        echo "Deleting old file: $oldFile"
        rm "$oldFile"
      fi
    done
  fi
done
