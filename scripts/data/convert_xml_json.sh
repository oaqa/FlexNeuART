#!/bin/bash -e

outDir=$1

if [ "$outDir" = "" ] ; then
  echo "Specify the output directory, e.g., output/manner"
  exit 1
fi
if [ ! -d "$outDir" ] ; then
  echo "Not a directory: $outDir"
  exit 1
fi

oldFilesArr=(SolrQuestionFile.txt SolrAnswerFile.txt)
newFilesArr=(QuestionFields.jsonl AnswerFields.jsonl)

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
