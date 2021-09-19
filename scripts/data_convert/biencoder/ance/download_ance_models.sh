#!/bin/bash -e
# Downloading and unpacking ANCE checkpoint models
dstDir="$1"

[ "$dstDir" != "" ] || { echo "Specify destination directory (1st arg)" ; exit 1 ; }

[ -d "$dstDir" ] || { echo "Not a directory: $dstDir (1st arg)" ; exit 1 ; }

source ./common_proc.sh

cd "$dstDir"

storePref="https://webdatamltrainingdiag842.blob.core.windows.net/semistructstore/OpenSource"


# Document MaxP
#wget "$storePref/Document_ANCE_MaxP_Checkpoint.zip"
#unzip Document_ANCE_MaxP_Checkpoint.zip
#chmod -R ogu+w .
#mv 'Document ANCE(MaxP) Checkpoint' Document_ANCE_MaxP_Checkpoint

# Unpacked model names/directories should match values
# in the dictionary DATA_TYPE_PATHS in flexneuart.data_convert.ance.data

# Document FirstP
wget "$storePref/Document_ANCE_FirstP_Checkpoint.zip"
unzip Document_ANCE_FirstP_Checkpoint.zip
chmod -R ogu+w .
mv 'Document ANCE(FirstP) Checkpoint' Document_ANCE_FirstP_Checkpoint

# Passage FirstP
wget  "$storePref/Passage_ANCE_FirstP_Checkpoint.zip"
unzip Passage_ANCE_FirstP_Checkpoint.zip
chmod -R ogu+w .
mv 'Passage ANCE(FirstP) Checkpoint' Passage_ANCE_FirstP_Checkpoint



# NQ
wget $storePref/nq.cp
# Trivia QA
wget $storePref/trivia.cp


