#!/bin/bash
DATA_PREF="data/squad"
sort $DATA_PREF/focus_words.txt|awk 'BEGIN{pw="";pc=0}{{if ($1 != pw) if(pw != "") {print pc" "pw;pc=0}};pw=$1;pc=pc+1}END{print pc" "pw}'|less|sort -n | awk '{if ($1>=5) print $0}' > $DATA_PREF/freq_focus_words.stat
cut -d \  -f 2 $DATA_PREF/freq_focus_words.stat > $DATA_PREF/freq_focus_words.txt 
