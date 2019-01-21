INDEX_METHOD_PREFIX="napp"
NMSLIB_METHOD="napp_qa1"
NMSLIB_HEADER_NAME="header_bm25_text"
NMSLIB_FIELDS="text"
FIELD_CODE_PIVOT="text_field"

PIVOT_FILE_PARAM="pivotFile=nmslib/$collect/pivots/pivots_${FIELD_CODE_PIVOT}_maxTermQty50K_pivotTermQty1000"

PARAMS=( \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=13" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=14" \

"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=10" \
"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=11" \
"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=12" \
"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=5" \
"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=6" \
"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=7" \
"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=8" \
"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=9" \

)
