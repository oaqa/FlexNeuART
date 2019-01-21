INDEX_METHOD_PREFIX="napp"
NMSLIB_METHOD="napp_qa1"
NMSLIB_HEADER_NAME="header_bm25_text"
FIELD_CODE_PIVOT="text_field"
NMSLIB_FIELDS="text"

PIVOT_FILE_PARAM="pivotFile=nmslib/$collect/pivots/pivots_${FIELD_CODE_PIVOT}_maxTermQty50K_pivotTermQty1000"

PARAMS=( \
"numPivot=8000,numPivotIndex=238,$PIVOT_FILE_PARAM" "numPivotSearch=12" \
"numPivot=8000,numPivotIndex=228,$PIVOT_FILE_PARAM" "numPivotSearch=13" \
\
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=11" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=12" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=13" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=14" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=15" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=17" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=19" \
)
