INDEX_METHOD_PREFIX="napp"
NMSLIB_METHOD="napp_qa1"
NMSLIB_HEADER_NAME="header_exper1_hash_payload"
FIELD_CODE_PIVOT="2field"
NMSLIB_FIELDS="text,text_alias1"

PIVOT_FILE_PARAM="pivotFile=nmslib/$collect/pivots/pivots_${FIELD_CODE_PIVOT}_maxTermQty50K_pivotTermQty1000"

PARAMS=( \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=11" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=13" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=15" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=15" \
"numPivot=8000,numPivotIndex=200,$PIVOT_FILE_PARAM" "numPivotSearch=19" \
"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=7" \
"numPivot=8000,numPivotIndex=100,$PIVOT_FILE_PARAM" "numPivotSearch=9" \
"numPivot=8000,numPivotIndex=50,$PIVOT_FILE_PARAM" "numPivotSearch=4" \
"numPivot=8000,numPivotIndex=50,$PIVOT_FILE_PARAM" "numPivotSearch=5" \
)
