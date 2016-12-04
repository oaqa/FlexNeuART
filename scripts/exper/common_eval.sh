
QRELS="output/$collect/${TEST_PART}/$QREL_FILE"

rm -f "${REPORT_DIR}/out_*"

for oneN in $NTEST_LIST ; do
  echo "======================================"
  echo "N=$oneN"
  echo "======================================"
  REPORT_PREF="${REPORT_DIR}/out_${oneN}"

  scripts/exper/eval_output.py "$QRELS"  "${TREC_RUN_DIR}/run_${oneN}" "$REPORT_PREF" "$oneN"
  check "eval_output.py"
done

echo "Deleting trec runs from the directory: ${TREC_RUN_DIR}"
rm ${TREC_RUN_DIR}/*
# There should be at least one run, so, if rm fails, it fails because files can't be deleted
check "rm ${TREC_RUN_DIR}/*" 
echo "Bzipping trec_eval output in the directory: ${REPORT_DIR}"
bzip2 ${REPORT_DIR}/*.trec_eval
check "bzip2 "${REPORT_DIR}/*.trec_eval""
echo "Bzipping gdeval output in the directory: ${REPORT_DIR}"
bzip2 ${REPORT_DIR}/*.gdeval
check "bzip2 "${REPORT_DIR}/*.gdeval""
