#!/bin/bash -e

source ./export_nmslib/export_nmslib_common.sh

checkVarNonEmpty "params"

./bin/ExportToNMSLIBDenseSparseFusion $params
