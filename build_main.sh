#!/bin/bash -e
#
# This scripts builds java binaries.
#
set -o pipefail
curr_dir=$PWD

if [ "$MAVEN_OFFILNE" = "1" ] ; then
  offlineBuildFlag=" --offline "
fi

# XmlIterator fails if the locale/encoding is not UTF8
export LANG=en_US.UTF-8
export LC_ALL=$LANG

# 1. build a fat jar and wrapper scripts
JAVA_SRC_PREF=java
pushd $JAVA_SRC_PREF
mvn $offlineBuildFlag -U clean package appassembler:assemble
popd

target_dir="$curr_dir/flexneuart/resources/jars"
mkdir -p "$target_dir"
cp ${JAVA_SRC_PREF}/target/FlexNeuART*fatjar.jar "$target_dir"

# 2. create a patched version of the wrapper scripts so that they use an installed version of the jar
target_dir="scripts/bin"
# Being a bit paranoid to avoid removal of the wrong directory
[ -z "$target_dir" ] && { echo "Bug: Empty destination directory!" ; exit ; }
rm -rf "$target_dir"
mkdir -p "$target_dir"
for f in `ls ${JAVA_SRC_PREF}/target/appassembler/bin|grep -v .bat$` ; do
  target_file="$target_dir/$f"
  echo "Patching script $f (target $target_file)"
  cat ${JAVA_SRC_PREF}/target/appassembler/bin/$f | \
  sed 's/^REPO=$/REPO=$(python -c "from flexneuart import get_jars_location ; print(get_jars_location())") || { echo "import error, did you install flexneuart library?" ; exit 1 ; }/'      > $target_file
  chmod +x "$target_file" 
done

# 3. pack all the scripts
mkdir -p flexneuart/resources/extra
cd scripts
rm -f scripts.tar.gz

# Note the h it tells us to follow symlinks!

tar cvfzh scripts.tar.gz \
  exper/sample_exper_desc/one_feat.model \
  data_convert/ir_datasets/sample_configs/* \
  data/stopwords.txt \
  bin/* \
  lib/RankLib.jar \
  `find  . -name '*.sh'` \
  `find  . -name '*.py'`
mv scripts.tar.gz ../flexneuart/resources/extra

echo "All is done!"

