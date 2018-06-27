#!/usr/bin/env bash
LF_VERSION="0.0.3"
declare -a PY_VERSIONS=("2.7")
declare -a PLATFORMS=("linux-64" "win-64")

for PY_VERSION in "${PY_VERSIONS[@]}"
do
  conda-build -c=conda-forge --python=${PY_VERSION} ~/code/laminarflow/conda/recipe

  for PLATFORM in "${PLATFORMS[@]}"
  do
    conda convert --platform=${PLATFORM} -o=/tmp/conda-bld ~/miniconda2/conda-bld/osx-64/laminarflow-${LF_VERSION}-py27_0.tar.bz2
    anaconda upload /tmp/conda-bld/${PLATFORM}/laminarflow-${LF_VERSION}-py27_0.tar.bz2
  done

done
