#!/bin/bash

# Copyright 2018 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e
TMPDIR=$(mktemp -d -t tmp.XXXXXXXXXXX)
RUNFILES=bazel-bin/streetlearn/pip_package/build_pip_package.runfiles/org_deepmind_streetlearn

# Build.
bazel build -c opt $COPT_FLAGS streetlearn/pip_package:build_pip_package

# 1: Copy /streetlearn to top level.
cp -R "${RUNFILES}/streetlearn" "${TMPDIR}"

# 2: Copy /*solib* dir to top level.
so_lib_dir=$(ls "$RUNFILES" | grep solib)
if [ -n "${so_lib_dir}" ]; then
  cp -R "${RUNFILES}/${so_lib_dir}" "${TMPDIR}"
fi

cp LICENSE "${TMPDIR}"
cp README.md "${TMPDIR}"
cp streetlearn/pip_package/MANIFEST.in "${TMPDIR}"
cp streetlearn/pip_package/setup.py "${TMPDIR}"

pushd "${TMPDIR}"
rm -f MANIFEST
echo $(date) : "=== Building wheel in ${TMPDIR}"
python setup.py bdist_wheel
popd

if [ $# -gt 0 ]; then
  DEST=$1
  mkdir -p "${DEST}"
  cp "${TMPDIR}/dist"/* "${DEST}"
else
  DEST="${TMPDIR}/dist"
fi

echo "Output wheel is in ${DEST}"
