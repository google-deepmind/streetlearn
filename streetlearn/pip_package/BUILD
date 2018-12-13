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

licenses(["notice"])  # Apache License 2.0

filegroup(
    name = "licenses",
    data = [
        "//:LICENSE",
        # TODO(user): add all licenses
    ],
)

# This is a fake target to populate build_pip_package.runfiles with the right
# dependency files (including __init__.py's).
py_binary(
    name = "setup",
    srcs = ["setup.py"],
    legacy_create_init = 1,
    deps = [
        "//streetlearn:streetlearn_engine_py",
        "//streetlearn/python/environment:environment_py",
    ],
)

sh_binary(
    name = "build_pip_package",
    srcs = ["build_pip_package.sh"],
    data = [
        "setup.py",
        ":licenses",
        ":setup",
    ],
)
