# Streetlearn BUILD file.
#
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

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache v2

exports_files(["LICENSE"])

py_binary(
    name = "experiment",
    srcs = ["experiment.py"],
    deps = [
        "//streetlearn/proto:streetlearn_py_pb2",
        "//streetlearn/python/agents:agents_py",
        "//streetlearn/python/environment:environment_py",
        "//streetlearn/python/scalable_agent:py_process",
        "//streetlearn/python/scalable_agent:vtrace",
    ],
)
