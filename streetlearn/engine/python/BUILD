# Streetlearn clif bindings BUILD file.
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

load("//:third_party/clif.bzl", "py_clif_cc")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

# Wraps the C++ StreetlearnEngine class for use from Python.
py_clif_cc(
    name = "streetlearn_engine",
    srcs = ["streetlearn_engine.clif"],
    clif_deps = [
        ":metadata_cache",
        ":color",
    ],
    deps = [
        "//streetlearn/engine:streetlearn_engine",
        "//streetlearn/proto:streetlearn_pyclif",
        "//third_party/absl/python:optional",
        "//third_party/absl/python/numpy:span_clif_lib",
    ],
)

# Wraps the C++ MetaData class for use from Python.
py_clif_cc(
    name = "metadata_cache",
    srcs = ["metadata_cache.clif"],
    deps = [
        "//streetlearn/engine:metadata_cache",
        "//streetlearn/proto:streetlearn_pyclif",
    ],
)

# Wraps the C++ Color class for use from Python.
py_clif_cc(
    name = "color",
    srcs = ["color.clif"],
    deps = [
        "//streetlearn/engine:color",
    ],
)

# Wraps the C++ test data for use from Python.
py_clif_cc(
    name = "test_dataset",
    testonly = 1,
    srcs = ["test_dataset.clif"],
    deps = [
        "//streetlearn/engine:test_dataset",
    ],
)

py_test(
    name = "streetlearn_engine_test",
    srcs = ["streetlearn_engine_test.py"],
    deps = [
        ":metadata_cache",
        ":streetlearn_engine",
        ":test_dataset",
        "//streetlearn/proto:streetlearn_py_pb2",
    ],
)
