# Streetlearn protocol buffers BUILD file.
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

load("//:third_party/clif.bzl", "pyclif_proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache License 2.0

exports_files(["LICENSE"])

proto_library(
    name = "streetlearn_proto",
    srcs = ["streetlearn.proto"],
)

cc_proto_library(
    name = "streetlearn_cc_pb2",
    deps = [":streetlearn_proto"],
)

py_proto_library(
    name = "streetlearn_py_pb2",
    srcs = ["streetlearn.proto"],
    default_runtime = "@com_google_protobuf//:protobuf_python",
    protoc = "@com_google_protobuf//:protoc",
)

pyclif_proto_library(
    name = "streetlearn_pyclif",
    proto_lib = ":streetlearn_proto",
)
