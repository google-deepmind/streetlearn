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

licenses(["notice"])  # Apache License 2.0

cc_library(
    name = "math",
    hdrs = [
        "math_util.h",
        "vector.h",
    ],
    deps = [
        ":logging",
        "@com_google_absl//absl/base:core_headers",
    ],
)

cc_library(
    name = "rtree_helper",
    srcs = ["rtree_helper.cc"],
    hdrs = ["rtree_helper.h"],
    copts = ["-fexceptions"],
    features = ["-use_header_modules"],
    deps = [
        "@com_google_absl//absl/memory",
        "@com_google_s2geometry//:s2",
    ],
)

cc_library(
    name = "image",
    hdrs = ["image.h"],
    deps = [
        ":logging",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "bitmap_util",
    srcs = ["bitmap_util.cc"],
    hdrs = ["bitmap_util.h"],
)

cc_library(
    name = "color",
    hdrs = [
        "color.h",
    ],
)

cc_library(
    name = "cairo_util",
    hdrs = ["cairo_util.h"],
    deps = [
        "@cairo",
    ],
)

cc_library(
    name = "logging",
    hdrs = ["logging.h"],
)

cc_library(
    name = "dataset",
    hdrs = ["dataset.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "dataset_factory",
    srcs = ["dataset_factory.cc"],
    hdrs = ["dataset_factory.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
        ":leveldb_dataset",
        ":logging",
        "@com_google_absl//absl/strings",
    ],
)

cc_library(
    name = "leveldb_dataset",
    srcs = ["leveldb_dataset.cc"],
    hdrs = ["leveldb_dataset.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":dataset",
        ":logging",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_leveldb//:leveldb",
    ],
)

cc_library(
    name = "streetlearn_engine",
    srcs = ["streetlearn_engine.cc"],
    hdrs = ["streetlearn_engine.h"],
    deps = [
        ":bitmap_util",
        ":dataset",
        ":dataset_factory",
        ":graph_renderer",
        ":logging",
        ":math",
        ":metadata_cache",
        ":pano_graph",
        ":pano_graph_node",
        ":pano_renderer",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/hash",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pano_graph",
    srcs = ["pano_graph.cc"],
    hdrs = ["pano_graph.h"],
    deps = [
        ":dataset",
        ":logging",
        ":math",
        ":metadata_cache",
        ":node_cache",
        ":pano_calculations",
        ":pano_graph_node",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/synchronization",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
    ],
)

cc_library(
    name = "pano_graph_node",
    srcs = ["pano_graph_node.cc"],
    hdrs = ["pano_graph_node.h"],
    deps = [
        ":image",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/types:span",
        "@opencv",
    ],
)

cc_library(
    name = "pano_projection",
    srcs = ["pano_projection.cc"],
    hdrs = ["pano_projection.h"],
    deps = [
        ":image",
        ":math",
        "@com_google_absl//absl/memory",
        "@opencv",
    ],
)

cc_library(
    name = "pano_renderer",
    srcs = ["pano_renderer.cc"],
    hdrs = ["pano_renderer.h"],
    deps = [
        ":color",
        ":logging",
        ":math",
        ":pano_graph_node",
        ":pano_projection",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@cairo",
    ],
)

cc_library(
    name = "pano_calculations",
    hdrs = ["pano_calculations.h"],
    deps = [
        ":pano_graph_node",
    ],
)

cc_library(
    name = "metadata_cache",
    srcs = ["metadata_cache.cc"],
    hdrs = ["metadata_cache.h"],
    deps = [
        ":dataset",
        ":logging",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/memory",
    ],
)

cc_library(
    name = "node_cache",
    srcs = ["node_cache.cc"],
    hdrs = ["node_cache.h"],
    deps = [
        ":pano_fetcher",
        ":pano_graph_node",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "pano_fetcher",
    srcs = ["pano_fetcher.cc"],
    hdrs = ["pano_fetcher.h"],
    deps = [
        ":dataset",
        ":pano_graph_node",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
    ],
)

cc_library(
    name = "graph_region_mapper",
    srcs = ["graph_region_mapper.cc"],
    hdrs = ["graph_region_mapper.h"],
    deps = [
        ":logging",
        ":math",
        "@com_google_s2geometry//:s2",
    ],
)

cc_library(
    name = "graph_image_cache",
    srcs = ["graph_image_cache.cc"],
    hdrs = ["graph_image_cache.h"],
    deps = [
        ":cairo_util",
        ":color",
        ":graph_region_mapper",
        ":image",
        ":logging",
        ":math",
        ":pano_graph",
        "@cairo",
        "@com_google_absl//absl/container:node_hash_map",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_s2geometry//:s2",
    ],
)

cc_library(
    name = "graph_renderer",
    srcs = ["graph_renderer.cc"],
    hdrs = ["graph_renderer.h"],
    deps = [
        ":cairo_util",
        ":color",
        ":graph_image_cache",
        ":image",
        ":logging",
        ":math",
        ":metadata_cache",
        ":pano_graph",
        ":rtree_helper",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@cairo",
        "@com_google_absl//absl/container:node_hash_set",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_absl//absl/types:span",
        "@com_google_s2geometry//:s2",
    ],
)

cc_library(
    name = "test_utils",
    testonly = 1,
    srcs = ["test_utils.cc"],
    hdrs = ["test_utils.h"],
    defines = ["STREETLEARN_SUPPRESS_COMMANDLINE_FLAGS"],
    deps = [
        ":image",
        ":logging",
        ":pano_graph_node",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/time",
        "@com_google_googletest//:gtest",
        "@opencv",
    ],
)

cc_library(
    name = "test_dataset",
    testonly = 1,
    srcs = ["test_dataset.cc"],
    hdrs = ["test_dataset.h"],
    deps = [
        ":image",
        ":leveldb_dataset",
        ":pano_graph_node",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_leveldb//:leveldb",
        "@opencv",
    ],
)

cc_test(
    name = "math_util_test",
    size = "medium",
    srcs = ["math_util_test.cc"],
    deps = [
        ":math",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "rtree_helper_test",
    size = "small",
    srcs = ["rtree_helper_test.cc"],
    deps = [
        ":rtree_helper",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_s2geometry//:s2",
    ],
)

cc_test(
    name = "image_test",
    size = "small",
    srcs = ["image_test.cc"],
    deps = [
        ":image",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "leveldb_dataset_test",
    size = "small",
    srcs = ["leveldb_dataset_test.cc"],
    deps = [
        ":leveldb_dataset",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/memory",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_leveldb//:leveldb",
    ],
)

cc_test(
    name = "bitmap_util_test",
    size = "small",
    srcs = ["bitmap_util_test.cc"],
    deps = [
        ":bitmap_util",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "streetlearn_engine_test",
    srcs = ["streetlearn_engine_test.cc"],
    data = [
        "test_data/graph_test_observer.png",
    ],
    deps = [
        ":dataset_factory",
        ":math",
        ":pano_calculations",
        ":streetlearn_engine",
        ":test_dataset",
        ":test_utils",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pano_projection_test",
    size = "medium",
    srcs = ["pano_projection_test.cc"],
    deps = [
        ":pano_projection",
        ":test_dataset",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@opencv",
    ],
)

cc_test(
    name = "metadata_cache_test",
    size = "small",
    srcs = ["metadata_cache_test.cc"],
    deps = [
        ":dataset_factory",
        ":metadata_cache",
        ":test_dataset",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "node_cache_test",
    size = "small",
    srcs = ["node_cache_test.cc"],
    deps = [
        ":dataset_factory",
        ":node_cache",
        ":test_dataset",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:flat_hash_map",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pano_fetcher_test",
    size = "small",
    srcs = ["pano_fetcher_test.cc"],
    deps = [
        ":dataset_factory",
        ":pano_fetcher",
        ":pano_graph_node",
        ":test_dataset",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/synchronization",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pano_renderer_test",
    srcs = ["pano_renderer_test.cc"],
    deps = [
        ":pano_graph",
        ":pano_renderer",
        ":test_dataset",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pano_calculations_test",
    size = "small",
    srcs = ["pano_calculations_test.cc"],
    deps = [
        ":pano_calculations",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pano_graph_test",
    size = "small",
    srcs = ["pano_graph_test.cc"],
    deps = [
        ":dataset_factory",
        ":pano_graph",
        ":test_dataset",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "pano_graph_node_test",
    size = "small",
    srcs = ["pano_graph_node_test.cc"],
    deps = [
        ":pano_graph_node",
        ":test_dataset",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "graph_region_mapper_test",
    size = "small",
    srcs = ["graph_region_mapper_test.cc"],
    deps = [
        ":graph_region_mapper",
        ":math",
        ":test_dataset",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_s2geometry//:s2",
    ],
)

cc_test(
    name = "graph_image_cache_test",
    size = "small",
    srcs = ["graph_image_cache_test.cc"],
    data = [
        "test_data/image_cache_test.png",
        "test_data/image_cache_test_zoomed.png",
    ],
    deps = [
        ":cairo_util",
        ":dataset_factory",
        ":graph_image_cache",
        ":math",
        ":pano_graph",
        ":test_dataset",
        ":test_utils",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/memory",
        "@com_google_absl//absl/strings",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
        "@com_google_s2geometry//:s2",
    ],
)

cc_test(
    name = "graph_renderer_test",
    size = "small",
    srcs = ["graph_renderer_test.cc"],
    copts = ["-fexceptions"],
    data = [
        "test_data/graph_test.png",
        "test_data/graph_test_observer.png",
        "test_data/graph_test_zoom2.png",
        "test_data/graph_test_zoom4.png",
        "test_data/graph_test_zoom8.png",
    ],
    features = ["-use_header_modules"],
    deps = [
        ":color",
        ":dataset_factory",
        ":graph_image_cache",
        ":graph_renderer",
        ":math",
        ":node_cache",
        ":pano_graph",
        ":test_dataset",
        ":test_utils",
        "//streetlearn/proto:streetlearn_cc_pb2",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/types:optional",
        "@com_google_googletest//:gtest",
        "@com_google_googletest//:gtest_main",
    ],
)

cc_test(
    name = "cairo_util_test",
    srcs = ["cairo_util_test.cc"],
    deps = [
        ":cairo_util",
        "@cairo",
        "@com_google_googletest//:gtest_main",
    ],
)
