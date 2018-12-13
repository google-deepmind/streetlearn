sh_binary(
    name = "pyclif",
    srcs = ["CLIF_PATH/clif/bin/pyclif"],
    visibility = ["//visibility:public"],
)

sh_binary(
    name = "proto",
    srcs = ["CLIF_PATH/clif/bin/pyclif_proto"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "cpp_runtime",
    srcs = glob(
        ["CLIF_PATH/clif/python/*.cc"],
        exclude = ["CLIF_PATH/clif/python/*_test.cc"],
    ),
    hdrs = glob(["CLIF_PATH/clif/python/*.h"]),
    includes = ["CLIF_PATH"],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_python//:python_headers",
        "@local_config_python//:numpy_headers",
        "@com_google_protobuf//:protobuf",
    ],
)
