# Bazel build for OpenCV.

cc_library(
    name = "opencv",
    srcs = glob(["lib/libopencv*.so"]),
    hdrs = glob(["include/opencv/**/*.hpp", "include/opencv/**/*.h"]),
    includes = ["include"],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
