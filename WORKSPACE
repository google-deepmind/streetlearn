workspace(name = "org_deepmind_streetlearn")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//:patched_http_archive.bzl", "patched_http_archive")
load("//:custom_rules.bzl", "local_repository_env")

http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-master",
    urls = ["https://github.com/google/googletest/archive/master.zip"],
)

http_archive(
    name = "com_google_absl",
    sha256 = "2148ee21d88efa81acd5eb1da567adaec9032ce4c26486b6518245e9b148c739",
    strip_prefix = "abseil-cpp-284378a71b32dfb3af4e3661f585e671d1b603a3",
    urls = ["https://github.com/abseil/abseil-cpp/archive/284378a71b32dfb3af4e3661f585e671d1b603a3.zip"],
)

http_archive(
    name = "com_google_protobuf",
    sha256 = "9510dd2afc29e7245e9e884336f848c8a6600a14ae726adb6befdb4f786f0be2",
    strip_prefix = "protobuf-3.6.1.3",
    urls = ["https://github.com/protocolbuffers/protobuf/archive/v3.6.1.3.zip"],
)

http_archive(
    name = "bazel_skylib",
    sha256 = "bb6f842f7f9ac9d4e5ef9092ba44a8d76520978d0ff3f730f26d7f8aa9343e5d",
    strip_prefix = "bazel-skylib-d7c5518fa061ae18a20d00b14082705d3d2d885d",
    urls = ["https://github.com/bazelbuild/bazel-skylib/archive/d7c5518fa061ae18a20d00b14082705d3d2d885d.zip"],
)

patched_http_archive(
    name = "com_google_s2geometry",
    build_file = "//third_party:s2geometry.BUILD",
    patch_file = "//third_party:s2.patch",
    sha256 = "5ce9147ca1caeafc702bac3976cb4423fb6a58877ff72988c7f010b23e651b7b",
    strip_prefix = "s2geometry-dfefe0cd330f22463c0eca4338da69762cae7b49",
    urls = ["https://github.com/google/s2geometry/archive/dfefe0cd330f22463c0eca4338da69762cae7b49.zip"],
)

http_archive(
    name = "com_google_leveldb",
    build_file = "//third_party:leveldb.BUILD",
    sha256 = "6d91bcb96547bd8d43013e5ac9f4c2ad3227e30e3dc472cdafd2910eb43c16f8",
    strip_prefix = "leveldb-1cb384088184be9840bd59b4040503a9fa9aee66",
    urls = ["https://github.com/google/leveldb/archive/1cb384088184be9840bd59b4040503a9fa9aee66.zip"],
)

http_archive(
    name = "org_cairographics_cairo",
    build_file = "//third_party:cairo.BUILD",
    sha256 = "7e87878658f2c9951a14fc64114d4958c0e65ac47530b8ac3078b2ce41b66a09",
    strip_prefix = "cairo-1.14.10",
    urls = [
        "https://mirror.bazel.build/www.cairographics.org/releases/cairo-1.14.10.tar.xz",
        "https://www.cairographics.org/releases/cairo-1.14.10.tar.xz",
    ],
)

http_archive(
    name = "org_tensorflow",
    sha256 = "dfee0f57366a6fab16a103d3a6d190c327f01f9a12651e45a128051eaf612f20",
    strip_prefix = "tensorflow-1.11.0",
    urls = [
        "https://mirror.bazel.build/github.com/tensorflow/tensorflow/archive/v1.11.0.zip",
        "https://github.com/tensorflow/tensorflow/archive/v1.11.0.zip",
    ],
)

http_archive(
    name = "io_bazel_rules_closure",
    sha256 = "013b820c64874dae78f3dbb561f1f6ee2b3367bbdc10f086534c0acddbd434e7",
    strip_prefix = "rules_closure-0.8.0",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_closure/archive/0.8.0.zip",
        "https://github.com/bazelbuild/rules_closure/archive/0.8.0.zip",
    ],
)

load("@org_tensorflow//tensorflow:workspace.bzl", "tf_workspace")

tf_workspace(tf_repo_name = "org_tensorflow")

local_repository_env(
    name = "clif",
    build_file = "//third_party:clif.BUILD",
    variable = "CLIF_PATH",
)

new_local_repository(
    name = "opencv",
    build_file = "third_party/opencv.BUILD",
    path = "/usr/local",
)

new_local_repository(
    name = "cairo",
    build_file = "third_party/cairo.BUILD",
    path = "/usr",
)
