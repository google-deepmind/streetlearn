# Unofficial Bazel build for LevelDB.

cc_library(
    name = "leveldb",
    srcs = glob(
        ["**/*.cc"],
        exclude = [
            "doc/**",
            "**/*_test.cc",
            "db/leveldbutil.cc",
            "db/db_bench.cc",
        ],
    ),
    hdrs = glob(
        ["**/*.h"],
        exclude = ["doc/**"],
    ),
    includes = ["include", "."],
    defines = [
        "LEVELDB_PLATFORM_POSIX=1",
        "LEVELDB_IS_BIG_ENDIAN=0",
    ],
    linkstatic = 1,
    visibility = ["//visibility:public"],
)
