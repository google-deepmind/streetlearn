"""Custom Bazel rules."""

def _local_repository_env_impl(repository_ctx):
    repository_ctx.symlink(
        repository_ctx.os.environ[repository_ctx.attr.variable],
        repository_ctx.attr.variable,
    )
    repository_ctx.symlink(repository_ctx.attr.build_file, "BUILD.bazel")
    return None

local_repository_env = repository_rule(
    implementation = _local_repository_env_impl,
    local = True,
    attrs = {
        "variable": attr.string(mandatory = True),
        "build_file": attr.label(mandatory = True),
    },
)
