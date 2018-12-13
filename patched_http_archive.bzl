"""This module contains implementation for patched_http_archive."""

# Executes specified command with arguments and calls 'fail' if it exited with
# non-zero code
def _execute_and_check_ret_code(repo_ctx, cmd_and_args):
    result = repo_ctx.execute(cmd_and_args, timeout = 10)
    if result.return_code != 0:
        fail(("Non-zero return code({1}) when executing '{0}':\n" + "Stdout: {2}\n" +
              "Stderr: {3}").format(
            " ".join(cmd_and_args),
            result.return_code,
            result.stdout,
            result.stderr,
        ))

# Apply a patch_file to the repository root directory
# Runs 'patch -p1'
def _apply_patch(repo_ctx, patch_file):
    cmd = [
        "patch",
        "-p1",
        "-d",
        repo_ctx.path("."),
        "-i",
        repo_ctx.path(patch_file),
    ]
    _execute_and_check_ret_code(repo_ctx, cmd)

# Download the repository and apply a patch to its root
def _patched_http_archive_impl(repo_ctx):
    repo_ctx.download_and_extract(
        repo_ctx.attr.urls,
        sha256 = repo_ctx.attr.sha256,
        stripPrefix = repo_ctx.attr.strip_prefix,
    )
    _apply_patch(repo_ctx, repo_ctx.attr.patch_file)
    if hasattr(repo_ctx.attr, "build_file"):
        repo_ctx.symlink(repo_ctx.attr.build_file, "BUILD.bazel")

patched_http_archive = repository_rule(
    implementation = _patched_http_archive_impl,
    attrs = {
        "patch_file": attr.label(),
        "build_file": attr.label(),
        "repository": attr.string(),
        "urls": attr.string_list(default = []),
        "sha256": attr.string(default = ""),
        "strip_prefix": attr.string(default = ""),
    },
)
