# Copyright 2026 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Repository rule for the Freedoom release asset."""

def _freedoom_archive_impl(ctx):
    last_error = "download was not attempted"
    for attempt in range(ctx.attr.attempts):
        ctx.report_progress(
            "Fetching Freedoom asset, attempt %d/%d" %
            (attempt + 1, ctx.attr.attempts),
        )
        result = ctx.download_and_extract(
            url = ctx.attr.urls,
            sha256 = ctx.attr.sha256,
            type = ctx.attr.type,
            strip_prefix = ctx.attr.strip_prefix,
            canonical_id = ctx.attr.canonical_id,
            allow_fail = True,
        )
        if result.success:
            ctx.file("BUILD.bazel", ctx.read(ctx.attr.build_file))
            return
        last_error = result.error

    fail("failed to fetch Freedoom asset after %d attempts: %s" % (
        ctx.attr.attempts,
        last_error,
    ))

freedoom_archive = repository_rule(
    implementation = _freedoom_archive_impl,
    attrs = {
        "attempts": attr.int(default = 8),
        "build_file": attr.label(allow_single_file = True, mandatory = True),
        "canonical_id": attr.string(default = "freedoom-0.12.1.zip"),
        "sha256": attr.string(mandatory = True),
        "strip_prefix": attr.string(default = ""),
        "type": attr.string(default = ""),
        "urls": attr.string_list(mandatory = True),
    },
)
