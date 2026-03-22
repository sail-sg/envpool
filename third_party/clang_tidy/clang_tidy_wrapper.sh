#!/usr/bin/env bash
# Copyright 2023 Garena Online Private Limited.
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

set -euo pipefail

if command -v clang-tidy-18 >/dev/null 2>&1; then
  clang_tidy_bin="clang-tidy-18"
else
  clang_tidy_bin="clang-tidy"
fi

gcc_install_dir="$(dirname "$(gcc -print-file-name=include)")"

mapfile -t builtin_include_dirs < <(
  echo | g++ -E -x c++ - -v 2>&1 | awk '
    /#include <...> search starts here:/ { in_list = 1; next }
    /End of search list./ { in_list = 0 }
    in_list {
      sub(/^[[:space:]]+/, "", $0)
      print $0
    }
  '
)

should_skip_include_dir() {
  local candidate="$1"
  local include_dir
  for include_dir in "${builtin_include_dirs[@]}"; do
    if [[ "${candidate}" == "${include_dir}" ]]; then
      return 0
    fi
  done
  return 1
}

filtered_args=()
inserted_gcc_install_dir=0
while (($#)); do
  if [[ "$1" == "-isystem" && $# -ge 2 ]] && should_skip_include_dir "$2"; then
    shift 2
    continue
  fi
  if [[ "$1" == "--" && ${inserted_gcc_install_dir} -eq 0 ]]; then
    filtered_args+=("$1" "--gcc-install-dir=${gcc_install_dir}")
    inserted_gcc_install_dir=1
    shift
    continue
  fi
  filtered_args+=("$1")
  shift
done

exec "${clang_tidy_bin}" "${filtered_args[@]}"
