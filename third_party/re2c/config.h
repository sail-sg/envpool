// Copyright 2026 Garena Online Private Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_RE2C_CONFIG_H_
#define THIRD_PARTY_RE2C_CONFIG_H_

#define PACKAGE_VERSION "4.5.1"

#ifndef RE2C_STDLIB_DIR
#define RE2C_STDLIB_DIR ""
#endif

#define HAVE_STDINT_H 1
#define HAVE_STDLIB_H 1
#define HAVE_STRING_H 1

#ifdef _WIN32
#define HAVE_IO_H 1
#else
#define HAVE_FCNTL_H 1
#define HAVE_SYS_STAT_H 1
#define HAVE_SYS_TYPES_H 1
#define HAVE_UNISTD_H 1
#endif

#endif  // THIRD_PARTY_RE2C_CONFIG_H_
