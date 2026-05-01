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

#ifdef _WIN32

#include <windows.h>

extern "C" int __stdcall MjObjDecoderDllMain(void* hinst, DWORD reason,
                                             void* reserved);
extern "C" int __stdcall MjStlDecoderDllMain(void* hinst, DWORD reason,
                                             void* reserved);

extern "C" BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
  return MjObjDecoderDllMain(hinst, reason, reserved) &&
         MjStlDecoderDllMain(hinst, reason, reserved);
}

#endif  // _WIN32
