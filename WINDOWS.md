# Windows Migration

Note that this merge request only intends to work on Windows. I haven't write code that switch support between linux and Windows. I'll update that feature after the whole thing can be build on Windows.

## Setup Environment On Windows

1. Install [Windows Powershell](https://learn.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.2). Do step 2 and 3 in Powershell.
2. Install [choco](https://chocolatey.org/install#individual)
3. Install bazelisk/bazel ``` choco install bazelisk```
4. Install [MinGW-w64](https://www.mingw-w64.org/) or via [released version](https://winlibs.com/#download-release)
5. Install [git bash](https://gitforwindows.org/) and make sure it uses MINGW64 emulation. Do all the following steps in bash.
6. Install [VC++ 2017](https://www.google.com/url?q=https://aka.ms/vs/15/release/vs_buildtools.exe&sa=D&source=docs&ust=1667504254128145&usg=AOvVaw3jOTmoHfuzAHgaScSgsmlE) on Windows
7. Install python 3.10 (instead of 3.11 because pygame installation fails on Windows)
8. Go where python.exe is installed, make a copy of python.exe and rename it python3.exe (it would bypass the error "python interpreter cannot be found with requirement")
9. Install [swig.exe](https://sourceforge.net/projects/swig/files/swigwin/swigwin-4.1.0/swigwin-4.1.0.zip/download?use_mirror=gigenet) and merge it into python folder and do [some hacks](https://stackoverflow.com/questions/44504899/installing-pocketsphinx-python-module-command-swig-exe-failed)
10. Install [Clang](https://bazel.build/configure/windows#using) and follow this [link](https://github.com/llvm/llvm-project/releases/tag/llvmorg-15.0.2). Don't forget to set BAZEL_VC and BAZEL_LLVM in environmental variable manually.
11. Now you can run some tests below 





## Can build
-  Dummy  
```bazel build --test_output=all //envpool/dummy:dummy --config=debug_win32 ```  
```bazel build --test_output=all //envpool/dummy:dummy_envpool_test --config=debug_win32```  

- Utils
- Box2d  
```bazel build --test_output=all //envpool/box2d:box2d_deterministic_test --config=debug_win32```
- Classic Control
- Toy_text 
- Mujoco
- Atari:atari_env


## TODO
- Atari:atari (Stuck on building opencv)
- Minor Format and Bug fix
- Windows/Linux Integration and submit (Working on this)
