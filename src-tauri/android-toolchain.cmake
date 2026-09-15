# Wrapper around the NDK's own toolchain file, used only when cross-compiling
# whisper.cpp for Android.
#
# whisper-rs-sys builds through the `cmake` crate, which offers no way to pass
# ANDROID_ABI. Left unset, the NDK toolchain falls back to armeabi-v7a and emits
# `-march=armv7-a`, while the crate has separately told clang `--target=aarch64`.
# clang then refuses the pair:
#
#     error: unsupported argument 'armv7-a' to option '-march='
#
# So the ABI is pinned here, before handing over to the real toolchain.

if(DEFINED ENV{ANDROID_ABI})
  set(ANDROID_ABI $ENV{ANDROID_ABI} CACHE STRING "" FORCE)
else()
  set(ANDROID_ABI arm64-v8a CACHE STRING "" FORCE)
endif()

# 24 matches the linkers configured in .cargo/config.toml
set(ANDROID_PLATFORM android-24 CACHE STRING "" FORCE)

if(NOT DEFINED ENV{NDK_HOME})
  message(FATAL_ERROR "NDK_HOME is not set; it is needed to find the NDK toolchain file")
endif()

include("$ENV{NDK_HOME}/build/cmake/android.toolchain.cmake")

# whisper-rs-sys adds MSVC's /utf-8 whenever the *host* is Windows, which
# includes cross-compiling to Android with clang. clang reads it as a filename:
#
#     clang++: error: no such file or directory: '/utf-8'
string(REPLACE "/utf-8" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "" FORCE)
