# Why there is an empty `libadvapi32.a` here

`whisper-rs-sys`'s build script emits `cargo:rustc-link-lib=advapi32` inside an
`if cfg!(target_os = "windows")`. A build script is compiled for the *host*, so
that branch is taken when cross-compiling from Windows to Android too, and the
Android linker then fails with `cannot find -ladvapi32`.

The whisper.cpp code that actually calls into advapi32 is behind `#ifdef _WIN32`
and is not compiled for Android, so nothing references those symbols. An empty
archive satisfies the linker without hiding anything: if a real reference ever
appeared, it would still fail to resolve.

Delete this once whisper-rs-sys reads `CARGO_CFG_TARGET_OS` instead of `cfg!`.
