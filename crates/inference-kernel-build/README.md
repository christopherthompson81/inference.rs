# inference-kernel-build

Build-time CUDA compilation for inference.rs's kernel crates (`build.rs` files): nvcc detection, arch lists,
incremental parallel compiles under cargo's jobserver, fatbins, static and shared kernel libraries, and fetched
dependencies such as CUTLASS (cached under `~/.cudaforge`, or `CUDAFORGE_HOME`; `CUDAFORGE_THREADS` caps the pool).

Derived from [cudaforge](https://github.com/guoqingbao/cudaforge) 0.1.6 by Guoqing Bao (MIT OR Apache-2.0), and
maintained here; upstream is no longer tracked. The cache directory and environment variables keep cudaforge's names
so existing caches and settings still apply.

## Changes from cudaforge 0.1.6

- `src/compute_cap.rs`, `src/builder.rs`: multi-arch builds. `CUDA_COMPUTE_CAP` takes a list (`80,86,90`, `8.6`,
  `sm_90`, `90a`), parsed by `parse_arch_list`; nvidia-smi's `8.6`-style output is parsed; `get_for_file` returns
  every arch a file builds for, `get_compute_caps` lists them, and each compile passes one `-gencode` per arch.
- clippy fixes (`strip_suffix` instead of manual slicing).
- New `src/jobserver.rs`. In `src/builder.rs`, each nvcc compile holds a cargo jobserver slot (the build script's
  implicit token, or one requested from cargo), so kernel crates building at once share `cargo -j` instead of each
  running a full pool. The thread-count warning is reworded to say so.
- `src/parallel.rs`: the default pool size is raised from half to all cores, since the jobserver now bounds
  concurrency. `CUDAFORGE_THREADS` still caps it, and its test is updated to match.
- `Cargo.toml`: adds the `jobserver` dependency.
- `src/builder.rs`: `build_loadable_lib` builds kernel libraries for `inference_tensor::kernel_library!` to open at
  runtime (a static archive on targets without dlopen). It keys the build dir by its compile inputs, under
  `<target>/<profile>/cuda-kernels` in dev builds (every cargo variant maps one copy) and under
  `~/.cache/inference-rs/kernels/built` otherwise. Each source compiles once per listed arch (objects named
  `<stem>-<hash>-sm<arch>.o`), and each library links once per arch into `sm<arch>/lib{name}.so` (soname
  `lib{name}.so`); a source whose override leaves an arch out links only into the archs it builds for. Every copy
  carries one build id, hashed over all archs' objects and written to `lib{name}.so.id`, which the loader checks; a
  `cuda-kernels/<name>.lib` pointer names the library for packaging. `split_lib` links the sources matching a pattern
  into a library of their own from the same compile pass. `build_lib` shares `compile_objects` with it, which removes a stale output when
  a compile fails, and links through `run_link`.
- `src/builder.rs`: `build_fatbin` builds one fatbin per source with the incremental, parallel path of `build_ptx`
  (both now pass `-o`), and `PtxOutput` writes `include_bytes!` consts for fatbins and lists the built `images()`.
  Sources inside the builder's own out dir (generated PTX) get no `rerun-if-changed`, which would rerun every build.
- `src/builder.rs`: `build_lib` and `build_loadable_lib` print the toolkit's library directory as a link search path,
  since the kernel libraries link `cudart` and cudarc no longer adds it when it loads its libraries at runtime.
- `src/builder.rs`: adds `KernelBuilder::compress_fatbin`, which passes `-compress-mode=size` from CUDA 12.8 and
  `-Xfatbin=-compress-all` before it, with a unit test for the version threshold.
- Rust edition 2024 (cudaforge is 2021), inherited from the workspace: clippy's collapsed `if let` chains and rustfmt's
  2024 style edition.
