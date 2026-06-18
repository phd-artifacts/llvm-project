# Agent Notes: libompfile (openmp/libompfile)

libompfile backend behavior and MPP shim usage.

## Scope intent

- Keep this file focused on libompfile runtime behavior.
- Shared commit workflow, submodule hygiene, and MkDocs maintenance are
  canonical in root `AGENTS.md`.

## Direction for libompfile work

- `libompfile` is the semantic layer for portable file I/O in this repo.
- New policy should prefer living here when it concerns backend-agnostic
  I/O meaning, request planning, scheduler hints, staging policy, or
  future runtime visibility of file operations.
- Avoid pushing semantic policy down into benchmark wrappers or directly
  into MPP transport code when the logic is meant to be portable across
  backends and systems.

## Relevant files

- Backend interface/state: `src/mpi_io_backend.h`
- Backend lifecycle: `src/mpi_io_backend_lifecycle.cpp`
- Backend operations and policy helpers:
  `src/mpi_io_backend_*.cpp`
- Public interface glue: `src/file_interface.cpp`,
  `include/file_interface.h`
- MPP shim: `src/mpp_shim.cpp`, `src/mpp_shim.h`
- Scheduler structs: `include/ompfile_sched.h`

## Backend modes

- MPI backend selected by `LIBOMPFILE_BACKEND=MPI`.
- Remote-only MPP mode is active when both are true:
  - `LIBOMPFILE_MPP_OPEN=1`
  - `LIBOMPFILE_MPP_IO=1`
- In remote-only mode, file operations use `ompfile::mpp::*` path and
  skip MPI file communicator duplication.

## MPP shim behavior

- `mpp_shim` uses `dlsym(RTLD_DEFAULT, "ompfile_mpp_*")` to resolve
  runtime bridge symbols.
- If symbols are missing or init fails, operations fail and log explicit
  diagnostics.
- Scheduler requests use `ompfile_mpp_sched_request` via shim API.

## Handle model

- Backend assigns local logical file ids.
- For MPP open path, local id maps to remote handle tracked in
  `remote_file_handle_map`.
- `readAt/writeAt` use mapped remote handle in MPP I/O mode.

## Useful knobs

- `LIBOMPFILE_MPP_PING=1` for health check.
- `LIBOMPFILE_MPI_COMM_SELF=1` to force `MPI_COMM_SELF` in non-remote-only
  mode.
- `LIBOMPFILE_SCHEDULER=HEADNODE` to request headnode scheduling policy.
- `LIBOMPFILE_OPT_OPEN_CACHE`, `LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN`,
  `LIBOMPFILE_OPT_STATS` influence proxy-side behavior.

## Failure triage

- `MPP shim init failed`: check runtime symbol export path and plugin
  initialization.
- Open/read errors with MPP enabled: inspect both libompfile logs and
  proxy/EventSystem logs together.
- Two-phase read cache entries are keyed by path plus mixed hint metadata
  (`epoch/stream/tile`); invalidating only the base file/path key after
  writes can leave stale entries alive under older hint variants.
- Until per-file mixed-key tracking exists, prefer conservative full
  two-phase cache invalidation on write-path fixes over partial key
  erasure.
- In remote-only MPP mode, the two-phase read cache is local to each
  proxy and has no cross-proxy invalidation for shared writable files.
  Keep it disabled/bypassed for writable distributed paths until the
  runtime can invalidate cached reads across proxies.

## Defensive checks

- When modifying libompfile C/C++ code, add `assert(...)` around internal
  bounds invariants (file-id ranges, map/vector index use, offset/size
  arithmetic).
- Keep explicit error-path checks for external/runtime failures
  (`errno`/return codes); do not rely on asserts for recoverable
  conditions.

## Skills entrypoints

- `skills/submodule-commit-flow/SKILL.md`
- `skills/mkdocs-sync/SKILL.md`
