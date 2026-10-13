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
- Scheduler structs: `include/ompfile_sched.h` (the plugin aliases them; it
  is not installed, the plugin reaches it through its include path)
- Bridge ABI: `include/ompfile_mpp_abi.h` (the `OMPFILE_MPP_ENTRYPOINTS`
  X-macro; a new `ompfile_mpp_*` call is one row here plus its definition in
  `rtl.cpp` and/or `ProxyDevice.cpp`)
- Environment readers shared with the proxy: `include/ompfile_env.h`. A knob
  both sides read gets one reader there (`schedulerIsHeadnode()`,
  `writebackDistributeActive()` / `writebackDistributeRankCount()`), never a
  local `getenv`; the runtime never reads an application knob (the Cholesky
  `OMPFILE_CHOLESKY_WORKER_NODES` / `_TARGET_REGIONS` fallback for the
  distribute rank count went in Oct 2026).
- Stage extent algebra (pure, unit-tested): `include/ompfile_extents.h`
- Source lists: `cmake/OmpFileSources.cmake` (also included by
  `application/tests/test-mpp-rebalance-fallback-regression`)
- Unit tests: `tests/` (built with the runtime, run by the build jobs;
  add a case there for any new pure helper)

## The one mode: remote-only MPP

- Remote-only MPP is libompfile's only mode (Oct 2026): every `omp_file_*`
  operation goes through `ompfile::mpp::*` to a proxy. The POSIX, io_uring and
  MPI-IO backends, the mixed mode (`MPP_OPEN=1 MPP_IO=0`), `IOBackendTy` and
  the liburing build dependency are gone. `MPIIOBackend` and the
  `mpi_io_backend_*` files keep their names until the A3 rename; the backend
  opens no file through MPI.
- `LIBOMPFILE_BACKEND` (unset, empty or `MPI`) and `LIBOMPFILE_MPP_OPEN` /
  `LIBOMPFILE_MPP_IO` (unset, empty or `1`) need not be set
  (`ompfile::env::kModeKnobs`). Any other value is an init failure: one
  `libompfile init failed: NAME='value' requests a removed mode; ...`
  `io_report` line, then every I/O call fails with EIO, the same path as an
  MPP shim init failure. There is no local fallback
  (`LIBOMPFILE_ALLOWFALLBACK` is gone).
- Removed knobs are listed once, in `ompfile::env::kRemovedKnobs`
  (`include/ompfile_env.h`). Backend init prints one
  `removed knob ignored: A, B` `io_report` line naming every entry set to a
  non-empty value. `docs/runtime-configuration.md`, "Removed knobs
  (Oct 2026)", mirrors the list: change both together. Lane wrappers fail on
  `FAIL` in a log, so that warning must never contain it.
- libompfile never calls `MPI_Init`: when MPI is not up it calls
  `ompfile::mpp::init()`, which brings up the MPI plugin, and the plugin
  initializes MPI (`MPI_THREAD_MULTIPLE`). libompfile only queries MPI state
  (`MPI_Initialized`, `MPI_Query_thread`, `MPI_Comm_rank` and the like); do
  not add an MPI call that does more. The MPI link stays for those queries.
- `omp_file_read` / `omp_file_write` / `omp_file_seek` (no explicit offset)
  return -1 with ENOTSUP; sequential I/O has no MPP path.
- `LIBOMPFILE_SCHEDULER` defaults to HEADNODE (Oct 2026). `LOCAL` stays
  selectable, but an origin-side open under it fails: the unplanned open
  (`ompfile::mpp::open`, then `handleOwnerRank`) needs
  `ompfile_mpp_handle_owner_rank`, which only the proxy defines. Semantics
  are in `docs/runtime-configuration.md`.

## MPP shim behavior

- `mpp_shim` uses `dlsym(RTLD_DEFAULT, "ompfile_mpp_*")` to resolve
  runtime bridge symbols.
- If symbols are missing or init fails, operations fail and log explicit
  diagnostics.
- Reads and writes need `ompfile_mpp_pread_ex` / `ompfile_mpp_pwrite_ex`
  (`MPP shim not available (ompfile_mpp_pread_ex missing)` otherwise); the
  plain `pread` / `pwrite` rows are gone. The proxy-only rows are optional:
  `preadNoStageEx` falls back to `preadEx`, the others fail with ENOSYS
  (`include/ompfile_mpp_abi.h`).
- Scheduler requests use `ompfile_mpp_sched_request` via shim API.

## Handle model

- Backend assigns local logical file ids.
- For MPP open path, local id maps to remote handle tracked in
  `remote_file_handle_map`.
- `readAt/writeAt` always route through the mapped remote handle.
- Under HEADNODE the client records each handle's OPEN-plan `AggregatorRank`
  (`rememberOpenPath`) and sends it back with that handle's CLOSE in
  `OmpFileIORequest::AggregatorRank`, which is how the headnode releases the
  rank's live-handle count (accounting only; placement weighs plan counts);
  `takePlannedRank` clears it so a retried close cannot release twice. A
  backend open that fails after a counted OPEN plan sends a release-only
  CLOSE (`FileHandle` -1) instead. A new path that takes an OPEN plan must
  close it or release it the same way.

## Useful knobs

- `LIBOMPFILE_MPP_PING=1` for health check.
- `LIBOMPFILE_DEBUG=0` silences `io_log`; the stats lines lanes parse go
  through `io_report` and cannot be silenced.
- `LIBOMPFILE_SCHEDULER`: HEADNODE is the default since Oct 2026; `LOCAL`
  turns the headnode scheduler off.
- `LIBOMPFILE_OPT_OPEN_CACHE`, `LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN`,
  `LIBOMPFILE_OPT_STATS` influence proxy-side behavior.
- The `Two-phase stats:` and `Async IO stats:` lines are a parser contract:
  keep the token order, never insert a token before an existing one, and keep
  `mpp_call_mutex_hold_us=` last on `Two-phase stats:`
  (`application/tests/benchmarks/libompfile-micro/extract_micro_metrics.py`
  stops there). Removing a token is a parser change too: grep `application/`
  and `spinner/` for its readers in the same change.

## Failure triage

- `MPP shim init failed`: check runtime symbol export path and plugin
  initialization.
- Every call fails with EIO right after start and the log has one
  `libompfile init failed:` line naming `LIBOMPFILE_BACKEND`, `_MPP_OPEN` or
  `_MPP_IO`: the environment asks for a removed mode; drop the export. A
  `removed knob ignored:` line is different: the run goes on.
- Open/read errors with MPP enabled: inspect both libompfile logs and
  proxy/EventSystem logs together.

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
