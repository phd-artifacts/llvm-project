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

- Client side (`src/`, Oct 2026 split):
  - `file_interface.cpp`: the C ABI only (public header
    `include/file_interface.h`);
  - `client_context.{h,cpp}`: `OmpFileClientContext` (token guard, async
    wiring, handle set; it takes the `ClientConfig` snapshot and builds the
    backend);
  - `io_scheduler.{h,cpp}`: `IOScheduler`, `LocalScheduler` and
    `HeadnodeScheduler`, built through `createLocalScheduler` /
    `createHeadnodeScheduler`;
  - `async_write_engine.h`: the header-only `AsyncWriteEngine`.
- Backend: `src/mpp_backend.h` and the twelve `src/mpp_backend_*.cpp`
  (`MppBackend`, renamed from `MPIIOBackend` in Oct 2026; the test accessor is
  `MppBackendTestAccess`). It opens no file through MPI. The `io_trace`
  prefixes deliberately keep the old class and function names
  (`MPIIOBackend::...`, also where the code moved, e.g. the
  `writeAtWithContext freshness write-commit` trace now printed by
  `reportFreshnessWriteCommit`), and the `MPI backend selected` `io_log` line
  is unchanged, so log and trace output did not change with the rename. The
  mangled C++ symbols in `libompfile.so` did (`_ZN10MppBackend...`): nothing
  dlsyms them (only `ompfile_mpp_*` is), but a symbol check written against
  the old class name now finds nothing.
- Backend state layout (`mpp_backend.h`): `settings` (`Settings`, resolved
  once by the constructor from `config_`, read-only afterwards, no lock);
  `handles` (`HandleTable`), `remote` (`RemoteHandleTable`) and
  `write_epochs` (`WriteEpochHistory`), all guarded by `handle_mutex`;
  `counters` (`Counters`, the stats atomics; the lock counters are `mutable`
  so the `const` `instrumented*Lock()` helpers can count). A new member goes
  into its subsystem's struct and a new stats counter into `Counters`, with
  the token rules under "Useful knobs" unchanged: the `io_report` format in
  `mpp_backend_metrics.cpp`, not the field order, is the parser contract.
  `counters` must stay declared before the three batchers, whose `Stats` point
  into it.
- Backend helpers (`mpp_backend_remote_read_handles.cpp`): `finishRemoteRead`
  (the one EPROTO / short-read zero-fill tail of every completed remote
  pread), `lookupRemoteHandle` / `lookupRemoteOwnerRank` (the locked map
  lookups), `runOnRemoteRankHandle` (the per-rank two-try loop with eviction
  before the second try; a template defined only in that file, so a caller
  in another file gets a link error) and `readAtRemoteRankUsing`. Call these
  instead of copying the blocks.
- Route resolution: `src/ompfile_route.h` (`ompfile::route`), the decide half
  of `readAtWithContext` / `writeAtWithContext`; the executors are `MppBackend`
  members in `mpp_backend_read_path.cpp` / `_write_path.cpp`
  (`docs/architecture.md`, "Client read and write routes"). A resolver makes
  no MPP call, takes no lock, never sleeps, touches no counter, reads no knob
  and writes no `errno`. A new route is one `ReadRoute` / `WriteRoute` value,
  its resolver branch, its executor case and a `tests/route_test.cpp` case;
  a new resolver term needs a matching `route_test` case too.
- Inline batching: `src/ompfile_batcher.h` (`ompfile::batch::Batcher`). The
  two-phase read window and the write batch are `LeaderPerBatch` instances,
  whose leader holds its window to an absolute deadline (enqueue
  notifications never end it); the dirty-owner pread batch is
  `LeaderUntilEmpty`. A new batching path instantiates it, never a
  hand-rolled queue. The batcher mutex is a leaf lock and is never held
  across `execute()`. `ExpectedCount` is the graph-plan close rule.
- MPP shim: `src/mpp_shim.cpp`, `src/mpp_shim.h`
- Scheduler structs: `include/ompfile_sched.h` (the plugin aliases them; it
  is not installed, the plugin reaches it through its include path)
- Bridge ABI: `include/ompfile_mpp_abi.h` (the `OMPFILE_MPP_ENTRYPOINTS`
  X-macro; a new `ompfile_mpp_*` call is one row here plus its definition in
  `offload/plugins-nextgen/mpi/src/OriginOmpFileBridge.cpp` (origin) and/or
  `ProxyOmpFileBridge.cpp` (proxy))
- Configuration: the rules live in `include/ompfile_env.h`, and every knob a
  consumer acts on is parsed once into a snapshot in
  `include/ompfile_config.h`:
  - `ClientConfig`, when `OmpFileClientContext` is created (the first
    `omp_file_*` call, possibly before MPI is up), copied into `MppBackend`; a
    directly constructed `MppBackend()` snapshots at its own construction;
  - `ProxyConfig`, in the `ProxyDevice` constructor (the `ProxyOmpFile` member
    `OmpFileConfig`);
  - `HeadnodeConfig`, in `OmpFileHeadnodeManager::initialize()`.

  A knob both sides read sits in `SharedConfig` (scheduler, stage mode, stage
  write mode, freshness guard, write-back distribution) with one rule.
  Consumers read fields, never `getenv` or `ompfile::env::` directly. The
  exceptions are the once-per-process gates: `debug_log.h`
  (`LIBOMPFILE_DEBUG`, `_DEBUG_TRACE`), the proxy's `ompfileProxyTraceEnabled`,
  the three EventSystem statics, and the `LIBOMPFILE_IO_TOKENS` `std::stoi` in
  `client_context.cpp` (the snapshot keeps the raw string, because the header
  is exception-free for the proxy). A new knob is one field plus its rule, a
  `tests/env_test.cpp` case and a `docs/runtime-configuration.md` row.
  Setting a knob after the first `omp_file_*` call has no effect. libompfile
  never reads an application knob (the Cholesky
  `OMPFILE_CHOLESKY_WORKER_NODES` / `_TARGET_REGIONS` fallback for the
  distribute rank count went in Oct 2026; the proxy's `OMPFILE_CHOLESKY_NB`
  stage-tile fallback is a known issue). `LIBOMPFILE_OPT_STATS` is the one
  shared knob with two parsers: libompfile accepts only `1`, the proxy's LLVM
  `BoolEnvar` also `true` / `yes` / `on`; unifying them is a behavior change
  left for later.
- Keys shared with the proxy and the headnode: `include/ompfile_keys.h`
  (`pathKey`, `tileKey` / `tileKeyForHint`, `readGroupKey`, `writeGroupKey`,
  `handleKey`). Path and tile keys cross process boundaries and every side
  recomputes them, so there is no local copy anywhere; never change a
  constant, the mix order or a guard. The path-key offset basis is
  deliberately not the standard FNV one (its last digit is missing) and must
  stay for wire and log compatibility. `tests/keys_test.cpp` pins every
  function against the bodies it replaced and against keys logged by sorgan
  jobs 3168-3170. The header is not installed.
- Stage extent algebra (pure, unit-tested): `include/ompfile_extents.h`
- Source lists: `cmake/OmpFileSources.cmake`: `OMPFILE_FRONTEND_SOURCES_REL`
  (the client side) and `OMPFILE_MPP_BACKEND_SOURCES_REL` (the backend and the
  shim). `application/tests/test-mpp-rebalance-fallback-regression` includes
  the file and compiles only the backend list (its prefixed copy is
  `OMPFILE_MPP_BACKEND_SOURCES`).
- Unit tests: `tests/` (`ompfile_unit_tests`: wire codec, extent algebra, env
  rules, keys, batcher, route resolution; built with the runtime, run by the
  build jobs; add a case there for any new pure helper). The batcher cases
  run real threads and two of them hold a deliberate 2 s window, so the
  binary takes a few seconds.

## The one mode: remote-only MPP

- Remote-only MPP is libompfile's only mode (Oct 2026): every `omp_file_*`
  operation goes through `ompfile::mpp::*` to a proxy. The POSIX, io_uring and
  MPI-IO backends, the mixed mode (`MPP_OPEN=1 MPP_IO=0`), `IOBackendTy` and
  the liburing build dependency are gone. The backend is `MppBackend`
  (renamed from `MPIIOBackend`, Oct 2026); it opens no file through MPI.
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
  `remote.remote_file_handle_map` (`RemoteHandleTable`, guarded by
  `handle_mutex`); read it through `lookupRemoteHandle` /
  `lookupRemoteOwnerRank` unless the caller already holds the lock.
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
