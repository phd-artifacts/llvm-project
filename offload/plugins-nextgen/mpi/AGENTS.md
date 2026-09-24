# Agent Notes: MPI Plugin (offload/plugins-nextgen/mpi)

MPP event routing, scheduler selection, and proxy-side I/O dispatch.

## Scope intent

- Keep this file focused on MPI-plugin runtime behavior.
- Shared commit workflow, submodule hygiene, and MkDocs maintenance are
  canonical in root `AGENTS.md`.

## Direction for MPP work

- Treat MPP as the distributed execution substrate for runtime-managed I/O:
  event transport, proxy execution, scheduler communication, and
  remote-worker mechanics belong here.
- Avoid turning this layer into the public semantic home for portable I/O
  policy; semantics should stay in `libompfile` unless the change is truly
  transport-specific.

## Relevant files

- Runtime bridge exports: `src/rtl.cpp`
- Proxy device and runtime bridge wrappers: `src/ProxyDevice.cpp`
  (`extern "C" ompfile_mpp_*` wrappers for proxy runtime)
- Headnode scheduler: `src/OmpFileHeadnodeManager.h`,
  `src/OmpFileHeadnodeManager.cpp`
- Event dispatch: `event_system/EventSystem.h`,
  `event_system/EventSystem.cpp`
- Bridge ABI shared with libompfile:
  `openmp/libompfile/include/ompfile_mpp_abi.h` (the entrypoint X-macro;
  both bridges define its prototypes); the scheduler wire structs are
  libompfile's `ompfile_sched.h`, aliased in `EventSystem.h`
- Shared environment readers: `openmp/libompfile/include/ompfile_env.h`
  (`normalizeStageWriteMode` and the `envBoolOrDefault` family forward to it)

## Event model

- File events: `OMPFILE_OPEN`, `OMPFILE_CLOSE`, `OMPFILE_PREAD`,
  `OMPFILE_PREAD_NO_STAGE`, `OMPFILE_PWRITE`, `OMPFILE_PING`
- Scheduler events: `OMPFILE_SCHED_REQUEST`, `OMPFILE_SCHED_PLAN`,
  `OMPFILE_SCHED_REQUEST_BATCH`
- Coherence events (write-back staging): `OMPFILE_STAGE_INVALIDATE`,
  `OMPFILE_FRESHNESS_QUERY`, `OMPFILE_FRESHNESS_WRITE_COMMIT`,
  `OMPFILE_FRESHNESS_MARK_FRESH`, `OMPFILE_PROXY_COPY_TILE`,
  `OMPFILE_FLUSH_DIRTY_TILE`, `OMPFILE_DIRTY_OWNER_PREAD`,
  `OMPFILE_DIRTY_OWNER_PREAD_BATCH`, `OMPFILE_DIRTY_OWNER_QUERY`
- The full enum with its pinned numeric values is `EventTypeTy` in
  `event_system/EventSystem.h`; the static_asserts there are the wire
  contract
- `ProxyDevice` creates rank-targeted events and waits for completion.

## Proxy queues and threads

- Three queues, one pool each: `ExecEventQueue` (`LAUNCH_KERNEL` only),
  `IoEventQueue` (every type `isFileIoEvent()` accepts — file ops, the
  `OMPFILE_SCHED_*` control events, the coherence events), `DataEventQueue`
  (everything else). The policy comment on those members in
  `EventSystem.h` is the contract; keep it current.
- A new `OMPFILE_*` event type must be added to `isFileIoEvent()` or it
  silently lands on the data queue.
- `OMPTARGET_NUM_IO_EVENT_HANDLERS` defaults to **4**, not 1, because an io
  handler can block: the owner-forwarding paths (`openOnRank`,
  `preadOnRank`, `pwriteOnRank`, `closeOnRank`) call `waitForEvent`, which
  is `EventTy::wait()`. With one io thread, two proxies forwarding to each
  other starve each other. Do not lower the default until those waits are
  `co_await`s.
- Each pool is at least one thread whatever its knob says.

## MPI call lock invariant (do not regress)

- `ompfileWithMPICallLock` serializes MPI calls process-wide when
  `LIBOMPFILE_MPI_SERIALIZE` is on (the default). **Only non-blocking MPI
  calls may go through it.** `sendBlocking`, `receiveBlocking` and
  `receiveInBatchsBlocking` call MPI directly on purpose: a rendezvous-sized
  blocking transfer held under that lock deadlocked two proxies that each
  wrote a file the other owned (Sep 2026, rtm-miniapp checkpoint lane;
  `docs/known-issues.md`). Safe because the runtime negotiates
  `MPI_THREAD_MULTIPLE`, each event owns its (communicator, tag), and
  `MPIRequestManagerTy` is move-only per-event state.
- The one blocking call still under the lock is the gate thread's
  `MPI_Mrecv` after a matching `MPI_Improbe`, which completes on call.

## Tracing

- NVTX ranges via `openmp/libompfile/include/ompfile_trace.h` only — never
  include NVTX directly. Compiled in when `OMPFILE_ENABLE_NVTX` (CMake,
  `openmp/libompfile/cmake/OmpFileNvtx.cmake`) finds the header — in practice
  always, since both clusters build in the `ompc-base` container.
- Event lifetimes are `rangeStart`/`rangeEnd` with the id in
  `promise_type::TraceRangeId`, never push/pop: a proxy event is resumed by
  whichever handler pops it. Push/pop (`Scope`) is for thread-local work
  only.
- Never put a range on a path that runs once per poll. `mpi-lock-wait` did,
  opening on every contended acquisition: 2.5 million ranges in a 15 s rtm
  run, a 7.5 GB nsys session, and a traced app rank that could not exit
  inside the case timeout. It now opens only past a 100 us wait
  (`OmpfileMPILockWaitTraceThreshold`); size any new range by its call
  count, not by how interesting it sounds.
- Collection recipe: `docs/getting-started.md`, "Tracing with Nsight
  Systems".

## Scheduler behavior

- `LIBOMPFILE_SCHEDULER=HEADNODE` triggers scheduler request on open.
- `OmpFileHeadnodeManager` keeps:
  - handler table (rank + in-flight load)
  - global file table (`path -> preferred aggregator rank`)
  - flightplan table (request tracking)
- If the scheduler request fails, or the plan names a rank that is not a
  worker, the open falls back to the local rank and says so on stderr
  (`MPIProxyDevice --> OMPFile scheduler fallback ...`, at every build
  level); a lane that sees that line measured the wrong aggregator.

## Ownership and handles

- `mppOpen` picks an aggregator rank, opens there, and stores
  local-handle -> {rank, remote_handle}.
- `mppPread`/`mppPwrite` route via that stored rank.
- `mppClose` closes on that stored rank and removes handle mapping.

## Debug checklist

- If `ActiveMPIPlugin is null`, confirm which process loads this code path
  and whether plugin init is valid in that role.
- If scheduling seems ignored, verify `LIBOMPFILE_SCHEDULER=HEADNODE` in
  the process environment.
- Use `LIBOMPFILE_OPT_STATS=1` to inspect open/close cache counters
  printed by proxy.
- Readthrough stage entries are process-local; if multiple proxies can
  share one host, the stage file path must include proxy-local identity.
  Keying only by source path lets separate proxy processes truncate the
  same `*.stage` file and produces misleading staged readback mismatches.
- `ProxyDevice::canUseOmpFileOpenCache()` already refuses cached writable
  opens. If a shared-file stale-read bug survives with
  `LIBOMPFILE_OPT_OPEN_CACHE=1`, do not assume `OmpFileOpenCacheByKey` is
  the cause; look deeper than proxy writable-open reuse.
- If the app rank reports PASS but the Slurm step still fails, inspect
  proxy rank tails for UCX `unexpected tag-receive descriptor` plus
  `MPIRequestManagerTy` shutdown errors; this indicates a proxy teardown
  bug, not a Cholesky numerical failure.
- Proxy/request-manager cleanup must not assume MPI is still callable
  during process teardown; late destructor-side MPI calls can fail with
  MPICH `internal_Cancel` after finalize and kill otherwise successful
  runs.
- In Cholesky packed-file lanes, MPICH `internal_Testall` on a proxy rank
  can come from the proxy-side async `OMPFILE_PWRITE` payload path, not
  from numerical failure or topology setup. A blocking receive control
  (`LIBOMPFILE_MPP_FORCE_BLOCKING_PWRITE=1`) is the fastest way to
  confirm that surface before broader tracing.
- `OMPFILE_PWRITE` origin events are sensitive to mixed request sets
  (payload Isends plus completion receives in one wait); split send and
  receive phases when debugging transport aborts to avoid chasing false
  topology/scheduler leads.

## Skills entrypoints

- `skills/submodule-commit-flow/SKILL.md`
- `skills/mkdocs-sync/SKILL.md`
