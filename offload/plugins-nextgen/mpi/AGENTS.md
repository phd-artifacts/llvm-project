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
  (`normalizeStageWriteMode` and the `envBoolOrDefault` family forward to it).
  A knob libompfile and the proxy both read has exactly one reader there —
  `schedulerIsHeadnode()` for `LIBOMPFILE_SCHEDULER`,
  `writebackDistributeActive()` for distributed write-back (the proxy's
  derived-tile dirty-flush completion uses it) — never a local `getenv`.

## Event model

- File events: `OMPFILE_OPEN`, `OMPFILE_CLOSE`, `OMPFILE_PREAD`,
  `OMPFILE_PREAD_NO_STAGE`, `OMPFILE_PWRITE`, `OMPFILE_PING`
- Scheduler events: `OMPFILE_SCHED_REQUEST`, `OMPFILE_SCHED_REQUEST_BATCH`
  (wire value 33, the former `OMPFILE_SCHED_PLAN`, is reserved)
- Coherence events (write-back staging): `OMPFILE_FRESHNESS_QUERY`,
  `OMPFILE_FRESHNESS_WRITE_COMMIT`, `OMPFILE_FLUSH_DIRTY_TILE`,
  `OMPFILE_DIRTY_OWNER_PREAD`, `OMPFILE_DIRTY_OWNER_PREAD_BATCH`,
  `OMPFILE_DIRTY_OWNER_QUERY` (wire values 36, 39 and 40, the former
  `OMPFILE_STAGE_INVALIDATE`, `OMPFILE_PROXY_COPY_TILE` and
  `OMPFILE_FRESHNESS_MARK_FRESH`, are reserved)
- The full enum with its pinned numeric values is `EventTypeTy` in
  `event_system/EventSystem.h`; the static_asserts there are the wire
  contract. A removed event leaves a "reserved" comment where its enumerator
  and its static_assert were, and the next enumerator keeps an explicit value
  (`SYNC = 34`); never renumber or reuse a reserved value
- `ProxyDevice` creates rank-targeted events and waits for completion.

## Proxy queues and threads

- Three queues, one pool each: `ExecEventQueue` (`LAUNCH_KERNEL` only),
  `IoEventQueue` (every type `isFileIoEvent()` accepts — file ops, the
  `OMPFILE_SCHED_*` control events, the coherence events), `DataEventQueue`
  (everything else). The policy comment on those members in
  `EventSystem.h` is the contract; keep it current.
- A new `OMPFILE_*` event type must be added to `isFileIoEvent()` or it
  silently lands on the data queue.
- `OMPTARGET_NUM_IO_EVENT_HANDLERS` defaults to **4**, not 1, as margin for
  the one place an io handler blocks: the headnode coherence round trips
  (`freshnessQueryOnHeadnode`, `freshnessWriteCommitOnHeadnode`,
  `completeDirtyFlushOnHeadnode`, also reached through
  `completeDerivedTileFlushesOnHeadnode`) under write-back staging on a
  non-headnode rank, via `waitForEvent`, which is `EventTy::wait()`. The
  owner-forwarding helpers wait only on kernel threads (see "Which threads
  can block on a peer" below). Do not lower the default until the coherence
  waits are `co_await`s.
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

- HEADNODE is the scheduler default since Oct 2026: `LIBOMPFILE_SCHEDULER`
  unset, empty, `HEADNODE` (any case) or an unknown value all mean HEADNODE,
  and only `LOCAL` turns it off (`ompfile::env::schedulerIsHeadnode()`).
  Under HEADNODE an open sends a scheduler request.
- `OmpFileHeadnodeManager` keeps:
  - handler table, per worker rank: `PlansIssued` (one per scalar plan, OPEN
    included, batch segments not; monotonic; the load of every pick),
    `PlannedTotal` (every plan plus one per batch segment; monotonic; the load
    of every pick under `LIBOMPFILE_SCHED_USE_PLANNED_LOAD_FOR_SKEW=1`) and
    `OpenHandles` (live open handles, accounting only: an OPEN plan adds one on
    the rank it names, the CLOSE carrying that rank in
    `OmpFileIORequest::AggregatorRank` removes it, saturating at 0)
  - global file table (`path -> preferred aggregator rank`)
- Every pick, OPEN included (scalar OPEN/PREAD/PWRITE/CLOSE, batch segments),
  weighs `PlansIssued` with the round-robin tie-breaker, exactly the rule from
  before the Oct 2026 cleanup (`InFlight` was that count under another name).
  `OpenHandles` never steers placement: an intermediate version that placed
  OPENs by it put an origin's sequentially opened files on one proxy rank and
  broke the Cholesky two-phase regression (`docs/known-issues.md`), and
  `test-mpp-rebalance-fallback-regression` (`sequential-open-spread`) now
  guards that. Making it a scheduler signal is a placement change, to be
  validated as one. `planRequest` keeps its first
  `pickLeastLoadedRankUnlocked` call for an OPEN whose result the OPEN branch
  overwrites, because the call advances the tie-breaker. Keep it.
- A CLOSE with `FileHandle < 0` is release-only: it gives back the live
  handle of its `AggregatorRank` and plans nothing (no count, no tie-breaker
  step). libompfile sends one when its backend open fails after a counted
  OPEN plan.
- Only libompfile's `HeadnodeScheduler` pairs a counted OPEN plan with its
  CLOSE. The bridge's own pick (`selectAggregatorRankForOpen`, in `mppOpen`)
  sets `OMPFILE_IO_REQUEST_PLACEMENT_ONLY` in `OmpFileIORequest::Flags`: the
  headnode places and counts it like any OPEN but adds no live handle. Any new
  OPEN-plan requester must do one or the other, or the live-handle counts only
  grow.
- `LIBOMPFILE_DEBUG_TRACE=1` on the headnode prints one
  `[ompfile-mpp][sched] open-handles op=open|close rank=<r> open_handles=<n>`
  line per live-handle change.
- If the scheduler request fails, or the plan names a rank that is not a
  worker, the open falls back to the local rank and says so on stderr
  (`MPIProxyDevice --> OMPFile scheduler fallback ...`, at every build
  level); a lane that sees that line measured the wrong aggregator.

## Ownership and handles

- `mppOpen` picks an aggregator rank (under HEADNODE a placement-only
  scheduler request), opens there, and stores
  local-handle -> {rank, remote_handle}.
- `mppPreadEx`/`mppPwriteEx` route via that stored rank (the plain
  `mppPread`/`mppPwrite` and their `ompfile_mpp_pread`/`_pwrite` bridges were
  removed in Oct 2026).
- `mppClose` closes on that stored rank and removes handle mapping.
- Several `ompfile_mpp_*` exports exist twice: the origin definition in
  `src/rtl.cpp` and a proxy-side one in `src/ProxyDevice.cpp`, which is what a
  call from inside a target region resolves to. A proxy-side definition must
  not assume the proxy running the region owns the file: the HEADNODE may
  place it on a peer, and which one varies per run. `ompfile_mpp_commit_stage_path_key`
  flushed only its own process until Sep 2026, so a commit from a region
  reached nothing whenever the file lived on the peer; it now broadcasts
  Action 3 like the origin (`commitPathKeyOnAllWorkers`).
- Under `LIBOMPFILE_WRITETHROUGH_FSYNC_POLICY=close` a write-through write's
  `fdatasync` waits in `OmpFileUnsyncedWriteFds` for the next commit or the
  logical close; the commit handler drains it (`syncDeferredWritesForPathKey`).
  Anything new that promises durability must drain that set too.
- An `OMPFILE_PWRITE` event sends its payload with a blocking send
  (`sendInBatchsBlocking` in `ompfilePwrite`, `EventSystem.cpp`, outside
  `ompfileWithMPICallLock`) on whichever thread resumes it, after the header
  and before the completion receives. The Sep 2026 split-phase seam
  (`ompfile_mpp_pwrite_submit` / `ompfile_mpp_poll_ex`) and the suspendable
  payload send (`LIBOMPFILE_MPP_ASYNC_PWRITE_SEND`) were removed in Oct 2026:
  a split-phase write freed no thread, because the resuming thread ran the
  send, and an `MPI_Isend` progresses only while some thread is inside MPI,
  so the polled send was slower than the blocking one. MPI progress, not a
  seam, is the missing piece for several writes in flight.
- `offload/src/exports` (the origin's version script) must list exactly the
  `ompfile_mpp_*` entrypoints `rtl.cpp` defines: a listed name with no
  definition, or a definition `dlsym` cannot reach, is the Sep 2026 export
  mismatch again. The proxy-only rows (`handle_owner_rank`,
  `pread_no_stage_ex` and the three `dirty_owner_*_ex`) have no origin
  definition or export (`rtl.cpp` defines exactly the 15 exported rows), so
  an origin caller takes the shim fallback documented in
  `ompfile_mpp_abi.h`; giving one an origin definition and export is a
  behavior change, not a cleanup. The gate checks the built
  artifact (`nm -D libomptarget.so`) against its own copy of the list,
  `MPP_LOCAL_EXPORTS` in `application/lib/mpp_local.sh`
  (`mpp_local_check_exports`), so the exports file, the `rtl.cpp` definitions
  and that array change together.
- `ProxyDevice::isPluginCompatible` only asks each plugin
  `is_plugin_compatible`; it must not move plugins out of, or reorder,
  `PluginManager.Plugins` (port of OMPC `5b27d1e59`, Oct 2026).
- Which threads can block on a peer: the owner-forwarding helpers
  (`openOnRank`, `preadOnRank`, `pwriteOnRank`, `closeOnRank`) run only on
  kernel threads via the `mpp*` bridges. An I/O-handler coroutine reaches
  `waitForEvent` only through the headnode coherence helpers
  (`freshnessQueryOnHeadnode`, `freshnessWriteCommitOnHeadnode`,
  `completeDirtyFlushOnHeadnode`, also via
  `completeDerivedTileFlushesOnHeadnode`), i.e. under write-back staging on a
  non-headnode rank. Keep it that way, or make the new wait a `co_await`.
- The coherent read refresh (`refreshTrackedOmpFileFdForRead`: re-open +
  `dup2` over the shared `O_RDWR` fd before a read) closes the old open file
  description, and NFS flushes the inode's dirty pages on every close. With
  unsynced writes outstanding (fsync policy `close`, no commit yet) every
  restore read pays the previous store's write-back - 1.8 s per proxy over
  320 reads on the rtm lane, the whole "restores slower after a close-policy
  flush run" term. `LIBOMPFILE_OPT_COHERENT_READ_REFRESH=0` skips it; safe
  only when every writer of a file is its owner proxy. Anything that wants
  cheaper revalidation must avoid closing a descriptor on a dirty inode.

## Debug checklist

- If `ActiveMPIPlugin is null`, confirm which process loads this code path
  and whether plugin init is valid in that role.
- If scheduling seems ignored, check that `LIBOMPFILE_SCHEDULER` is not
  `LOCAL` in the process environment (unset means HEADNODE since Oct 2026).
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
