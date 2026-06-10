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

## Event model

- OMPFile events: `OMPFILE_OPEN`, `OMPFILE_CLOSE`, `OMPFILE_PREAD`,
  `OMPFILE_PWRITE`, `OMPFILE_PING`
- Scheduler events: `OMPFILE_SCHED_REQUEST`, `OMPFILE_SCHED_PLAN`
- `ProxyDevice` creates rank-targeted events and waits for completion.

## Scheduler behavior

- `LIBOMPFILE_SCHEDULER=HEADNODE` triggers scheduler request on open.
- `OmpFileHeadnodeManager` keeps:
  - handler table (rank + in-flight load)
  - global file table (`path -> preferred aggregator rank`)
  - flightplan table (request tracking)
- If scheduler request fails, current behavior falls back to local rank
  selection path.

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
