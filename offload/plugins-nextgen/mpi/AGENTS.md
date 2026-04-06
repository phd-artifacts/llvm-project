# Agent Notes: MPI Plugin (offload/plugins-nextgen/mpi)

This scope covers MPP event routing, scheduler selection, and proxy-side I/O dispatch.

## Scope intent
- Keep this file focused on MPI-plugin runtime behavior.
- Shared commit workflow, submodule hygiene, and MkDocs maintenance are canonical in root `AGENTS.md`.

## Direction for MPP work
- Treat MPP as the distributed execution substrate for runtime-managed I/O:
  event transport, proxy execution, scheduler communication, and remote-worker
  mechanics belong here.
- Avoid turning this layer into the public semantic home for portable I/O
  policy; semantics should stay in `libompfile` unless the change is truly
  transport-specific.

## Relevant files
- Runtime bridge exports:
  - `src/rtl.cpp`
  - `src/ProxyDevice.cpp` (`extern "C" ompfile_mpp_*` wrappers for proxy runtime)
- Headnode scheduler:
  - `src/OmpFileHeadnodeManager.h`
  - `src/OmpFileHeadnodeManager.cpp`
- Event dispatch:
  - `event_system/EventSystem.h`
  - `event_system/EventSystem.cpp`

## Event model
- OMPFile events include:
  - `OMPFILE_OPEN`, `OMPFILE_CLOSE`, `OMPFILE_PREAD`, `OMPFILE_PWRITE`, `OMPFILE_PING`
  - scheduler events: `OMPFILE_SCHED_REQUEST`, `OMPFILE_SCHED_PLAN`
- `ProxyDevice` creates rank-targeted events and waits for completion.

## Scheduler behavior
- `LIBOMPFILE_SCHEDULER=HEADNODE` triggers scheduler request on open.
- `OmpFileHeadnodeManager` keeps:
  - handler table (rank + in-flight load)
  - global file table (`path -> preferred aggregator rank`)
  - flightplan table (request tracking)
- If scheduler request fails, current behavior falls back to local rank selection path.

## Ownership and handles
- `mppOpen` picks an aggregator rank, opens there, and stores local-handle -> {rank, remote_handle}.
- `mppPread`/`mppPwrite` route via that stored rank.
- `mppClose` closes on that stored rank and removes handle mapping.

## Debug checklist
- If `ActiveMPIPlugin is null`, confirm which process loads this code path and whether plugin init is valid in that role.
- If scheduling seems ignored, verify `LIBOMPFILE_SCHEDULER=HEADNODE` in the process environment.
- Use `LIBOMPFILE_OPT_STATS=1` to inspect open/close cache counters printed by proxy.

## Skills entrypoints
- `skills/submodule-commit-flow/SKILL.md`
- `skills/mkdocs-sync/SKILL.md`
