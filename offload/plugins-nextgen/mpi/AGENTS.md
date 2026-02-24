# Agent Notes: MPI Plugin (offload/plugins-nextgen/mpi)

This scope covers MPP event routing, scheduler selection, and proxy-side I/O dispatch.

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

## Commit instructions
- Commit MPI plugin changes in this submodule first:
  - `cd /scratch/rodrigo.freitas/io-playground/llvm-infra/llvm-project`
  - `git add offload/plugins-nextgen/mpi/...`
  - `git commit -m "mpi-plugin: ..."`
- Then update the superproject pointer:
  - `cd /scratch/rodrigo.freitas/io-playground`
  - `git add llvm-infra/llvm-project`
  - `git commit -m "chore(submodule): bump llvm-project"`
