# Agent Notes: libompfile (openmp/libompfile)

This scope covers libompfile backend behavior and MPP shim usage.

## Relevant files
- Backend logic:
  - `src/mpi_io_backend.cpp`
  - `src/mpi_io_backend.h`
- Public interface glue:
  - `src/file_interface.cpp`
  - `include/file_interface.h`
- MPP shim:
  - `src/mpp_shim.cpp`
  - `src/mpp_shim.h`
- Scheduler structs:
  - `include/ompfile_sched.h`

## Backend modes
- MPI backend selected by `LIBOMPFILE_BACKEND=MPI`.
- Remote-only MPP mode is active when both are true:
  - `LIBOMPFILE_MPP_OPEN=1`
  - `LIBOMPFILE_MPP_IO=1`
- In remote-only mode, file operations use `ompfile::mpp::*` path and skip MPI file communicator duplication.

## MPP shim behavior
- `mpp_shim` uses `dlsym(RTLD_DEFAULT, "ompfile_mpp_*")` to resolve runtime bridge symbols.
- If symbols are missing or init fails, operations fail and log explicit diagnostics.
- Scheduler requests use `ompfile_mpp_sched_request` via shim API.

## Handle model
- Backend assigns local logical file ids.
- For MPP open path, local id maps to remote handle tracked in `remote_file_handle_map`.
- `readAt/writeAt` use mapped remote handle in MPP I/O mode.

## Useful knobs
- `LIBOMPFILE_MPP_PING=1` for health check.
- `LIBOMPFILE_MPI_COMM_SELF=1` to force `MPI_COMM_SELF` in non-remote-only mode.
- `LIBOMPFILE_SCHEDULER=HEADNODE` to request headnode scheduling policy.
- `LIBOMPFILE_OPT_OPEN_CACHE`, `LIBOMPFILE_OPT_OPEN_CACHE_KEEP_OPEN`, `LIBOMPFILE_OPT_STATS` influence proxy-side behavior.

## Failure triage
- `MPP shim init failed`: check runtime symbol export path and plugin initialization.
- Open/read errors with MPP enabled: inspect both libompfile logs and proxy/EventSystem logs together.

## Defensive checks
- When modifying libompfile C/C++ code, add `assert(...)` around internal bounds
  invariants (file-id ranges, map/vector index use, offset/size arithmetic).
- Keep explicit error-path checks for external/runtime failures (`errno`/return
  codes); do not rely on asserts for recoverable conditions.

## Commit instructions
- Commit message format must follow Rodrigo Ceccato style:
  - subject line: `[scope]: short message`
  - scope describes the implementation change, not the subfolder name.
  - preferred scopes in this area: `mpp-libompfile`, `runtime`, `tests`, `docs`.
  - use `[docs]:` only for docs-only commits.
  - body: concise bullet list with what changed and why.
  - do not use `Step 1:`, `Step 2:` style.
  - preferred command form:
    - `git commit -m "[mpp-libompfile]: short message" -m "- bullet 1\n- bullet 2\n\nWhy:\n- ..."`
- Commit libompfile code changes in this submodule first:
  - `cd /scratch/rodrigo.freitas/io-playground/llvm-infra/llvm-project`
  - `git add openmp/libompfile/...`
  - `git commit -m "[mpp-libompfile]: short message" -m "- bullet 1\n- bullet 2\n\nWhy:\n- ..."`
- Then update the superproject pointer:
  - `cd /scratch/rodrigo.freitas/io-playground`
  - `git add llvm-infra/llvm-project`
  - `git commit -m "[runtime]: short message" -m "- bullet 1\n- bullet 2\n\nWhy:\n- ..."`

## MkDocs sync requirements
- Any libompfile runtime behavior change must update MkDocs in the same workstream.
- Keep `docs/roadmap.md` marked with:
  - current phase/stage
  - limitations and instability conditions
  - observed speedups/slowdowns on sorgan
  - current/expected challenges
- Keep architecture and execution diagrams aligned with current code paths.
- Ensure docs always cover:
  - disk access pattern description
  - execution flow description
  - software architecture description
