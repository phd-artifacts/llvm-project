#ifndef OMPFILE_MPP_ABI_H
#define OMPFILE_MPP_ABI_H

// The C ABI between libompfile's MPP shim (mpp_shim.cpp, which resolves every
// entrypoint with dlsym) and the MPI plugin's two runtime bridges: rtl.cpp on
// the origin rank (omptarget.rtl.mpi) and ProxyDevice.cpp on a proxy rank
// (llvm-offload-mpi-proxy-device). The test hook
// ompfile_freshness_query_proxy_test_hook.cpp stubs a subset for the
// out-of-tree regression apps.
//
// Every entrypoint is one row of OMPFILE_MPP_ENTRYPOINTS. The shim generates
// its function-pointer table and its dlsym loader from the rows, and both
// bridges define their functions behind the extern "C" prototypes declared
// below, so a signature that drifts between the sides is a compile error in
// the TU that drifted, not a silent mismatch at run time.
//
// Adding an entrypoint: one row here, then the definition in rtl.cpp (origin
// side) and/or ProxyDevice.cpp (proxy side). A side that does not define a
// row yields a null pointer to the shim there. Five rows are proxy-only:
// handle_owner_rank, pread_no_stage_ex, dirty_owner_pread_ex,
// dirty_owner_pread_batch_ex and dirty_owner_query_ex. The shim treats each
// as optional: preadNoStageEx falls back to preadEx, the others fail with
// ENOSYS. The origin's offload/src/exports must list exactly the rows rtl.cpp
// defines.

#include "ompfile_sched.h"

#include <cstdint>
#include <type_traits>

namespace ompfile {

// Segment descriptor of the dirty-owner batched pread bridge call. One wire
// layout shared by the shim, both bridges and the proxy's reply framing.
struct OmpFileDirtyOwnerPreadBatchSegment {
  int64_t Offset = 0;
  uint64_t Size = 0;
  uint64_t ExpectedVersion = 0;
  uint64_t ClientSegmentId = 0;
};

static_assert(std::is_standard_layout_v<OmpFileDirtyOwnerPreadBatchSegment>);
static_assert(std::is_trivially_copyable_v<OmpFileDirtyOwnerPreadBatchSegment>);

} // namespace ompfile

// X(name, return_type, parameter_list); the entrypoint symbol is
// ompfile_mpp_<name>. Row order is the order of the shim's table.
#define OMPFILE_MPP_ENTRYPOINTS(X)                                            \
  X(init, int, ())                                                            \
  X(submit, int, (uint64_t Token))                                            \
  X(open, int, (const char *Path, int Flags, int Mode, int *Handle))          \
  X(open_on_rank, int,                                                        \
    (const char *Path, int Flags, int Mode, int Rank, int *Handle))           \
  X(handle_owner_rank, int, (int Handle, int *RankOut))                       \
  X(close, int, (int Handle))                                                 \
  X(pread_ex, int,                                                            \
    (int Handle, int64_t Offset, void *Buffer, uint64_t Size,                 \
     uint64_t *BytesRead))                                                    \
  X(pread_no_stage_ex, int,                                                   \
    (int Handle, int64_t Offset, void *Buffer, uint64_t Size,                 \
     uint64_t *BytesRead))                                                    \
  X(dirty_owner_pread_ex, int,                                                \
    (int Handle, int SourceRank, uint64_t ExpectedVersion, int64_t Offset,    \
     void *Buffer, uint64_t Size, uint64_t *BytesRead))                       \
  X(dirty_owner_pread_batch_ex, int,                                          \
    (int Handle, int SourceRank,                                              \
     const ompfile::OmpFileDirtyOwnerPreadBatchSegment *Segments,             \
     uint64_t SegmentCount, void *const *Buffers, uint64_t *BytesRead,        \
     int *Statuses, int *Errnos))                                             \
  X(dirty_owner_query_ex, int,                                                \
    (int Handle, int SourceRank, uint64_t ExpectedVersion, int64_t Offset,    \
     uint64_t Size, int *StateOut))                                           \
  X(pwrite_ex, int,                                                           \
    (int Handle, int64_t Offset, const void *Buffer, uint64_t Size,           \
     uint64_t *BytesWritten))                                                 \
  X(freshness_query, int,                                                     \
    (const ompfile::OmpFileFreshnessQueryRequest *Request,                    \
     ompfile::OmpFileFreshnessQueryReply *Reply))                             \
  X(freshness_write_commit, int,                                              \
    (uint64_t PathKey, int WriterRank, uint64_t TileId, int WriteThroughMode, \
     uint64_t *CommittedVersion))                                             \
  X(flush_dirty_tile, int,                                                    \
    (uint64_t PathKey, int *SourceRank, uint64_t *FlushedVersion))            \
  X(commit_stage_path_key, int, (uint64_t PathKey))                           \
  X(sched_request, int,                                                       \
    (const ompfile::OmpFileIORequest *Request, const char *Path,              \
     ompfile::OmpFileIOPlan *Plan))                                           \
  X(sched_request_batch, int,                                                 \
    (const ompfile::OmpFileIOBatchRequest *Request,                           \
     const void *RequestPayload, uint64_t RequestPayloadBytes,                \
     ompfile::OmpFileIOBatchPlan *Plan, void *PlanPayload,                    \
     uint64_t PlanPayloadCapBytes, uint64_t *PlanPayloadOutBytes))            \
  X(poll, int, (uint64_t Token, int *Done))                                   \
  X(finalize, int, ())

extern "C" {
#define OMPFILE_MPP_DECLARE_ENTRYPOINT(name, ret, params)                    \
  ret ompfile_mpp_##name params;
OMPFILE_MPP_ENTRYPOINTS(OMPFILE_MPP_DECLARE_ENTRYPOINT)
#undef OMPFILE_MPP_DECLARE_ENTRYPOINT
} // extern "C"

#endif // OMPFILE_MPP_ABI_H
