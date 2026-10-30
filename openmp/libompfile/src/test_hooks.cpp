// Test-only entry points (include/ompfile_test_hooks.h). Not public API; see
// the header for why they exist and how their absence from file_interface.h
// is checked.

// ompfile_sched.h (via client_context.h) must come before file_interface.h
// (via ompfile_test_hooks.h): file_interface.h #defines
// OMPFILE_IO_HINT_ABI_VERSION, the name of a constexpr in ompfile_sched.h,
// so the macro must not exist yet when ompfile_sched.h is parsed. Every other
// TU includes them in this order.
#include "client_context.h"
#include "debug_log.h"
#include "mpp_backend.h"
#include "ompfile_sched.h"
#include "mpi.h"

#include "ompfile_test_hooks.h"

#include <cerrno>
#include <cstdint>

extern "C" int ompfile_test_open_relative(const char *path, int peer,
                                          int *writer_rank, int *owner_rank) {
  if (writer_rank) *writer_rank = -1;
  if (owner_rank) *owner_rank = -1;
  if (!path || !*path || !writer_rank || !owner_rank ||
      (peer != 0 && peer != 1)) {
    errno = EINVAL;
    return -1;
  }
  int initialized = 0, finalized = 0, rank = -1, size = 0;
  MPI_Initialized(&initialized);
  if (initialized) MPI_Finalized(&finalized);
  if (!initialized || finalized) { errno = EHOSTDOWN; return -1; }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 3 || rank < 0 || rank >= size - 1) {
    errno = EINVAL;
    return -1;
  }
  const int requested = peer ? (rank + 1) % (size - 1) : rank;
  struct ScopedOwner {
    int Previous = OmpFileTestOpenOwnerRank;
    explicit ScopedOwner(int rank) { OmpFileTestOpenOwnerRank = rank; }
    ~ScopedOwner() { OmpFileTestOpenOwnerRank = Previous; }
  } scope(requested);
  auto &ctx = OmpFileClientContext::getInstance();
  const int handle = ctx.openFile(path);
  if (handle < 0) return -1;
  auto &backend = static_cast<MppBackend &>(*ctx.io_backend);
  const int actual = backend.lookupRemoteOwnerRank(handle);
  if (actual != requested) {
    (void)ctx.closeFile(handle);
    errno = EPROTO;
    return -1;
  }
  *writer_rank = rank;
  *owner_rank = actual;
  return handle;
}

extern "C" int ompfile_test_pwrite_on_rank(int file_handle, long offset,
                                           const void *data, size_t size,
                                           int rank,
                                           const omp_file_io_hint_v1 *hint) {
  OmpFileClientContext &ctx = OmpFileClientContext::getInstance();
  if (!ctx.isOpenHandle(file_handle)) {
    errno = EBADF;
    return -1;
  }
  if (rank < 0 || offset < 0 || (!data && size > 0)) {
    errno = EINVAL;
    return -1;
  }
  // Same program order as a synchronous omp_file_pwrite: every queued async
  // write lands first.
  ctx.waitForQueuedWrites();
  OmpFileClientContext::IOResourceGuard guard(ctx.io_resource_token);

  ompfile::OmpFileWriteRequestContext context{};
  context.FileHandle = file_handle;
  context.Offset = static_cast<int64_t>(offset);
  context.Size = static_cast<uint64_t>(size);
  // The conversion client_context.cpp's applyIoHint does: a hint of another
  // ABI version is ignored.
  if (hint && hint->abi_version == OMPFILE_IO_HINT_ABI_VERSION) {
    context.Hint.AbiVersion = hint->abi_version;
    context.Hint.HintFlags = hint->hint_flags;
    context.Hint.EpochId = hint->epoch_id;
    context.Hint.StreamId = hint->stream_id;
    context.Hint.TileId = hint->tile_id;
    context.Hint.Role = hint->role;
  }
  uint64_t path_key = 0;
  // MppBackend is the only backend (OmpFileClientContext builds it), so the
  // downcast is exact.
  MppBackend &backend = static_cast<MppBackend &>(*ctx.io_backend);
  if (backend.getFilePathKey(file_handle, path_key)) {
    context.PathKey = path_key;
    context.ContextFlags |= ompfile::OMPFILE_WRITE_CTX_HAS_PATH_KEY;
  }
  io_log("Test placement hook: pwrite file=%d offset=%ld size=%zu rank=%d\n",
         file_handle, offset, size, rank);
  return backend.writeAtPlacedRank(context, rank, data, size);
}
