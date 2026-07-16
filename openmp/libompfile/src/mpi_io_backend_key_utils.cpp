#include "debug_log.h"
#include "mpi_io_backend.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <limits>

uint64_t MPIIOBackend::computePathKey(const char *path) {
  constexpr uint64_t offset_basis = 1469598103934665603ULL;
  constexpr uint64_t prime = 1099511628211ULL;
  uint64_t hash = offset_basis;
  if (!path)
    return hash;

  for (const unsigned char *p = reinterpret_cast<const unsigned char *>(path);
       *p != '\0'; ++p) {
    hash ^= static_cast<uint64_t>(*p);
    hash *= prime;
  }
  return hash;
}

uint64_t MPIIOBackend::mixHintIntoKey(uint64_t base_key,
                                      const ompfile::OmpFileIOHint &hint) {
  uint64_t mixed = base_key;
  auto mix = [&mixed](uint64_t value) {
    mixed ^= value + 0x9e3779b97f4a7c15ULL + (mixed << 6) + (mixed >> 2);
  };

  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_STREAM) != 0)
    mix(hint.StreamId);
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_EPOCH) != 0)
    mix(hint.EpochId);
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0)
    mix(hint.TileId);
  return mixed;
}

uint64_t MPIIOBackend::mixWriteHintIntoKey(uint64_t base_key,
                                           const ompfile::OmpFileIOHint &hint) {
  uint64_t mixed = base_key;
  auto mix = [&mixed](uint64_t value) {
    mixed ^= value + 0x9e3779b97f4a7c15ULL + (mixed << 6) + (mixed >> 2);
  };

  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_STREAM) != 0)
    mix(hint.StreamId);
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0)
    mix(hint.TileId);
  return mixed;
}

uint64_t MPIIOBackend::resolveTwoPhaseKey(
    const ompfile::OmpFileReadRequestContext &context) {
  uint64_t base_key = 0;
  if ((context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY) != 0) {
    base_key = context.PathKey;
    return mixHintIntoKey(base_key, context.Hint);
  }

  uint64_t path_key = 0;
  if (getFilePathKey(context.FileHandle, path_key))
    base_key = path_key;
  else
    base_key = static_cast<uint64_t>(static_cast<uint32_t>(context.FileHandle));
  return mixHintIntoKey(base_key, context.Hint);
}

int MPIIOBackend::failStrictMpp(const char *op_name) const {
  errno = EIO;
  if (!strict_failure_logged.exchange(true, std::memory_order_relaxed)) {
    io_log("Error: '%s' failed because MPP init did not complete and fallback "
           "is disabled. Set LIBOMPFILE_ALLOWFALLBACK=1 to allow fallback "
           "behavior.\n",
           op_name ? op_name : "(unknown)");
  }
  io_trace("MPIIOBackend::failStrictMpp this=%p op=%s mpp_requested=%d "
           "allow_fallback=%d strict_mpp_init_failed=%d\n",
           static_cast<const void *>(this), op_name ? op_name : "(unknown)",
           static_cast<int>(mpp_requested), static_cast<int>(allow_fallback),
           static_cast<int>(strict_mpp_init_failed));
  return -1;
}

void MPIIOBackend::updateAtomicMax(std::atomic<uint64_t> &counter,
                                   uint64_t candidate) {
  uint64_t observed = counter.load(std::memory_order_relaxed);
  while (candidate > observed &&
         !counter.compare_exchange_weak(observed, candidate,
                                        std::memory_order_relaxed,
                                        std::memory_order_relaxed)) {
  }
}

size_t MPIIOBackend::computeTwoPhaseTargetReadSize(long start, long end,
                                                   size_t request_count,
                                                   bool remote_only,
                                                   uint64_t sieve_bytes,
                                                   uint64_t max_batch_bytes) {
  if (start < 0 || end < start) {
    io_log("Error: computeTwoPhaseTargetReadSize received a non-monotonic "
           "range start=%ld end=%ld; refusing to compute a target size.\n",
           start, end);
    return 0;
  }
  const size_t read_size = static_cast<size_t>(end - start);
  if (sieve_bytes == 0)
    return read_size;

  // Remote-only MPP intentionally does not cache two-phase overreads across
  // proxies, so data sieving isolated requests only adds transfer volume.
  if (remote_only && request_count <= 1)
    return read_size;

  uint64_t target_size =
      std::max<uint64_t>(static_cast<uint64_t>(read_size), sieve_bytes);
  if (max_batch_bytes > 0)
    target_size = std::min<uint64_t>(target_size, max_batch_bytes);

  const uint64_t size_t_max =
      static_cast<uint64_t>(std::numeric_limits<size_t>::max());
  target_size = std::min<uint64_t>(target_size, size_t_max);
  const uint64_t long_max =
      static_cast<uint64_t>(std::numeric_limits<long>::max());
  const uint64_t start_u64 = static_cast<uint64_t>(start);
  if (start_u64 >= long_max)
    return read_size;
  const uint64_t max_extent = long_max - start_u64;
  target_size = std::min<uint64_t>(target_size, max_extent);

  if (target_size < static_cast<uint64_t>(read_size))
    return read_size;
  return static_cast<size_t>(target_size);
}
