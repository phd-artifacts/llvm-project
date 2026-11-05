#include "debug_log.h"
#include "mpp_backend.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <limits>

uint64_t MppBackend::resolveTwoPhaseKey(
    const ompfile::OmpFileReadRequestContext &context) {
  uint64_t base_key = 0;
  if ((context.ContextFlags & ompfile::OMPFILE_READ_CTX_HAS_PATH_KEY) != 0) {
    base_key = context.PathKey;
    return ompfile::keys::readGroupKey(base_key, context.Hint);
  }

  uint64_t path_key = 0;
  if (getFilePathKey(context.FileHandle, path_key))
    base_key = path_key;
  else
    base_key = ompfile::keys::handleKey(context.FileHandle);
  return ompfile::keys::readGroupKey(base_key, context.Hint);
}

int MppBackend::failStrictMpp(const char *op_name) const {
  errno = EIO;
  if (!strict_failure_logged.exchange(true, std::memory_order_relaxed)) {
    io_log("Error: '%s' failed because libompfile initialization did not "
           "complete (MPP init failed or a removed mode was requested); "
           "remote-only MPP is the only mode.\n",
           op_name ? op_name : "(unknown)");
  }
  io_trace("MPIIOBackend::failStrictMpp this=%p op=%s "
           "strict_mpp_init_failed=%d\n",
           static_cast<const void *>(this), op_name ? op_name : "(unknown)",
           static_cast<int>(settings.strict_mpp_init_failed));
  return -1;
}

size_t MppBackend::computeTwoPhaseTargetReadSize(long start, long end,
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
