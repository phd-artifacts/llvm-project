#include "mpi_io_backend.h"
#include "debug_log.h"

#include <cstring>
#include <mutex>
#include <vector>

bool MPIIOBackend::tryServeTwoPhaseReadCache(uint64_t key, long offset,
                                             void *data, size_t size) {
  if (!data || size == 0)
    return false;

  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  const auto it = two_phase_read_cache.find(key);
  if (it == two_phase_read_cache.end())
    return false;

  const TwoPhaseReadCacheEntry &entry = it->second;
  if (offset < entry.Start)
    return false;
  const uint64_t delta = static_cast<uint64_t>(offset - entry.Start);
  if (delta > static_cast<uint64_t>(entry.Data.size()))
    return false;
  if (size > entry.Data.size() - static_cast<size_t>(delta))
    return false;

  std::memcpy(data, entry.Data.data() + static_cast<size_t>(delta), size);
  two_phase_cache_hit_count.fetch_add(1, std::memory_order_relaxed);
  return true;
}

void MPIIOBackend::updateTwoPhaseReadCache(uint64_t key, long start,
                                           const std::vector<char> &data) {
  if (data.empty())
    return;

  TwoPhaseReadCacheEntry entry{};
  entry.Start = start;
  entry.Data = data;

  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  two_phase_read_cache[key] = std::move(entry);
}

void MPIIOBackend::invalidateTwoPhaseReadCacheKey(uint64_t key) {
  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  two_phase_read_cache.erase(key);
}

void MPIIOBackend::invalidateTwoPhaseReadCacheForFile(int file_id) {
  uint64_t path_key = 0;
  const bool has_path_key = getFilePathKey(file_id, path_key);
  const uint64_t fallback_key =
      static_cast<uint64_t>(static_cast<uint32_t>(file_id));
  const std::lock_guard<std::mutex> lock(two_phase_mutex);
  const size_t cleared_entries = two_phase_read_cache.size();
  if (cleared_entries == 0)
    return;

  // Two-phase cache entries are stored under hint-mixed keys. Once writes happen,
  // the current cache key space is not invertible back to every mixed variant for
  // the file, so clear conservatively instead of serving stale read-after-write
  // data under an old epoch/stream/tile key.
  two_phase_read_cache.clear();
  io_trace("MPIIOBackend::invalidateTwoPhaseReadCacheForFile file_id=%d "
           "has_path_key=%d path_key=%llu fallback_key=%llu cleared=%zu\n",
           file_id, static_cast<int>(has_path_key),
           static_cast<unsigned long long>(path_key),
           static_cast<unsigned long long>(fallback_key), cleared_entries);
}
