#include "mpi_io_backend.h"

#include <cassert>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <vector>

uint64_t MPIIOBackend::computeSourceStageAffinityKey(
    uint64_t path_key, const ompfile::OmpFileIOHint &hint) {
  uint64_t key = path_key ^ 0x9e3779b97f4a7c15ULL;
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_STREAM) != 0)
    key ^= hint.StreamId + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
  key ^= hint.TileId + 0x9e3779b97f4a7c15ULL + (key << 6) + (key >> 2);
  return key;
}

bool MPIIOBackend::selectSourceStageAffinityRank(
    uint64_t path_key, const ompfile::OmpFileIOHint &hint,
    const std::vector<int> &worker_ranks, int &rank_out) {
  rank_out = -1;
  if (path_key == 0 || worker_ranks.empty())
    return false;
  const uint64_t key = computeSourceStageAffinityKey(path_key, hint);
  rank_out = worker_ranks[static_cast<size_t>(key % worker_ranks.size())];
  if (rank_out < 0) {
    rank_out = -1;
    return false;
  }
  return true;
}

std::vector<int> MPIIOBackend::computeWorkerRanks(int world_size,
                                                  int headnode_rank) {
  std::vector<int> ranks;
  if (world_size <= 0)
    return ranks;
  for (int rank = 0; rank < world_size; ++rank) {
    if (rank != headnode_rank)
      ranks.push_back(rank);
  }
  return ranks;
}

bool MPIIOBackend::isForecastHintValid(
    const ompfile::OmpFileIOHint &hint) const {
  if (forecast_mode != ForecastMode::CholeskyOracle)
    return false;
  return (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0;
}

bool MPIIOBackend::isForecastRoleHintValid(
    const ompfile::OmpFileIOHint &hint) const {
  return isForecastHintValid(hint) &&
         (hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_ROLE) != 0;
}

bool MPIIOBackend::shouldBypassStageForForecastRead(
    const ompfile::OmpFileIOHint &hint) const {
  return isForecastRoleHintValid(hint) &&
         hint.Role == ompfile::OMPFILE_IO_ROLE_TARGET_READ;
}

bool MPIIOBackend::shouldUseNoStageFallback(const ompfile::OmpFileIOHint &hint,
                                            bool force_no_stage) const {
  return force_no_stage || shouldBypassStageForForecastRead(hint);
}

bool MPIIOBackend::shouldAttemptPlannerRebalancedRead(
    int rc, bool force_no_stage_fallback, bool planner_rebalanced,
    int planned_rank) const {
  return rc != 0 && !force_no_stage_fallback && planner_rebalanced &&
         planned_rank >= 0;
}

bool MPIIOBackend::isSourceStageAffinityEligible(
    const ompfile::OmpFileIOHint &hint, bool has_path_key,
    bool remote_only) const {
  if (!remote_only || !has_path_key)
    return false;
  if (forecast_mode != ForecastMode::CholeskyOracle ||
      stage_affinity_mode != StageAffinityMode::CholeskyOracle)
    return false;
  const char *stage_mode = std::getenv("LIBOMPFILE_STAGE_MODE");
  if (!stage_mode || std::strcmp(stage_mode, "readthrough") != 0)
    return false;
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_ROLE) == 0 ||
      hint.Role != ompfile::OMPFILE_IO_ROLE_SOURCE_READ)
    return false;
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) == 0)
    return false;
  return true;
}

bool MPIIOBackend::canAttemptSourceStageAffinityRead(
    const ompfile::OmpFileIOHint &hint, int file_id, long start, size_t size,
    bool has_path_key, bool remote_only, bool planner_force_fallback) {
  if (planner_force_fallback)
    return false;
  if (!isSourceStageAffinityEligible(hint, has_path_key, remote_only))
    return false;
  const char *reason = "none";
  int reason_errno = 0;
  return canApplyRebalancedRead(hint, file_id, start, size, reason,
                                reason_errno, /*record_stats=*/false);
}

void MPIIOBackend::recordForecastRead(const ompfile::OmpFileIOHint &hint,
                                      size_t size) {
  if (!isForecastHintValid(hint))
    return;
  forecast_hint_read_count.fetch_add(1, std::memory_order_relaxed);
  forecast_hint_read_bytes_total.fetch_add(size, std::memory_order_relaxed);
  if (!isForecastRoleHintValid(hint)) {
    forecast_role_unknown_count.fetch_add(1, std::memory_order_relaxed);
    forecast_role_unknown_bytes_total.fetch_add(size,
                                                std::memory_order_relaxed);
    return;
  }
  if (hint.Role == ompfile::OMPFILE_IO_ROLE_SOURCE_READ) {
    forecast_source_read_stageable_count.fetch_add(1,
                                                   std::memory_order_relaxed);
    forecast_source_read_stageable_bytes_total.fetch_add(
        size, std::memory_order_relaxed);
    return;
  }
  if (shouldBypassStageForForecastRead(hint)) {
    forecast_target_read_bypass_count.fetch_add(1, std::memory_order_relaxed);
    forecast_target_read_bypass_bytes_total.fetch_add(
        size, std::memory_order_relaxed);
    return;
  }
  forecast_role_unknown_count.fetch_add(1, std::memory_order_relaxed);
  forecast_role_unknown_bytes_total.fetch_add(size, std::memory_order_relaxed);
}

void MPIIOBackend::recordForecastWrite(const ompfile::OmpFileIOHint &hint,
                                       size_t size) {
  if (!isForecastHintValid(hint))
    return;
  forecast_hint_write_count.fetch_add(1, std::memory_order_relaxed);
  forecast_hint_write_bytes_total.fetch_add(size, std::memory_order_relaxed);
}

bool MPIIOBackend::readAtSourceStageAffinityRankWithBytes(
    int file_id, uint64_t path_key, const ompfile::OmpFileIOHint &hint,
    long offset, void *data, size_t size, size_t &bytes_read, int &rank_out) {
  rank_out = -1;
  int finalized = 0;
  if (MPI_Finalized(&finalized) != MPI_SUCCESS || finalized)
    return false;
  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS)
    return false;
  if (!initialized)
    return false;
  int world_size = 0;
  if (MPI_Comm_size(MPI_COMM_WORLD, &world_size) != MPI_SUCCESS)
    return false;
  const int headnode_rank = world_size > 0 ? world_size - 1 : -1;
  const std::vector<int> workers =
      computeWorkerRanks(world_size, headnode_rank);
  if (!selectSourceStageAffinityRank(path_key, hint, workers, rank_out))
    return false;
  return readAtRemoteRankWithBytes(file_id, rank_out, offset, data, size,
                                   bytes_read);
}

size_t MPIIOBackend::computeForecastTargetReadSize(
    const ompfile::OmpFileIOHint &hint, long start, long end,
    size_t request_count, bool remote_only, uint64_t sieve_bytes,
    uint64_t max_batch_bytes) const {
  assert(start >= 0 && end >= start &&
         "Forecast read range must be monotonic.");
  if (isForecastHintValid(hint))
    return static_cast<size_t>(end - start);
  return computeTwoPhaseTargetReadSize(start, end, request_count, remote_only,
                                       sieve_bytes, max_batch_bytes);
}
