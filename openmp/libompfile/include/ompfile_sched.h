#ifndef OMPFILE_SCHED_H
#define OMPFILE_SCHED_H

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

namespace ompfile {

constexpr uint32_t OMPFILE_SCHED_BATCH_ABI_VERSION = 1u;
constexpr uint32_t OMPFILE_IO_HINT_ABI_VERSION = 1u;

enum OmpFileIOHintFlags : uint32_t {
  OMPFILE_IO_HINT_HAS_EPOCH = 1u << 0,
  OMPFILE_IO_HINT_HAS_STREAM = 1u << 1,
  OMPFILE_IO_HINT_HAS_TILE = 1u << 2,
};

struct OmpFileIOHint {
  uint32_t AbiVersion = OMPFILE_IO_HINT_ABI_VERSION;
  uint32_t HintFlags = 0;
  uint64_t EpochId = 0;
  uint64_t StreamId = 0;
  uint64_t TileId = 0;
};

enum class OmpFileIOOp : uint32_t {
  OPEN = 0,
  CLOSE = 1,
  PREAD = 2,
  PWRITE = 3,
  PREFETCH = 4,
};

struct OmpFileIORequest {
  uint64_t RequestId = 0;
  OmpFileIOOp Op = OmpFileIOOp::OPEN;
  int32_t FileHandle = -1;
  int32_t Flags = 0;
  int32_t Mode = 0;
  int32_t ClientRank = -1;
  int64_t Offset = 0;
  uint64_t Size = 0;
  uint32_t PathSize = 0;
  uint32_t HintFlags = 0;
  uint64_t EpochId = 0;
  uint64_t StreamId = 0;
  uint64_t TileId = 0;
};

struct OmpFileIOPlan {
  uint64_t RequestId = 0;
  int32_t AggregatorRank = -1;
  int32_t RemoteHandle = -1;
  int32_t Status = 0;
  int32_t Errno = 0;
  int64_t Offset = 0;
  uint64_t Size = 0;
  uint32_t PlanFlags = 0;
  uint32_t Reserved = 0;
};

enum OmpFileBatchRequestFlags : uint32_t {
  OMPFILE_BATCH_REQ_FAIL_ON_ANY_ERROR = 1u << 0,
  OMPFILE_BATCH_REQ_DISABLE_SCALAR_FALLBACK = 1u << 1,
};

enum OmpFileBatchPlanFlags : uint32_t {
  OMPFILE_BATCH_PLAN_BATCH_API = 1u << 0,
  OMPFILE_BATCH_PLAN_SCALAR_FALLBACK = 1u << 1,
  OMPFILE_BATCH_PLAN_FILE_AFFINITY = 1u << 2,
  OMPFILE_BATCH_PLAN_REBALANCED = 1u << 3,
};

struct OmpFileIOBatchSegment {
  uint64_t SegmentId = 0;
  int32_t FileHandle = -1;
  int32_t ClientRank = -1;
  int64_t Offset = 0;
  uint64_t Size = 0;
  uint64_t PathKey = 0;
  uint32_t SegmentFlags = 0;
  uint32_t Reserved = 0;
  uint64_t EpochId = 0;
  uint64_t StreamId = 0;
  uint64_t TileId = 0;
};

struct OmpFileIOBatchRequest {
  uint32_t AbiVersion = OMPFILE_SCHED_BATCH_ABI_VERSION;
  uint32_t SegmentCount = 0;
  uint32_t RequestFlags = 0;
  uint32_t Reserved0 = 0;
  uint64_t BatchId = 0;
  uint32_t PayloadBytes = 0;
  uint32_t Reserved1 = 0;
};

struct OmpFileIOBatchPlanEntry {
  uint64_t SegmentId = 0;
  int32_t AggregatorRank = -1;
  int32_t RemoteHandle = -1;
  int32_t Status = 0;
  int32_t Errno = 0;
  int64_t Offset = 0;
  uint64_t Size = 0;
  uint32_t PlanFlags = 0;
  uint32_t Reserved = 0;
};

struct OmpFileIOBatchPlan {
  uint32_t AbiVersion = OMPFILE_SCHED_BATCH_ABI_VERSION;
  uint32_t SegmentCount = 0;
  uint32_t PlanFlags = 0;
  uint32_t Reserved0 = 0;
  uint64_t BatchId = 0;
  int32_t Status = 0;
  int32_t Errno = 0;
  uint32_t PayloadBytes = 0;
  uint32_t Reserved1 = 0;
};

static_assert(std::is_standard_layout_v<OmpFileIORequest>);
static_assert(std::is_standard_layout_v<OmpFileIOPlan>);
static_assert(std::is_standard_layout_v<OmpFileIOHint>);
static_assert(std::is_standard_layout_v<OmpFileIOBatchSegment>);
static_assert(std::is_standard_layout_v<OmpFileIOBatchRequest>);
static_assert(std::is_standard_layout_v<OmpFileIOBatchPlanEntry>);
static_assert(std::is_standard_layout_v<OmpFileIOBatchPlan>);

inline uint64_t batchSegmentPayloadBytes(uint32_t segment_count) {
  return static_cast<uint64_t>(segment_count) *
         static_cast<uint64_t>(sizeof(OmpFileIOBatchSegment));
}

inline uint64_t batchPlanPayloadBytes(uint32_t segment_count) {
  return static_cast<uint64_t>(segment_count) *
         static_cast<uint64_t>(sizeof(OmpFileIOBatchPlanEntry));
}

inline bool encodeBatchSegments(const std::vector<OmpFileIOBatchSegment> &segments,
                                std::vector<uint8_t> &payload) {
  const uint64_t bytes = batchSegmentPayloadBytes(
      static_cast<uint32_t>(segments.size()));
  if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return false;
  payload.resize(static_cast<size_t>(bytes));
  if (payload.empty())
    return true;
  std::memcpy(payload.data(), segments.data(), payload.size());
  return true;
}

inline bool decodeBatchSegments(const void *payload, size_t payload_bytes,
                                uint32_t expected_segment_count,
                                std::vector<OmpFileIOBatchSegment> &segments) {
  const uint64_t expected_bytes = batchSegmentPayloadBytes(expected_segment_count);
  if (expected_bytes != static_cast<uint64_t>(payload_bytes))
    return false;
  segments.resize(expected_segment_count);
  if (expected_segment_count == 0)
    return true;
  if (!payload)
    return false;
  std::memcpy(segments.data(), payload, payload_bytes);
  return true;
}

inline bool encodeBatchPlanEntries(
    const std::vector<OmpFileIOBatchPlanEntry> &entries,
    std::vector<uint8_t> &payload) {
  const uint64_t bytes =
      batchPlanPayloadBytes(static_cast<uint32_t>(entries.size()));
  if (bytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
    return false;
  payload.resize(static_cast<size_t>(bytes));
  if (payload.empty())
    return true;
  std::memcpy(payload.data(), entries.data(), payload.size());
  return true;
}

inline bool decodeBatchPlanEntries(const void *payload, size_t payload_bytes,
                                   uint32_t expected_entry_count,
                                   std::vector<OmpFileIOBatchPlanEntry> &entries) {
  const uint64_t expected_bytes = batchPlanPayloadBytes(expected_entry_count);
  if (expected_bytes != static_cast<uint64_t>(payload_bytes))
    return false;
  entries.resize(expected_entry_count);
  if (expected_entry_count == 0)
    return true;
  if (!payload)
    return false;
  std::memcpy(entries.data(), payload, payload_bytes);
  return true;
}

enum OmpFileReadContextFlags : uint32_t {
  OMPFILE_READ_CTX_HAS_PATH_KEY = 1u << 0,
  OMPFILE_READ_CTX_HAS_PLAN = 1u << 1,
};

enum OmpFileWriteContextFlags : uint32_t {
  OMPFILE_WRITE_CTX_HAS_PATH_KEY = 1u << 0,
};

struct OmpFileReadRequestContext {
  uint64_t RequestId = 0;
  int32_t FileHandle = -1;
  int32_t ClientRank = -1;
  int64_t Offset = 0;
  uint64_t Size = 0;
  uint64_t PathKey = 0;
  uint32_t ContextFlags = 0;
  uint32_t Reserved = 0;
  OmpFileIOHint Hint{};
  OmpFileIOPlan Plan{};
};

struct OmpFileWriteRequestContext {
  uint64_t RequestId = 0;
  int32_t FileHandle = -1;
  int32_t ClientRank = -1;
  int64_t Offset = 0;
  uint64_t Size = 0;
  uint64_t PathKey = 0;
  uint32_t ContextFlags = 0;
  uint32_t Reserved = 0;
  OmpFileIOHint Hint{};
};

static_assert(std::is_standard_layout_v<OmpFileReadRequestContext>);
static_assert(std::is_standard_layout_v<OmpFileWriteRequestContext>);

} // namespace ompfile

#endif // OMPFILE_SCHED_H
