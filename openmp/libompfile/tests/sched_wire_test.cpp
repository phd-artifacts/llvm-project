// The scheduler wire structs and the batch payload codec in ompfile_sched.h,
// and the dirty-owner batch segment of ompfile_mpp_abi.h. The sizes pin the
// ABI both sides of the shim rely on: a field added on one side changes a
// number here before it can change a lane result.

#include "ompfile_mpp_abi.h"
#include "ompfile_sched.h"
#include "ompfile_test.h"

#include <cstring>
#include <type_traits>
#include <vector>

using namespace ompfile;

static_assert(std::is_trivially_copyable_v<OmpFileIOHint>);
static_assert(std::is_trivially_copyable_v<OmpFileIORequest>);
static_assert(std::is_trivially_copyable_v<OmpFileIOPlan>);
static_assert(std::is_trivially_copyable_v<OmpFileIOBatchSegment>);
static_assert(std::is_trivially_copyable_v<OmpFileIOBatchRequest>);
static_assert(std::is_trivially_copyable_v<OmpFileIOBatchPlanEntry>);
static_assert(std::is_trivially_copyable_v<OmpFileIOBatchPlan>);
static_assert(std::is_trivially_copyable_v<OmpFileFreshnessQueryRequest>);
static_assert(std::is_trivially_copyable_v<OmpFileFreshnessQueryReply>);

OMPFILE_TEST(wire_sizes_are_the_abi) {
  OMPFILE_CHECK_EQ(sizeof(OmpFileIOHint), 40u);
  OMPFILE_CHECK_EQ(sizeof(OmpFileIORequest), 80u);
  OMPFILE_CHECK_EQ(sizeof(OmpFileIOPlan), 48u);
  OMPFILE_CHECK_EQ(sizeof(OmpFileIOBatchSegment), 72u);
  OMPFILE_CHECK_EQ(sizeof(OmpFileIOBatchRequest), 32u);
  OMPFILE_CHECK_EQ(sizeof(OmpFileIOBatchPlanEntry), 48u);
  OMPFILE_CHECK_EQ(sizeof(OmpFileIOBatchPlan), 40u);
  OMPFILE_CHECK_EQ(sizeof(OmpFileFreshnessQueryRequest), 32u);
  OMPFILE_CHECK_EQ(sizeof(OmpFileFreshnessQueryReply), 32u);
  OMPFILE_CHECK_EQ(sizeof(OmpFileDirtyOwnerPreadBatchSegment), 32u);
  OMPFILE_CHECK_EQ(OMPFILE_IO_HINT_ABI_VERSION, 1u);
  OMPFILE_CHECK_EQ(OMPFILE_SCHED_BATCH_ABI_VERSION, 1u);
  OMPFILE_CHECK_EQ(OMPFILE_FRESHNESS_QUERY_ABI_VERSION, 1u);
}

OMPFILE_TEST(hint_role_is_carried) {
  OmpFileIOHint Hint;
  OMPFILE_CHECK_EQ(Hint.Role, static_cast<uint32_t>(OMPFILE_IO_ROLE_UNKNOWN));
  OMPFILE_CHECK_EQ(OMPFILE_IO_HINT_HAS_ROLE, 1u << 3);
}

OMPFILE_TEST(batch_segments_round_trip) {
  std::vector<OmpFileIOBatchSegment> In(3);
  for (uint32_t I = 0; I < In.size(); ++I) {
    In[I].SegmentId = 100 + I;
    In[I].FileHandle = static_cast<int32_t>(I);
    In[I].Offset = static_cast<int64_t>(I) * 4096;
    In[I].Size = 4096;
    In[I].PathKey = 0xabcdef;
    In[I].TileId = I;
  }
  std::vector<uint8_t> Payload;
  OMPFILE_CHECK(encodeBatchSegments(In, Payload));
  OMPFILE_CHECK_EQ(Payload.size(), batchSegmentPayloadBytes(3));

  std::vector<OmpFileIOBatchSegment> Out;
  OMPFILE_CHECK(decodeBatchSegments(Payload.data(), Payload.size(), 3, Out));
  OMPFILE_CHECK_EQ(Out.size(), 3u);
  OMPFILE_CHECK_EQ(std::memcmp(In.data(), Out.data(), Payload.size()), 0);
}

OMPFILE_TEST(batch_segments_reject_size_mismatch_and_null_payload) {
  std::vector<OmpFileIOBatchSegment> Out;
  std::vector<uint8_t> Payload(batchSegmentPayloadBytes(2));
  OMPFILE_CHECK(!decodeBatchSegments(Payload.data(), Payload.size(), 3, Out));
  OMPFILE_CHECK(!decodeBatchSegments(Payload.data(), Payload.size() - 1, 2, Out));
  OMPFILE_CHECK(!decodeBatchSegments(nullptr, Payload.size(), 2, Out));
}

OMPFILE_TEST(batch_segments_zero_count_is_empty_and_ok) {
  std::vector<OmpFileIOBatchSegment> In;
  std::vector<uint8_t> Payload{1, 2, 3};
  OMPFILE_CHECK(encodeBatchSegments(In, Payload));
  OMPFILE_CHECK(Payload.empty());
  std::vector<OmpFileIOBatchSegment> Out(5);
  OMPFILE_CHECK(decodeBatchSegments(nullptr, 0, 0, Out));
  OMPFILE_CHECK(Out.empty());
}

OMPFILE_TEST(batch_plan_entries_round_trip) {
  std::vector<OmpFileIOBatchPlanEntry> In(2);
  In[0].SegmentId = 7;
  In[0].AggregatorRank = 1;
  In[0].PlanFlags = OMPFILE_BATCH_PLAN_BATCH_API | OMPFILE_BATCH_PLAN_REBALANCED;
  In[1].SegmentId = 8;
  In[1].Status = -1;
  In[1].Errno = 5;
  std::vector<uint8_t> Payload;
  OMPFILE_CHECK(encodeBatchPlanEntries(In, Payload));
  OMPFILE_CHECK_EQ(Payload.size(), batchPlanPayloadBytes(2));
  std::vector<OmpFileIOBatchPlanEntry> Out;
  OMPFILE_CHECK(decodeBatchPlanEntries(Payload.data(), Payload.size(), 2, Out));
  OMPFILE_CHECK_EQ(Out.size(), 2u);
  OMPFILE_CHECK_EQ(Out[0].PlanFlags, In[0].PlanFlags);
  OMPFILE_CHECK_EQ(Out[1].Errno, 5);
  OMPFILE_CHECK(!decodeBatchPlanEntries(Payload.data(), Payload.size(), 1, Out));
}
