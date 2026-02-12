#ifndef OMPFILE_SCHED_H
#define OMPFILE_SCHED_H

#include <cstdint>

namespace ompfile {

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
  uint32_t Reserved = 0;
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

} // namespace ompfile

#endif // OMPFILE_SCHED_H
