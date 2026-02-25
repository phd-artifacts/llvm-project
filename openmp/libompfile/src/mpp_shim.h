#ifndef OMPFILE_MPP_SHIM_H
#define OMPFILE_MPP_SHIM_H

#include <cstdint>
#include <cstddef>

#include "ompfile_sched.h"

namespace ompfile {
namespace mpp {

bool init();
bool submit(uint64_t token);
bool poll(uint64_t token, bool &done);
void finalize();
bool ping();

bool open(const char *path, int flags, int mode, int &handle);
bool close(int handle);
bool preadEx(int handle, int64_t offset, void *buffer, size_t size,
             size_t &bytes_read);
bool pread(int handle, int64_t offset, void *buffer, size_t size);
bool pwrite(int handle, int64_t offset, const void *buffer, size_t size);
bool schedRequest(const ompfile::OmpFileIORequest &request, const char *path,
                  ompfile::OmpFileIOPlan &plan);
bool schedBatchRequest(
    const ompfile::OmpFileIOBatchRequest &request,
    const std::vector<ompfile::OmpFileIOBatchSegment> &segments,
    ompfile::OmpFileIOBatchPlan &plan,
    std::vector<ompfile::OmpFileIOBatchPlanEntry> &entries);

} // namespace mpp
} // namespace ompfile

#endif // OMPFILE_MPP_SHIM_H
