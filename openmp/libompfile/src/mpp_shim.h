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
bool openOnRank(const char *path, int flags, int mode, int rank, int &handle);
bool handleOwnerRank(int handle, int &rank_out);
bool close(int handle);
bool preadEx(int handle, int64_t offset, void *buffer, size_t size,
             size_t &bytes_read);
bool preadNoStageEx(int handle, int64_t offset, void *buffer, size_t size,
                    size_t &bytes_read);
bool dirtyOwnerPreadEx(int handle, int source_rank, uint64_t expected_version,
                       int64_t offset, void *buffer, size_t size,
                       size_t &bytes_read);
bool dirtyOwnerQueryEx(int handle, int source_rank, uint64_t expected_version,
                       int64_t offset, size_t size, int &state_out);
bool pread(int handle, int64_t offset, void *buffer, size_t size);
bool pwriteEx(int handle, int64_t offset, const void *buffer, size_t size,
              size_t &bytes_written);
bool pwrite(int handle, int64_t offset, const void *buffer, size_t size);
bool stageInvalidatePathKey(uint64_t path_key, uint64_t generation,
                            const char *path);
bool freshnessQuery(uint64_t path_key, uint64_t local_version,
                    int requester_rank, uint32_t &decision_out,
                    int &source_rank_out, uint64_t &selected_version_out);
bool freshnessWriteCommit(uint64_t path_key, int writer_rank, uint64_t tile_id,
                          bool write_through_mode,
                          uint64_t &committed_version_out);
bool proxyCopyTile(uint64_t path_key, uint64_t tile_id, int source_rank,
                   int dest_rank, uint64_t version);
bool freshnessMarkFresh(uint64_t path_key, int rank, uint64_t version);
bool flushDirtyTile(uint64_t path_key, int &source_rank_out,
                    uint64_t &flushed_version_out);
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
