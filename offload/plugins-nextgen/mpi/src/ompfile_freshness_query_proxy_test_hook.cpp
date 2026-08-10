#include "OmpFileHeadnodeManager.h"

#include <cerrno>

extern "C" int ompfile_mpp_init() { return 0; }

extern "C" int ompfile_mpp_handle_owner_rank(int Handle, int *RankOut) {
  if (!RankOut)
    return EINVAL;
  if (Handle < 0)
    return EBADF;
  *RankOut = 1;
  return 0;
}

extern "C" int ompfile_mpp_freshness_query_proxy_test(
    const OmpFileFreshnessQueryRequest *Request,
    OmpFileFreshnessQueryReply *Reply) {
  if (!Request || !Reply)
    return EINVAL;
  if (!OmpFileHeadnodeManager::instance().handleFreshnessQueryRequest(*Request,
                                                                      *Reply)) {
    return errno != 0 ? errno : EIO;
  }
  return 0;
}

extern "C" int ompfile_mpp_freshness_write_commit(uint64_t PathKey,
                                                    int WriterRank,
                                                    uint64_t TileId,
                                                    int WriteThroughMode,
                                                    uint64_t *CommittedVersion) {
  (void)TileId;
  if (!CommittedVersion)
    return EINVAL;
  uint64_t committed = 0;
  if (!OmpFileHeadnodeManager::instance().commitTileFreshnessWrite(
          PathKey, WriterRank, WriteThroughMode != 0, committed)) {
    return errno != 0 ? errno : EIO;
  }
  *CommittedVersion = committed;
  return 0;
}

extern "C" int ompfile_mpp_proxy_copy_tile(uint64_t PathKey, uint64_t TileId,
                                             int SourceRank, int DestRank,
                                             uint64_t Version) {
  (void)TileId;
  if (PathKey == 0 || SourceRank < 0 || DestRank < 0 || Version == 0)
    return EINVAL;
  if (!OmpFileHeadnodeManager::instance().registerTileCopy(
          PathKey, SourceRank, DestRank, Version)) {
    return errno != 0 ? errno : EIO;
  }
  return 0;
}

extern "C" int ompfile_mpp_freshness_mark_fresh(uint64_t PathKey, int Rank,
                                                  uint64_t Version) {
  if (!OmpFileHeadnodeManager::instance().markTileFreshFromCopy(PathKey, Rank,
                                                                 Version)) {
    return errno != 0 ? errno : EIO;
  }
  return 0;
}

extern "C" int ompfile_mpp_flush_dirty_tile(uint64_t PathKey, int *SourceRank,
                                             uint64_t *FlushedVersion) {
  if (!SourceRank || !FlushedVersion)
    return EINVAL;
  int Source = -1;
  uint64_t Version = 0;
  if (!OmpFileHeadnodeManager::instance().flushDirtyTile(PathKey, Source,
                                                          Version)) {
    return errno != 0 ? errno : EIO;
  }
  *SourceRank = Source;
  *FlushedVersion = Version;
  return 0;
}
