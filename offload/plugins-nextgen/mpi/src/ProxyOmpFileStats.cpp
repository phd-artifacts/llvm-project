//===-- ProxyOmpFileStats.cpp - Proxy OMPFile stats lines -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The single home of the proxy stats emitters the lane parsers key on:
// proxy_event_stats (from ProxyEventStats.h), OMPFile stats and OMPFile
// writeback stats. Format strings and argument order are part of that
// contract.
//
//===----------------------------------------------------------------------===//

#include "ProxyOmpFile.h"

#include "ProxyEventStats.h"

void ProxyOmpFile::reportProxyEventStats(const char *Scope) {
  if (!proxyEventStatsEnabled())
    return;
  ProxyEventStatsTy &S = proxyEventStats();
  for (size_t I = 0; I < ProxyEventStatsTy::kTypes; ++I) {
    const uint64_t N = S.Count[I].load(std::memory_order_relaxed);
    if (N == 0)
      continue;
    const uint64_t Q = S.QueueUs[I].load(std::memory_order_relaxed);
    const uint64_t V = S.ServiceUs[I].load(std::memory_order_relaxed);
    REPORT("MPIProxyDevice --> proxy_event_stats rank=%d scope=%s type=%s "
           "events=%llu queue_us=%llu service_us=%llu max_total_us=%llu "
           "resumes=%llu avg_queue_us=%.1f avg_service_us=%.1f\n",
           EventSystem.LocalRank, Scope,
           EventTypeToString(static_cast<EventTypeTy>(I)).c_str(),
           static_cast<unsigned long long>(N),
           static_cast<unsigned long long>(Q),
           static_cast<unsigned long long>(V),
           static_cast<unsigned long long>(
               S.MaxTotalUs[I].load(std::memory_order_relaxed)),
           static_cast<unsigned long long>(
               S.Resumes[I].load(std::memory_order_relaxed)),
           static_cast<double>(Q) / static_cast<double>(N),
           static_cast<double>(V) / static_cast<double>(N));
  }
}

void ProxyOmpFile::reportOmpFileStats(const char *Scope) {
  reportProxyEventStats(Scope);
  if (!OmpFileOptStats)
    return;

  uint64_t OpenReq = OmpFileStatsOpenRequests.load(std::memory_order_relaxed);
  uint64_t OpenSys = OmpFileStatsOpenSyscalls.load(std::memory_order_relaxed);
  uint64_t OpenHits =
      OmpFileStatsOpenCacheHits.load(std::memory_order_relaxed);
  uint64_t OpenEioRetries =
      OmpFileStatsOpenEioRetries.load(std::memory_order_relaxed);
  uint64_t OpenEioFailures =
      OmpFileStatsOpenEioFailures.load(std::memory_order_relaxed);
  uint64_t CloseReq =
      OmpFileStatsCloseRequests.load(std::memory_order_relaxed);
  uint64_t CloseSys =
      OmpFileStatsCloseSyscalls.load(std::memory_order_relaxed);
  uint64_t CloseDeferred =
      OmpFileStatsCloseDeferred.load(std::memory_order_relaxed);
  uint64_t PwriteAsyncEvents =
      OmpFileStatsPwriteAsyncEvents.load(std::memory_order_relaxed);
  uint64_t PwriteAsyncFragments =
      OmpFileStatsPwriteAsyncFragments.load(std::memory_order_relaxed);
  uint64_t PwriteBlockingFallbacks =
      OmpFileStatsPwriteBlockingFallbacks.load(std::memory_order_relaxed);
  uint64_t PwritePayloadBytes =
      OmpFileStatsPwritePayloadBytes.load(std::memory_order_relaxed);
  uint64_t PwriteFailures =
      OmpFileStatsPwriteFailures.load(std::memory_order_relaxed);
  uint64_t StagedReadHits =
      OmpFileStatsStagedReadHits.load(std::memory_order_relaxed);
  uint64_t StagedReadBytes =
      OmpFileStatsStagedReadBytes.load(std::memory_order_relaxed);
  uint64_t StageBypassReads =
      OmpFileStatsStageBypassReads.load(std::memory_order_relaxed);
  uint64_t StageBypassBytes =
      OmpFileStatsStageBypassBytes.load(std::memory_order_relaxed);
  uint64_t StageLookupHits =
      OmpFileStatsStageLookupHits.load(std::memory_order_relaxed);
  uint64_t StageLookupMisses =
      OmpFileStatsStageLookupMisses.load(std::memory_order_relaxed);
  uint64_t StagePopulateCount =
      OmpFileStatsStagePopulateCount.load(std::memory_order_relaxed);
  uint64_t StagePopulateFailures =
      OmpFileStatsStagePopulateFailures.load(std::memory_order_relaxed);
  uint64_t StagePopulateBytes =
      OmpFileStatsStagePopulateBytes.load(std::memory_order_relaxed);
  uint64_t StagePopulateUs =
      OmpFileStatsStagePopulateUs.load(std::memory_order_relaxed);
  uint64_t StageCopyUs =
      OmpFileStatsStageCopyUs.load(std::memory_order_relaxed);
  uint64_t StageFsyncUs =
      OmpFileStatsStageFsyncUs.load(std::memory_order_relaxed);
  uint64_t StageReopenUs =
      OmpFileStatsStageReopenUs.load(std::memory_order_relaxed);
  uint64_t StageLockWaitUs =
      OmpFileStatsStageLockWaitUs.load(std::memory_order_relaxed);
  uint64_t StageLockHoldUs =
      OmpFileStatsStageLockHoldUs.load(std::memory_order_relaxed);
  uint64_t SourcePreadBytes =
      OmpFileStatsSourcePreadBytes.load(std::memory_order_relaxed);
  uint64_t SourcePreadUs =
      OmpFileStatsSourcePreadUs.load(std::memory_order_relaxed);
  uint64_t StagedPreadUs =
      OmpFileStatsStagedPreadUs.load(std::memory_order_relaxed);
  uint64_t SourcePwriteBytes =
      OmpFileStatsSourcePwriteBytes.load(std::memory_order_relaxed);
  uint64_t SourcePwriteOps =
      OmpFileStatsSourcePwriteOps.load(std::memory_order_relaxed);
  uint64_t SourcePwriteUs =
      OmpFileStatsSourcePwriteUs.load(std::memory_order_relaxed);
  uint64_t SourceFsyncOps =
      OmpFileStatsSourceFsyncOps.load(std::memory_order_relaxed);
  uint64_t SourceFsyncUs =
      OmpFileStatsSourceFsyncUs.load(std::memory_order_relaxed);
  uint64_t StagingInvalidations =
      OmpFileStatsStagingInvalidations.load(std::memory_order_relaxed);
  uint64_t StagingRangeInvalidations =
      OmpFileStatsStagingRangeInvalidations.load(std::memory_order_relaxed);
  uint64_t StagingFullInvalidations =
      OmpFileStatsStagingFullInvalidations.load(std::memory_order_relaxed);
  uint64_t StagingInvalidatedBytes =
      OmpFileStatsStagingInvalidatedBytes.load(std::memory_order_relaxed);
  uint64_t StagingWriteBypass =
      OmpFileStatsStagingWriteBypassCount.load(std::memory_order_relaxed);
  uint64_t StagedWriteUpdates =
      OmpFileStatsStagedWriteUpdates.load(std::memory_order_relaxed);
  uint64_t StagedWriteBytes =
      OmpFileStatsStagedWriteBytes.load(std::memory_order_relaxed);
  uint64_t StageWriteFailures =
      OmpFileStatsStageWriteFailures.load(std::memory_order_relaxed);
  uint64_t StageWriteUs =
      OmpFileStatsStageWriteUs.load(std::memory_order_relaxed);
  uint64_t StageWriteFsyncOps =
      OmpFileStatsStageWriteFsyncOps.load(std::memory_order_relaxed);
  uint64_t StageWriteFsyncUs =
      OmpFileStatsStageWriteFsyncUs.load(std::memory_order_relaxed);
  uint64_t StageDirtyFlushUs =
      OmpFileStatsStageDirtyFlushUs.load(std::memory_order_relaxed);
  uint64_t StageDirtyFlushReadUs =
      OmpFileStatsStageDirtyFlushReadUs.load(std::memory_order_relaxed);
  uint64_t StageWritebackCaptures =
      OmpFileStatsStageWritebackCaptures.load(std::memory_order_relaxed);
  uint64_t StageWritebackCaptureBytes =
      OmpFileStatsStageWritebackCaptureBytes.load(std::memory_order_relaxed);
  uint64_t StageDirtyBytes =
      OmpFileStatsStageDirtyBytes.load(std::memory_order_relaxed);
  uint64_t StageDirtyFlushes =
      OmpFileStatsStageDirtyFlushes.load(std::memory_order_relaxed);
  uint64_t StageDirtyFlushBytes =
      OmpFileStatsStageDirtyFlushBytes.load(std::memory_order_relaxed);
  uint64_t StageDirtyFlushFailures =
      OmpFileStatsStageDirtyFlushFailures.load(std::memory_order_relaxed);
  uint64_t StageDirtyFlushCoalesced =
      OmpFileStatsStageDirtyFlushCoalesced.load(std::memory_order_relaxed);
  uint64_t StagingEvictions =
      OmpFileStatsStagingEvictions.load(std::memory_order_relaxed);
  uint64_t CoherentReadRefreshes =
      OmpFileStatsCoherentReadRefreshes.load(std::memory_order_relaxed);
  uint64_t CoherentReadRefreshFailures =
      OmpFileStatsCoherentReadRefreshFailures.load(std::memory_order_relaxed);
  uint64_t CoherentReadRefreshUs =
      OmpFileStatsCoherentReadRefreshUs.load(std::memory_order_relaxed);
  uint64_t StageFreshnessGuardBypasses =
      OmpFileStatsStageFreshnessGuardBypasses.load(std::memory_order_relaxed);
  uint64_t DirtyOwnerForwardReads =
      OmpFileStatsDirtyOwnerForwardReads.load(std::memory_order_relaxed);
  uint64_t DirtyOwnerForwardBytes =
      OmpFileStatsDirtyOwnerForwardBytes.load(std::memory_order_relaxed);
  uint64_t DirtyOwnerForwardFailures =
      OmpFileStatsDirtyOwnerForwardFailures.load(std::memory_order_relaxed);
  uint64_t LocalDisjointWrites =
      OmpFileStatsLocalDisjointWrites.load(std::memory_order_relaxed);
  uint64_t LocalDisjointBytes =
      OmpFileStatsLocalDisjointBytes.load(std::memory_order_relaxed);
  uint64_t LocalDisjointFallbacks =
      OmpFileStatsLocalDisjointFallbacks.load(std::memory_order_relaxed);
  uint64_t LocalDisjointReads =
      OmpFileStatsLocalDisjointReads.load(std::memory_order_relaxed);
  uint64_t LocalDisjointReadBytes =
      OmpFileStatsLocalDisjointReadBytes.load(std::memory_order_relaxed);
  uint64_t LocalDisjointReadFallbacks =
      OmpFileStatsLocalDisjointReadFallbacks.load(std::memory_order_relaxed);
  uint64_t LocalDisjointCombinedWrites =
      OmpFileStatsLocalDisjointCombinedWrites.load(std::memory_order_relaxed);
  uint64_t LocalDisjointCombineFlushes =
      OmpFileStatsLocalDisjointCombineFlushes.load(std::memory_order_relaxed);
  uint64_t LocalDisjointReadaheadHits =
      OmpFileStatsLocalDisjointReadaheadHits.load(std::memory_order_relaxed);
  uint64_t LocalDisjointReadaheadFills =
      OmpFileStatsLocalDisjointReadaheadFills.load(std::memory_order_relaxed);
  uint64_t LocalDisjointStripedWrites =
      OmpFileStatsLocalDisjointStripedWrites.load(std::memory_order_relaxed);
  uint64_t LocalDisjointStripes =
      OmpFileStatsLocalDisjointStripes.load(std::memory_order_relaxed);
  uint64_t LocalDisjointStripedReads =
      OmpFileStatsLocalDisjointStripedReads.load(std::memory_order_relaxed);
  uint64_t LocalDisjointReadStripes =
      OmpFileStatsLocalDisjointReadStripes.load(std::memory_order_relaxed);
  size_t CacheEntries = 0;

  {
    const std::lock_guard<std::mutex> Lock(OmpFileOpenCacheMutex);
    CacheEntries = OmpFileOpenCacheByKey.size();
  }

  fprintf(stderr,
          "MPIProxyDevice --> OMPFile stats [%s] rank=%d stats_schema=2 "
          "open_req=%llu open_sys=%llu open_hits=%llu "
          "open_eio_retries=%llu open_eio_failures=%llu "
          "close_req=%llu close_sys=%llu close_deferred=%llu "
          "cache_entries=%zu pwrite_async_events=%llu "
          "pwrite_async_fragments=%llu pwrite_blocking_fallbacks=%llu "
          "pwrite_payload_bytes=%llu pwrite_failures=%llu "
          "topology_file_set=%d topology_loaded=%d topology_entries=%llu "
          "stage_mode=%s stage_root_policy=%s stage_root=%s "
          "stage_class=%s shared_storage_path=%s shared_storage_class=%s "
          "storage_environment=%s stage_sync_policy=%s "
          "stage_populate_mode=%s stage_window_bytes=%llu "
          "stage_min_free_bytes=%llu "
          "staged_read_hits=%llu staged_read_bytes=%llu "
          "staging_bypass_reads=%llu staging_bypass_bytes=%llu "
          "staged_write_updates=%llu staged_write_bytes=%llu "
          "stage_writeback_captures=%llu stage_writeback_capture_bytes=%llu "
          "stage_dirty_bytes=%llu stage_dirty_flushes=%llu "
          "stage_dirty_flush_bytes=%llu stage_dirty_flush_failures=%llu "
          "stage_dirty_flush_coalesced=%llu "
          "staging_write_bypass_count=%llu stage_write_failures=%llu "
          "stage_write_us_total=%llu stage_write_fsync_ops=%llu "
          "stage_write_fsync_us_total=%llu "
          "stage_dirty_flush_us_total=%llu "
          "stage_dirty_flush_read_us_total=%llu "
          "stage_lookup_hits=%llu stage_lookup_misses=%llu "
          "stage_populate_count=%llu stage_populate_failures=%llu "
          "stage_populate_bytes=%llu stage_populate_us_total=%llu "
          "stage_copy_us_total=%llu stage_fsync_us_total=%llu "
          "stage_reopen_us_total=%llu stage_lock_wait_us_total=%llu "
          "stage_lock_hold_us_total=%llu "
          "source_pread_bytes=%llu "
          "source_pread_us_total=%llu staged_pread_us_total=%llu "
          "source_pwrite_bytes=%llu source_pwrite_ops=%llu "
          "source_pwrite_us_total=%llu "
          "source_fsync_ops=%llu source_fsync_us_total=%llu "
          "staging_invalidations=%llu staging_range_invalidations=%llu "
          "staging_full_invalidations=%llu "
          "staging_invalidated_bytes=%llu "
          "coherent_read_refreshes=%llu "
          "coherent_read_refresh_failures=%llu "
          "coherent_read_refresh_us_total=%llu "
          "staging_evictions=%llu "
          "stage_freshness_guard_bypasses=%llu dirty_owner_forward_reads=%llu dirty_owner_forward_bytes=%llu dirty_owner_forward_failures=%llu "
          "local_disjoint_writes=%llu local_disjoint_bytes=%llu local_disjoint_fallbacks=%llu "
          "local_disjoint_reads=%llu local_disjoint_read_bytes=%llu local_disjoint_read_fallbacks=%llu "
          "local_disjoint_combined_writes=%llu local_disjoint_combine_flushes=%llu "
          "local_disjoint_readahead_hits=%llu local_disjoint_readahead_fills=%llu "
          "local_disjoint_write_streams=%llu local_disjoint_striped_writes=%llu "
          "local_disjoint_stripes=%llu local_disjoint_read_streams=%llu "
          "local_disjoint_striped_reads=%llu local_disjoint_read_stripes=%llu\n",
          Scope ? Scope : "unknown", EventSystem.LocalRank,
          static_cast<unsigned long long>(OpenReq),
          static_cast<unsigned long long>(OpenSys),
          static_cast<unsigned long long>(OpenHits),
          static_cast<unsigned long long>(OpenEioRetries),
          static_cast<unsigned long long>(OpenEioFailures),
          static_cast<unsigned long long>(CloseReq),
          static_cast<unsigned long long>(CloseSys),
          static_cast<unsigned long long>(CloseDeferred), CacheEntries,
          static_cast<unsigned long long>(PwriteAsyncEvents),
          static_cast<unsigned long long>(PwriteAsyncFragments),
          static_cast<unsigned long long>(PwriteBlockingFallbacks),
          static_cast<unsigned long long>(PwritePayloadBytes),
          static_cast<unsigned long long>(PwriteFailures),
          static_cast<int>(!OmpFileTopologyFile.empty()),
          static_cast<int>(OmpFileTopologyLoaded),
          static_cast<unsigned long long>(OmpFileTopologyEntries),
          OmpFileStageMode.c_str(), OmpFileStageRootPolicy.c_str(),
          OmpFileStageRoot.empty() ? "(unset)" : OmpFileStageRoot.c_str(),
          OmpFileSelectedStageClass.empty()
              ? "(unset)"
              : OmpFileSelectedStageClass.c_str(),
          OmpFileSharedStoragePath.empty()
              ? "(unset)"
              : OmpFileSharedStoragePath.c_str(),
          OmpFileSharedStorageClass.empty()
              ? "(unset)"
              : OmpFileSharedStorageClass.c_str(),
          OmpFileStorageEnvironment.c_str(), OmpFileStageSyncPolicy.c_str(),
          OmpFileStagePopulateMode.c_str(),
          static_cast<unsigned long long>(OmpFileStageWindowBytes),
          static_cast<unsigned long long>(OmpFileStageMinFreeBytes),
          static_cast<unsigned long long>(StagedReadHits),
          static_cast<unsigned long long>(StagedReadBytes),
          static_cast<unsigned long long>(StageBypassReads),
          static_cast<unsigned long long>(StageBypassBytes),
          static_cast<unsigned long long>(StagedWriteUpdates),
          static_cast<unsigned long long>(StagedWriteBytes),
          static_cast<unsigned long long>(StageWritebackCaptures),
          static_cast<unsigned long long>(StageWritebackCaptureBytes),
          static_cast<unsigned long long>(StageDirtyBytes),
          static_cast<unsigned long long>(StageDirtyFlushes),
          static_cast<unsigned long long>(StageDirtyFlushBytes),
          static_cast<unsigned long long>(StageDirtyFlushFailures),
          static_cast<unsigned long long>(StageDirtyFlushCoalesced),
          static_cast<unsigned long long>(StagingWriteBypass),
          static_cast<unsigned long long>(StageWriteFailures),
          static_cast<unsigned long long>(StageWriteUs),
          static_cast<unsigned long long>(StageWriteFsyncOps),
          static_cast<unsigned long long>(StageWriteFsyncUs),
          static_cast<unsigned long long>(StageDirtyFlushUs),
          static_cast<unsigned long long>(StageDirtyFlushReadUs),
          static_cast<unsigned long long>(StageLookupHits),
          static_cast<unsigned long long>(StageLookupMisses),
          static_cast<unsigned long long>(StagePopulateCount),
          static_cast<unsigned long long>(StagePopulateFailures),
          static_cast<unsigned long long>(StagePopulateBytes),
          static_cast<unsigned long long>(StagePopulateUs),
          static_cast<unsigned long long>(StageCopyUs),
          static_cast<unsigned long long>(StageFsyncUs),
          static_cast<unsigned long long>(StageReopenUs),
          static_cast<unsigned long long>(StageLockWaitUs),
          static_cast<unsigned long long>(StageLockHoldUs),
          static_cast<unsigned long long>(SourcePreadBytes),
          static_cast<unsigned long long>(SourcePreadUs),
          static_cast<unsigned long long>(StagedPreadUs),
          static_cast<unsigned long long>(SourcePwriteBytes),
          static_cast<unsigned long long>(SourcePwriteOps),
          static_cast<unsigned long long>(SourcePwriteUs),
          static_cast<unsigned long long>(SourceFsyncOps),
          static_cast<unsigned long long>(SourceFsyncUs),
          static_cast<unsigned long long>(StagingInvalidations),
          static_cast<unsigned long long>(StagingRangeInvalidations),
          static_cast<unsigned long long>(StagingFullInvalidations),
          static_cast<unsigned long long>(StagingInvalidatedBytes),
          static_cast<unsigned long long>(CoherentReadRefreshes),
          static_cast<unsigned long long>(CoherentReadRefreshFailures),
          static_cast<unsigned long long>(CoherentReadRefreshUs),
          static_cast<unsigned long long>(StagingEvictions),
          static_cast<unsigned long long>(StageFreshnessGuardBypasses),
          static_cast<unsigned long long>(DirtyOwnerForwardReads),
          static_cast<unsigned long long>(DirtyOwnerForwardBytes),
          static_cast<unsigned long long>(DirtyOwnerForwardFailures),
          static_cast<unsigned long long>(LocalDisjointWrites),
          static_cast<unsigned long long>(LocalDisjointBytes),
          static_cast<unsigned long long>(LocalDisjointFallbacks),
          static_cast<unsigned long long>(LocalDisjointReads),
          static_cast<unsigned long long>(LocalDisjointReadBytes),
          static_cast<unsigned long long>(LocalDisjointReadFallbacks),
          static_cast<unsigned long long>(LocalDisjointCombinedWrites),
          static_cast<unsigned long long>(LocalDisjointCombineFlushes),
          static_cast<unsigned long long>(LocalDisjointReadaheadHits),
          static_cast<unsigned long long>(LocalDisjointReadaheadFills),
          static_cast<unsigned long long>(OmpFileLocalDisjointWriteStreams),
          static_cast<unsigned long long>(LocalDisjointStripedWrites),
          static_cast<unsigned long long>(LocalDisjointStripes),
          static_cast<unsigned long long>(OmpFileLocalDisjointReadStreams),
          static_cast<unsigned long long>(LocalDisjointStripedReads),
          static_cast<unsigned long long>(LocalDisjointReadStripes));

  // stats_schema=2 marks the corrected field alignment: the format now
  // carries dirty_flush_coalesced, so every counter after it lines up with
  // its own argument. A writeback line with no stats_schema token predates
  // that fix; in those bundles staged_write_updates, staged_write_bytes and
  // write_bypass_count are shifted by one and write_failures never printed
  // StageWriteFailures. Do not compare the two layouts.
  fprintf(stderr,
          "MPIProxyDevice --> OMPFile writeback stats [%s] rank=%d "
          "stats_schema=2 "
          "stage_write_mode=%s captures=%llu capture_bytes=%llu "
          "dirty_bytes=%llu dirty_flushes=%llu dirty_flush_bytes=%llu "
          "dirty_flush_failures=%llu dirty_flush_coalesced=%llu "
          "staged_write_updates=%llu "
          "staged_write_bytes=%llu write_bypass_count=%llu "
          "write_failures=%llu\n",
          Scope ? Scope : "unknown", EventSystem.LocalRank,
          OmpFileStageWriteMode.c_str(),
          static_cast<unsigned long long>(StageWritebackCaptures),
          static_cast<unsigned long long>(StageWritebackCaptureBytes),
          static_cast<unsigned long long>(StageDirtyBytes),
          static_cast<unsigned long long>(StageDirtyFlushes),
          static_cast<unsigned long long>(StageDirtyFlushBytes),
          static_cast<unsigned long long>(StageDirtyFlushFailures),
          static_cast<unsigned long long>(StageDirtyFlushCoalesced),
          static_cast<unsigned long long>(StagedWriteUpdates),
          static_cast<unsigned long long>(StagedWriteBytes),
          static_cast<unsigned long long>(StagingWriteBypass),
          static_cast<unsigned long long>(StageWriteFailures));
}
