#include "client_context.h"
#include "debug_log.h"
#include "file_interface.h"
#include "mpp_shim.h"
#include "ompfile_keys.h"
#include "ompfile_config.h"
#include "ompfile_graph_plan.h"
#include "ompfile_trace.h"
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <vector>

// One range per public call, on the calling thread (thread-local, so
// push/pop). Payload = the request size where there is one.
#define OMPFILE_API_TRACE(Name, Size)                                          \
  ompfile::trace::Scope OmpFileApiTrace(ompfile::trace::Domain::LibOmpFile,   \
                                        Name, ompfile::trace::Color::Io,       \
                                        static_cast<uint64_t>(Size))

// Runs one public call and, on a proxy, notes it for the footprint of the task
// whose target region made it. The time is what the region spent blocked here,
// so an async write counts its enqueue, not the write behind it. A failed call
// keeps its time but touches no range.
template <typename OpFn>
static int with_task_io(ompfile::mpp::TaskIoKind kind, int file_handle,
                        long offset, size_t size, OpFn &&op) {
  if (!ompfile::mpp::taskIoNoteAvailable())
    return op();
  const auto start = std::chrono::steady_clock::now();
  const int rc = op();
  const double seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
          .count();
  if (kind == ompfile::mpp::TaskIoMeta || rc < 0) {
    ompfile::mpp::taskIoNote(0, -1, ompfile::mpp::TaskIoMeta, 0, 0, seconds);
    return rc;
  }
  uint64_t path_key = 0;
  int owner_rank = -1;
  OmpFileClientContext::getInstance().taskIoTarget(file_handle, path_key,
                                                   owner_rank);
  ompfile::mpp::taskIoNote(path_key, owner_rank, kind, offset, size, seconds);
  return rc;
}

extern "C" {

// Reads cannot be fire-and-forget through the fixed C ABI (there is no
// completion handle to return the buffer through later), so an async-requested
// read is served synchronously. The read paths already drain any in-flight
// async writes first, so the data is coherent. Log the downgrade once.
static void note_async_read_synchronous(int async) {
  if (!async)
    return;
  static std::once_flag once;
  std::call_once(once, [] {
    io_log("Async read requested; served synchronously (no async-read "
           "completion handle in the ABI).\n");
  });
}

int omp_file_declare_next_task_io(const char *path, long offset, size_t size,
                                  int is_write) {
  if (path == nullptr || path[0] == '\0' || offset < -1 ||
      (offset >= 0 && size == 0)) {
    errno = EINVAL;
    return -1;
  }
  return ompfile::mpp::declareNextTaskIo(ompfile::keys::pathKey(path), offset,
                                         offset < 0 ? 0 : size, is_write != 0);
}

int ompfile_graph_plan_writes(const ompfile_graph_write *writes, int count) {
  if (count < 0 || (count > 0 && writes == nullptr)) {
    errno = EINVAL;
    return -1;
  }
  static const ompfile::config::ClientConfig config =
      ompfile::config::ClientConfig::fromEnvironment();
  if (!config.GraphPlan.Value || count == 0)
    return 0;
  if (!config.graphPlanCoherent()) {
    static std::once_flag once;
    std::call_once(once, [] {
      io_report("graph plan ignored: LIBOMPFILE_GRAPH_PLAN=1 needs "
                "LIBOMPFILE_STAGE_MODE=readthrough, write-back stage and tile "
                "freshness, LIBOMPFILE_STAGE_FRESHNESS_GUARD=1, "
                "LIBOMPFILE_OPT_DIRTY_OWNER_FORWARDING=1 and "
                "LIBOMPFILE_OPT_TWO_PHASE=0\n");
    });
    return 0;
  }

  // One range per run of touching writes to a file.
  std::vector<ompfile_graph_write> sorted(writes, writes + count);
  std::sort(sorted.begin(), sorted.end(),
            [](const ompfile_graph_write &a, const ompfile_graph_write &b) {
              return a.path_key != b.path_key ? a.path_key < b.path_key
                                              : a.offset < b.offset;
            });
  std::vector<ompfile::OmpFileIOBatchSegment> segments;
  for (const ompfile_graph_write &w : sorted) {
    if (w.path_key == 0 || w.offset < 0 || w.extent == 0)
      continue;
    const int64_t end = w.offset + static_cast<int64_t>(w.extent);
    if (!segments.empty() && segments.back().PathKey == w.path_key &&
        segments.back().Offset + static_cast<int64_t>(segments.back().Size) >=
            w.offset) {
      ompfile::OmpFileIOBatchSegment &last = segments.back();
      last.Size = static_cast<uint64_t>(
          std::max(end, last.Offset + static_cast<int64_t>(last.Size)) -
          last.Offset);
      continue;
    }
    ompfile::OmpFileIOBatchSegment segment{};
    segment.SegmentId = segments.size();
    segment.PathKey = w.path_key;
    segment.Offset = w.offset;
    segment.Size = w.extent;
    segments.push_back(segment);
  }
  if (segments.empty())
    return 0;
  if (!ompfile::mpp::init()) {
    errno = EHOSTDOWN;
    return -1;
  }

  static std::atomic<uint64_t> batch_id{1};
  ompfile::OmpFileIOBatchRequest request{};
  request.AbiVersion = ompfile::OMPFILE_SCHED_BATCH_ABI_VERSION;
  request.SegmentCount = static_cast<uint32_t>(segments.size());
  request.RequestFlags = ompfile::OMPFILE_BATCH_REQ_PLACE_WRITES;
  request.BatchId = batch_id.fetch_add(1, std::memory_order_relaxed);
  ompfile::OmpFileIOBatchPlan plan{};
  std::vector<ompfile::OmpFileIOBatchPlanEntry> entries;
  if (!ompfile::mpp::schedBatchRequest(request, segments, plan, entries))
    return -1;
  if (plan.Status != 0) {
    errno = plan.Errno != 0 ? plan.Errno : EIO;
    return -1;
  }
  return static_cast<int>(segments.size());
}

int omp_file_open(const char *filename) {
  OMPFILE_API_TRACE("omp_file_open", 0);
  io_trace("omp_file_open api enter filename=%s\n",
           filename ? filename : "(null)");
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = with_task_io(ompfile::mpp::TaskIoMeta, -1, 0, 0, [&] {
    return ctx.openFile(filename);
  });
  io_trace("omp_file_open api exit rc=%d\n", rc);
  return rc;
}

int omp_file_write(int file_handle, const void *data, size_t size, int async) {
  auto &ctx = OmpFileClientContext::getInstance();
  return with_task_io(ompfile::mpp::TaskIoWrite, file_handle, -1, size, [&] {
    return ctx.submitWrite(file_handle, /*offset=*/0, /*has_offset=*/false,
                           data, size, /*hint=*/nullptr, async != 0);
  });
}

int omp_file_pwrite(int file_handle, long offset, const void *data, size_t size,
                    int async) {
  OMPFILE_API_TRACE("omp_file_pwrite", size);
  auto &ctx = OmpFileClientContext::getInstance();
  return with_task_io(ompfile::mpp::TaskIoWrite, file_handle, offset, size, [&] {
    return ctx.submitWrite(file_handle, offset, /*has_offset=*/true, data, size,
                           /*hint=*/nullptr, async != 0);
  });
}

int omp_file_pwrite_hint(int file_handle, long offset, const void *data,
                         size_t size, int async,
                         const omp_file_io_hint_v1 *hint) {
  OMPFILE_API_TRACE("omp_file_pwrite_hint", size);
  auto &ctx = OmpFileClientContext::getInstance();
  return with_task_io(ompfile::mpp::TaskIoWrite, file_handle, offset, size, [&] {
    return ctx.submitWrite(file_handle, offset, /*has_offset=*/true, data, size,
                           hint, async != 0);
  });
}

int omp_file_pwrite_owned(int file_handle, long offset, void *data,
                          size_t size, void (*release)(void *)) {
  OMPFILE_API_TRACE("omp_file_pwrite_owned", size);
  io_trace("omp_file_pwrite_owned api enter file_handle=%d offset=%ld "
           "size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc =
      with_task_io(ompfile::mpp::TaskIoWrite, file_handle, offset, size, [&] {
        return ctx.submitWriteOwned(file_handle, offset, data, size,
                                    release ? release : &::free,
                                    /*hint=*/nullptr);
      });
  io_trace("omp_file_pwrite_owned api exit rc=%d\n", rc);
  return rc;
}

int omp_file_pwrite_owned_hint(int file_handle, long offset, void *data,
                               size_t size, void (*release)(void *),
                               const omp_file_io_hint_v1 *hint) {
  OMPFILE_API_TRACE("omp_file_pwrite_owned_hint", size);
  io_trace("omp_file_pwrite_owned_hint api enter file_handle=%d offset=%ld "
           "size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc =
      with_task_io(ompfile::mpp::TaskIoWrite, file_handle, offset, size, [&] {
        return ctx.submitWriteOwned(file_handle, offset, data, size,
                                    release ? release : &::free, hint);
      });
  io_trace("omp_file_pwrite_owned_hint api exit rc=%d\n", rc);
  return rc;
}

int omp_file_flush(int file_handle) {
  OMPFILE_API_TRACE("omp_file_flush", 0);
  io_trace("omp_file_flush api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = with_task_io(ompfile::mpp::TaskIoMeta, file_handle, 0, 0, [&] {
    return ctx.flushFile(file_handle);
  });
  io_trace("omp_file_flush api exit rc=%d\n", rc);
  return rc;
}

int omp_file_flush_epoch(int file_handle, uint64_t epoch) {
  OMPFILE_API_TRACE("omp_file_flush_epoch", epoch);
  io_trace("omp_file_flush_epoch api enter file_handle=%d epoch=%llu\n",
           file_handle, static_cast<unsigned long long>(epoch));
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = with_task_io(ompfile::mpp::TaskIoMeta, file_handle, 0, 0, [&] {
    return ctx.flushFileEpoch(file_handle, epoch);
  });
  io_trace("omp_file_flush_epoch api exit rc=%d\n", rc);
  return rc;
}

int omp_file_commit(int file_handle) {
  OMPFILE_API_TRACE("omp_file_commit", 0);
  io_trace("omp_file_commit api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = with_task_io(ompfile::mpp::TaskIoMeta, file_handle, 0, 0, [&] {
    return ctx.commitFile(file_handle);
  });
  io_trace("omp_file_commit api exit rc=%d\n", rc);
  return rc;
}

int omp_file_pread(int file_handle, long offset, void *data, size_t size,
                   int async) {
  OMPFILE_API_TRACE("omp_file_pread", size);
  note_async_read_synchronous(async);
  io_trace("omp_file_pread api enter file_handle=%d offset=%ld size=%zu\n",
           file_handle, offset, size);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc =
      with_task_io(ompfile::mpp::TaskIoRead, file_handle, offset, size, [&] {
        return ctx.readFileAt(file_handle, data, size, offset);
      });
  io_trace("omp_file_pread api exit rc=%d\n", rc);
  return rc;
}

int omp_file_pread_hint(int file_handle, long offset, void *data, size_t size,
                        int async, const omp_file_io_hint_v1 *hint) {
  OMPFILE_API_TRACE("omp_file_pread_hint", size);
  note_async_read_synchronous(async);
  auto &ctx = OmpFileClientContext::getInstance();
  return with_task_io(ompfile::mpp::TaskIoRead, file_handle, offset, size, [&] {
    return ctx.readFileAtHint(file_handle, data, size, offset, hint);
  });
}

int omp_file_read(int file_handle, void *data, size_t size, int async) {
  note_async_read_synchronous(async);
  auto &ctx = OmpFileClientContext::getInstance();
  return with_task_io(ompfile::mpp::TaskIoRead, file_handle, -1, size, [&] {
    return ctx.readFile(file_handle, data, size);
  });
}

int omp_file_close(int file_handle) {
  OMPFILE_API_TRACE("omp_file_close", 0);
  io_trace("omp_file_close api enter file_handle=%d\n", file_handle);
  auto &ctx = OmpFileClientContext::getInstance();
  const int rc = with_task_io(ompfile::mpp::TaskIoMeta, file_handle, 0, 0, [&] {
    return ctx.closeFile(file_handle);
  });
  io_trace("omp_file_close api exit rc=%d\n", rc);
  return rc;
}

int omp_file_seek(int file_handle, long offset) {
  auto &ctx = OmpFileClientContext::getInstance();
  return ctx.seekFile(file_handle, offset);
}

} // extern "C"
