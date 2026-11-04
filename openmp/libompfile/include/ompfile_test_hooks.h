#ifndef OMPFILE_TEST_HOOKS_H
#define OMPFILE_TEST_HOOKS_H

// Test-only entry points of libompfile. NOT part of the public API: this
// header is not installed (only file_interface.h is), file_interface.h does
// not declare these, and application/tests/test-mpp-rebalance-fallback-
// regression/test_runtime_test_access_hidden.cpp fails to compile if it ever
// does. The symbols are exported from libompfile.so only so that a regression
// under application/tests/ can reach them from a target region; an
// application must never call them.
//
// Why they exist: the knobs that used to put dirty data on more than one
// proxy (LIBOMPFILE_OPT_WRITEBACK_DISTRIBUTE_WRITES / _RANKS, removed
// Oct 2026) were the only production way to exercise the multi-owner
// coherence machinery (freshness COPY_FROM_RANK answers, dirty-owner
// forwarding, the per-(file, rank) remote handle cache) that Phase E keeps.
// A test now places the writes itself.

#include "file_interface.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Feature macro: a runtime whose libompfile exports the hooks below.
#define OMPFILE_HAVE_TEST_PWRITE_ON_RANK 1
#define OMPFILE_HAVE_TEST_OPEN_RELATIVE 1

// Test-only controlled OPEN. Called on a proxy (inside a target region),
// peer=0 places the file on that proxy; peer=1 on the next proxy cyclically.
// The runtime resolves ranks, obtains a counted headnode OPEN plan, and checks
// the actual opened owner. Subsequent ordinary I/O is unchanged. Reports the
// actual caller/owner; fails closed on invalid input or conflicting placement.
int ompfile_test_open_relative(const char *path, int peer, int *writer_rank,
                               int *owner_rank);

// Writes [offset, offset + size) of the open handle `file_handle` on proxy
// rank `rank` instead of on the handle's owner: drains the async queue,
// opens (once, then cached per (file, rank) until close) a handle for the
// file on that rank, writes there, records the write epoch and, when `hint`
// carries a tile, commits its freshness version with `rank` as the writer,
// exactly as a write that rank owned would. Synchronous; fails closed: 0 on
// success, -1 with errno set otherwise (EBADF for a handle that is not open,
// EINVAL for a negative rank or offset, the transport's errno when the
// placed write fails; there is no fallback to the owner). `hint` may be NULL.
int ompfile_test_pwrite_on_rank(int file_handle, long offset, const void *data,
                                size_t size, int rank,
                                const omp_file_io_hint_v1 *hint);

#ifdef __cplusplus
}
#endif

#endif // OMPFILE_TEST_HOOKS_H
