#ifndef OMPFILE_MPP_SHIM_H
#define OMPFILE_MPP_SHIM_H

#include <cstdint>
#include <cstddef>

namespace ompfile {
namespace mpp {

bool init();
bool submit(uint64_t token);
bool poll(uint64_t token, bool &done);
void finalize();
bool ping();

bool open(const char *path, int flags, int mode, int &handle);
bool close(int handle);
bool pread(int handle, int64_t offset, void *buffer, size_t size);
bool pwrite(int handle, int64_t offset, const void *buffer, size_t size);

} // namespace mpp
} // namespace ompfile

#endif // OMPFILE_MPP_SHIM_H
