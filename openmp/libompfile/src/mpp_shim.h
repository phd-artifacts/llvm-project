#ifndef OMPFILE_MPP_SHIM_H
#define OMPFILE_MPP_SHIM_H

#include <cstdint>

namespace ompfile {
namespace mpp {

bool init();
bool submit(uint64_t token);
bool poll(uint64_t token, bool &done);
void finalize();
bool ping();

} // namespace mpp
} // namespace ompfile

#endif // OMPFILE_MPP_SHIM_H
