#include "mpp_shim.h"

#include "debug_log.h"

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <dlfcn.h>
#include <thread>

namespace {

using MppInitFn = int (*)();
using MppSubmitFn = int (*)(uint64_t);
using MppPollFn = int (*)(uint64_t, int *);
using MppFinalizeFn = int (*)();

struct MppApi {
  MppInitFn init = nullptr;
  MppSubmitFn submit = nullptr;
  MppPollFn poll = nullptr;
  MppFinalizeFn finalize = nullptr;
};

MppApi loadMppApi() {
  MppApi loaded;
  loaded.init = reinterpret_cast<MppInitFn>(dlsym(RTLD_DEFAULT,
                                                  "ompfile_mpp_init"));
  loaded.submit = reinterpret_cast<MppSubmitFn>(dlsym(RTLD_DEFAULT,
                                                      "ompfile_mpp_submit"));
  loaded.poll = reinterpret_cast<MppPollFn>(dlsym(RTLD_DEFAULT,
                                                  "ompfile_mpp_poll"));
  loaded.finalize = reinterpret_cast<MppFinalizeFn>(dlsym(RTLD_DEFAULT,
                                                          "ompfile_mpp_finalize"));
  return loaded;
}

MppApi &getMppApi() {
  static MppApi api = loadMppApi();
  return api;
}

} // namespace

namespace ompfile {
namespace mpp {

bool init() {
  static std::atomic<bool> ready{false};
  if (ready.load(std::memory_order_acquire))
    return true;

  auto &api = getMppApi();
  if (!api.init)
    api = loadMppApi();

  if (!api.init) {
    io_log("MPP shim not available (ompfile_mpp_init missing).\n");
    return false;
  }

  if (api.init() != 0) {
    io_log("MPP shim init failed.\n");
    return false;
  }

  ready.store(true, std::memory_order_release);
  io_log("MPP shim initialized.\n");
  return true;
}

bool submit(uint64_t token) {
  if (!init())
    return false;

  const auto &api = getMppApi();
  if (!api.submit)
    return false;
  return api.submit(token) == 0;
}

bool poll(uint64_t token, bool &done) {
  done = false;
  if (!init())
    return false;

  const auto &api = getMppApi();
  if (!api.poll)
    return false;

  int raw_done = 0;
  if (api.poll(token, &raw_done) != 0)
    return false;
  done = raw_done != 0;
  return true;
}

void finalize() {
  const auto &api = getMppApi();
  if (api.finalize)
    api.finalize();
}

bool ping() {
  static std::atomic<uint64_t> token_seed{1};
  const uint64_t token = token_seed.fetch_add(1, std::memory_order_relaxed);

  if (!submit(token))
    return false;

  for (int i = 0; i < 100; ++i) {
    bool done = false;
    if (!poll(token, done))
      return false;
    if (done)
      return true;
    std::this_thread::yield();
  }

  io_log("MPP shim ping timed out.\n");
  return false;
}

} // namespace mpp
} // namespace ompfile
