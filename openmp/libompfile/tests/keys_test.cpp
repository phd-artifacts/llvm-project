// The keys in ompfile_keys.h, pinned two ways: against verbatim copies of the
// bodies they replaced (Oct 2026), and against keys logged by real sorgan runs
// (the headnode's `write-version path_key=` and the proxy's `derived tile dirty
// flush complete ... key=` lines in
// artifacts/cluster/sorgan/results/job-{3168,3169,3170}/logs/slurm-driver.log).
// If a log check fails while every legacy check passes, the provenance of the
// logged value is wrong, not the code: report it, never edit a constant or a
// golden value to make it pass.

#include "ompfile_keys.h"
#include "ompfile_test.h"

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

// Verbatim copies of the replaced bodies (renamed).
uint64_t oldClientPathKey(const char *path) {
  constexpr uint64_t offset_basis = 1469598103934665603ULL;
  constexpr uint64_t prime = 1099511628211ULL;
  uint64_t hash = offset_basis;
  if (!path)
    return hash;
  for (const unsigned char *p = reinterpret_cast<const unsigned char *>(path);
       *p != '\0'; ++p) {
    hash ^= static_cast<uint64_t>(*p);
    hash *= prime;
  }
  return hash;
}

uint64_t oldHeadnodePathKey(std::string_view Path) {
  constexpr uint64_t OffsetBasis = 1469598103934665603ULL;
  constexpr uint64_t Prime = 1099511628211ULL;
  uint64_t Hash = OffsetBasis;
  for (const unsigned char C : Path) {
    Hash ^= static_cast<uint64_t>(C);
    Hash *= Prime;
  }
  return Hash;
}

uint64_t oldMixHint(uint64_t base_key, const ompfile::OmpFileIOHint &hint) {
  uint64_t mixed = base_key;
  auto mix = [&mixed](uint64_t value) {
    mixed ^= value + 0x9e3779b97f4a7c15ULL + (mixed << 6) + (mixed >> 2);
  };
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_STREAM) != 0)
    mix(hint.StreamId);
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_EPOCH) != 0)
    mix(hint.EpochId);
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0)
    mix(hint.TileId);
  return mixed;
}

uint64_t oldMixWriteHint(uint64_t base_key,
                         const ompfile::OmpFileIOHint &hint) {
  uint64_t mixed = base_key;
  auto mix = [&mixed](uint64_t value) {
    mixed ^= value + 0x9e3779b97f4a7c15ULL + (mixed << 6) + (mixed >> 2);
  };
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_STREAM) != 0)
    mix(hint.StreamId);
  if ((hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) != 0)
    mix(hint.TileId);
  return mixed;
}

uint64_t oldClientTileKey(uint64_t PathKey, const ompfile::OmpFileIOHint &Hint) {
  if (PathKey == 0 ||
      (Hint.HintFlags & ompfile::OMPFILE_IO_HINT_HAS_TILE) == 0)
    return PathKey;
  uint64_t X = PathKey ^ 0x9e3779b97f4a7c15ULL;
  X ^= Hint.TileId + 0x9e3779b97f4a7c15ULL + (X << 6) + (X >> 2);
  return X != 0 ? X : PathKey;
}

uint64_t oldProxyTileKey(uint64_t PathKey, uint64_t TileId) {
  if (PathKey == 0)
    return 0;
  uint64_t X = PathKey ^ 0x9e3779b97f4a7c15ULL;
  X ^= TileId + 0x9e3779b97f4a7c15ULL + (X << 6) + (X >> 2);
  return X != 0 ? X : PathKey;
}

constexpr const char *kOracle3168 =
    "/scratch/rodrigo.freitas/io-stage/results/job-3168/"
    "mpp-global-oracle-coherence/global-oracle-coherence.bin";
constexpr const char *kOracle3169 =
    "/scratch/rodrigo.freitas/io-stage/results/job-3169/"
    "mpp-global-oracle-coherence/global-oracle-coherence.bin";
constexpr const char *kCholesky3170 =
    "/scratch/rodrigo.freitas/io-stage/results/job-3170/"
    "ompc-cholesky-files-mpp-compare/optimized/ompc-cholesky-files-mpp.bin";

std::vector<std::string> pathCorpus() {
  std::vector<std::string> Corpus = {"",
                                     "a",
                                     "w",
                                     "r",
                                     "tile-q",
                                     "tile-write-commit",
                                     "tile-dirty-flush",
                                     "/tmp/x.bin",
                                     "a//b",
                                     "a/b",
                                     kOracle3168,
                                     kOracle3169,
                                     kCholesky3170};
  Corpus.push_back(std::string(4096, 'z'));
  // High bytes: pins the char -> unsigned char conversion.
  Corpus.push_back(std::string{'\xc3', '\xa9', 't', '\x80', '\xff'});
  return Corpus;
}

const uint64_t kKeys[] = {0,
                          1,
                          2,
                          0xffffffffULL,
                          0x9e3779b97f4a7c15ULL,
                          1469598103934665603ULL,
                          14764257196743165094ULL,
                          std::numeric_limits<uint64_t>::max()};
const uint64_t kTiles[] = {0,  1,   2,
                           7,  10,  101,
                           0x7fffffffffffffffULL,
                           std::numeric_limits<uint64_t>::max()};

ompfile::OmpFileIOHint hintWith(uint32_t Flags, uint64_t Epoch,
                                uint64_t Stream, uint64_t Tile) {
  ompfile::OmpFileIOHint H{};
  H.HintFlags = Flags;
  H.EpochId = Epoch;
  H.StreamId = Stream;
  H.TileId = Tile;
  return H;
}

} // namespace

using namespace ompfile::keys;

static_assert(pathKey("") == 1469598103934665603ULL);
static_assert(tileKeyForHint(5, ompfile::OmpFileIOHint{}) == 5);

OMPFILE_TEST(keys_path_key_matches_legacy_copies) {
  for (const std::string &S : pathCorpus()) {
    OMPFILE_CHECK_EQ(pathKey(std::string_view(S)), oldHeadnodePathKey(S));
    OMPFILE_CHECK_EQ(pathKey(S.c_str()), oldClientPathKey(S.c_str()));
    OMPFILE_CHECK_EQ(pathKey(S), pathKey(S.c_str()));
  }
  OMPFILE_CHECK_EQ(pathKey(static_cast<const char *>(nullptr)),
                   oldClientPathKey(nullptr));
}

OMPFILE_TEST(keys_path_key_embedded_nul) {
  const std::string N("ab\0cd", 5);
  OMPFILE_CHECK_EQ(pathKey(std::string_view(N)), oldHeadnodePathKey(N));
  OMPFILE_CHECK_EQ(pathKey(N.c_str()), pathKey("ab"));
  OMPFILE_CHECK(pathKey(std::string_view(N)) != pathKey("ab"));
}

OMPFILE_TEST(keys_tile_key_matches_legacy_copies) {
  for (const uint64_t K : kKeys) {
    for (const uint64_t T : kTiles) {
      OMPFILE_CHECK_EQ(tileKey(K, T), oldProxyTileKey(K, T));
      // Every combination of HAS_EPOCH | HAS_STREAM | HAS_TILE | HAS_ROLE.
      for (uint32_t F = 0; F < 16; ++F) {
        const ompfile::OmpFileIOHint H = hintWith(F, 3, 5, T);
        OMPFILE_CHECK_EQ(tileKeyForHint(K, H), oldClientTileKey(K, H));
      }
    }
  }
}

OMPFILE_TEST(keys_group_keys_match_legacy_mixes) {
  for (const uint64_t K : kKeys) {
    for (uint32_t F = 0; F < 16; ++F) {
      const ompfile::OmpFileIOHint A = hintWith(F, 3, 5, 9);
      const ompfile::OmpFileIOHint B = hintWith(F, 9, 3, 5);
      OMPFILE_CHECK_EQ(readGroupKey(K, A), oldMixHint(K, A));
      OMPFILE_CHECK_EQ(readGroupKey(K, B), oldMixHint(K, B));
      OMPFILE_CHECK_EQ(writeGroupKey(K, A), oldMixWriteHint(K, A));
      OMPFILE_CHECK_EQ(writeGroupKey(K, B), oldMixWriteHint(K, B));
    }
  }
}

OMPFILE_TEST(keys_handle_key_zero_extends) {
  OMPFILE_CHECK_EQ(handleKey(0), 0ULL);
  OMPFILE_CHECK_EQ(handleKey(7), 7ULL);
  OMPFILE_CHECK_EQ(handleKey(-1), 0xffffffffULL);
  OMPFILE_CHECK_EQ(handleKey(std::numeric_limits<int32_t>::min()),
                   0x80000000ULL);
}

OMPFILE_TEST(keys_match_sorgan_job_logs) {
  OMPFILE_CHECK_EQ(pathKey(kOracle3168), 14764257196743165094ULL);
  OMPFILE_CHECK_EQ(tileKey(pathKey(kOracle3168), 0), 3842491440462061490ULL);
  OMPFILE_CHECK_EQ(tileKey(pathKey(kOracle3168), 1), 3842491440462061489ULL);
  OMPFILE_CHECK_EQ(pathKey(kOracle3169), 2397149331531985981ULL);
  OMPFILE_CHECK_EQ(tileKey(pathKey(kOracle3169), 0), 1560517660896313399ULL);
  OMPFILE_CHECK_EQ(tileKey(pathKey(kOracle3169), 1), 1560517660896313352ULL);
  OMPFILE_CHECK_EQ(pathKey(kCholesky3170), 10033430449299206809ULL);
  OMPFILE_CHECK_EQ(tileKey(pathKey(kCholesky3170), 0),
                   17522137244162309172ULL);
  OMPFILE_CHECK_EQ(tileKey(pathKey(kCholesky3170), 1),
                   17522137244162309173ULL);
  OMPFILE_CHECK_EQ(tileKey(pathKey(kCholesky3170), 10),
                   17522137244162309198ULL);
  OMPFILE_CHECK_EQ(tileKey(pathKey(kCholesky3170), 101),
                   17522137244162309521ULL);
}
