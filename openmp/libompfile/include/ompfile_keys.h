#ifndef OMPFILE_KEYS_H
#define OMPFILE_KEYS_H

// The keys libompfile, the proxy and the headnode derive from a path, a tile,
// a hint or a handle: one definition per kind (Oct 2026). Path and tile keys
// cross process boundaries (OmpFileFreshnessQueryRequest::PathKey, the
// freshness/flush/commit bridge calls, OmpFileIOBatchSegment::PathKey) and each
// side recomputes them from its own copy of the path, so a change on one side
// only is a silent coherence break. tests/keys_test.cpp pins every function
// against the bodies it replaced and against keys logged by real runs: never
// change a constant, an operand order or a guard.

#include "ompfile_sched.h"

#include <cstdint>
#include <string_view>

namespace ompfile {
namespace keys {

// NOT the standard FNV-1a 64-bit offset basis (14695981039346656037): every
// copy has always used it with its last decimal digit missing. Keep it.
inline constexpr uint64_t kPathKeyOffsetBasis = 1469598103934665603ULL;
inline constexpr uint64_t kPathKeyPrime = 1099511628211ULL; // FNV-64 prime
inline constexpr uint64_t kHashCombineConstant = 0x9e3779b97f4a7c15ULL;
static_assert(kPathKeyPrime == 0x100000001b3ULL);

// boost::hash_combine on 64 bits.
constexpr uint64_t hashCombine(uint64_t Seed, uint64_t Value) noexcept {
  return Seed ^ (Value + kHashCombineConstant + (Seed << 6) + (Seed >> 2));
}

// Path key: FNV-1a (with the basis above) over every byte of Path exactly as
// given. No normalization (a//b and a/b are different keys); embedded NULs
// count.
constexpr uint64_t pathKey(std::string_view Path) noexcept {
  uint64_t Hash = kPathKeyOffsetBasis;
  for (const char C : Path) {
    Hash ^= static_cast<uint64_t>(static_cast<unsigned char>(C));
    Hash *= kPathKeyPrime;
  }
  return Hash;
}

// C string: the bytes before the first NUL; a null pointer hashes like the
// empty path (the basis).
constexpr uint64_t pathKey(const char *Path) noexcept {
  return Path ? pathKey(std::string_view(Path)) : kPathKeyOffsetBasis;
}

// Base key of a handle with no tracked path: its 32 bits, zero-extended.
constexpr uint64_t handleKey(int FileHandle) noexcept {
  return static_cast<uint64_t>(static_cast<uint32_t>(FileHandle));
}

// Tile key: the key of one tile in the headnode TileFreshnessTable. A zero
// path key stays zero; a mix that lands on zero falls back to PathKey. Not
// the group mixes below: it pre-XORs PathKey with the constant.
constexpr uint64_t tileKey(uint64_t PathKey, uint64_t TileId) noexcept {
  if (PathKey == 0)
    return 0;
  const uint64_t X = hashCombine(PathKey ^ kHashCombineConstant, TileId);
  return X != 0 ? X : PathKey;
}

// Client form: a hint without HAS_TILE leaves PathKey unchanged.
constexpr uint64_t tileKeyForHint(uint64_t PathKey,
                                  const OmpFileIOHint &Hint) noexcept {
  if ((Hint.HintFlags & OMPFILE_IO_HINT_HAS_TILE) == 0)
    return PathKey;
  return tileKey(PathKey, Hint.TileId);
}

// Two-phase read group key: stream, epoch, tile, in that order. Sent to the
// headnode as OmpFileIOBatchSegment::PathKey.
constexpr uint64_t readGroupKey(uint64_t BaseKey,
                                const OmpFileIOHint &Hint) noexcept {
  uint64_t Key = BaseKey;
  if ((Hint.HintFlags & OMPFILE_IO_HINT_HAS_STREAM) != 0)
    Key = hashCombine(Key, Hint.StreamId);
  if ((Hint.HintFlags & OMPFILE_IO_HINT_HAS_EPOCH) != 0)
    Key = hashCombine(Key, Hint.EpochId);
  if ((Hint.HintFlags & OMPFILE_IO_HINT_HAS_TILE) != 0)
    Key = hashCombine(Key, Hint.TileId);
  return Key;
}

// Write-batch group key: stream, tile; the epoch is deliberately not mixed.
// Process-local.
constexpr uint64_t writeGroupKey(uint64_t BaseKey,
                                 const OmpFileIOHint &Hint) noexcept {
  uint64_t Key = BaseKey;
  if ((Hint.HintFlags & OMPFILE_IO_HINT_HAS_STREAM) != 0)
    Key = hashCombine(Key, Hint.StreamId);
  if ((Hint.HintFlags & OMPFILE_IO_HINT_HAS_TILE) != 0)
    Key = hashCombine(Key, Hint.TileId);
  return Key;
}

static_assert(pathKey(std::string_view{}) == kPathKeyOffsetBasis);
static_assert(pathKey(static_cast<const char *>(nullptr)) ==
              kPathKeyOffsetBasis);
static_assert(tileKey(0, 1) == 0);
static_assert(handleKey(-1) == 0xffffffffULL);

} // namespace keys
} // namespace ompfile

#endif // OMPFILE_KEYS_H
