#ifndef OMPFILE_EXTENTS_H
#define OMPFILE_EXTENTS_H

// Byte-range extent algebra of the proxy stage: which [Begin, End) ranges of
// a staged file are covered, merged as they are added, split as they are
// removed. Pure functions over a sorted, non-overlapping vector, moved out
// of ProxyDevice so they can be unit-tested; ProxyDevice keeps thin
// forwarders at its call sites.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace ompfile {
namespace extents {

struct Extent {
  uint64_t Begin = 0;
  uint64_t End = 0;
};

inline uint64_t saturatingAdd(uint64_t Base, uint64_t Delta) {
  if (Delta > std::numeric_limits<uint64_t>::max() - Base)
    return std::numeric_limits<uint64_t>::max();
  return Base + Delta;
}

inline uint64_t alignDown(uint64_t Value, uint64_t Alignment) {
  if (Alignment == 0)
    return Value;
  return (Value / Alignment) * Alignment;
}

inline uint64_t alignUp(uint64_t Value, uint64_t Alignment) {
  if (Alignment == 0)
    return Value;
  const uint64_t Remainder = Value % Alignment;
  if (Remainder == 0)
    return Value;
  const uint64_t Delta = Alignment - Remainder;
  return saturatingAdd(Value, Delta);
}

// True when [Begin, End) is fully inside the union of Extents (sorted by
// Begin, non-overlapping). An empty range is trivially covered.
inline bool isCoveredByExtents(const std::vector<Extent> &Extents,
                               uint64_t Begin, uint64_t End) {
  if (End <= Begin)
    return true;
  uint64_t Cursor = Begin;
  for (const Extent &E : Extents) {
    if (E.End <= Cursor)
      continue;
    if (E.Begin > Cursor)
      return false;
    Cursor = std::max(Cursor, E.End);
    if (Cursor >= End)
      return true;
  }
  return Cursor >= End;
}

// True when any extent intersects [Begin, End).
inline bool overlapsAnyExtent(const std::vector<Extent> &Extents,
                              uint64_t Begin, uint64_t End) {
  if (End <= Begin)
    return false;
  for (const Extent &E : Extents) {
    if (E.End <= Begin)
      continue;
    if (E.Begin >= End)
      return false;
    return true;
  }
  return false;
}

// Insert [Begin, End) and merge touching or overlapping extents.
inline void addCoveredExtent(std::vector<Extent> &Extents, uint64_t Begin,
                             uint64_t End) {
  if (End <= Begin)
    return;
  Extent NewExtent{Begin, End};
  Extents.push_back(NewExtent);
  std::sort(Extents.begin(), Extents.end(),
            [](const Extent &Lhs, const Extent &Rhs) {
              return Lhs.Begin < Rhs.Begin;
            });
  std::vector<Extent> Merged;
  Merged.reserve(Extents.size());
  for (const Extent &E : Extents) {
    if (Merged.empty() || E.Begin > Merged.back().End) {
      Merged.push_back(E);
      continue;
    }
    Merged.back().End = std::max(Merged.back().End, E.End);
  }
  Extents.swap(Merged);
}

// Remove [Begin, End) from the covered set, splitting extents that straddle
// it. Returns how many covered bytes were removed.
inline uint64_t removeCoveredRange(std::vector<Extent> &Extents,
                                   uint64_t Begin, uint64_t End) {
  if (End <= Begin || Extents.empty())
    return 0;

  uint64_t RemovedBytes = 0;
  std::vector<Extent> Updated;
  Updated.reserve(Extents.size());
  for (const Extent &E : Extents) {
    if (E.End <= Begin || E.Begin >= End) {
      Updated.push_back(E);
      continue;
    }
    const uint64_t OverlapBegin = std::max(E.Begin, Begin);
    const uint64_t OverlapEnd = std::min(E.End, End);
    if (OverlapEnd > OverlapBegin)
      RemovedBytes += (OverlapEnd - OverlapBegin);

    if (E.Begin < Begin)
      Updated.push_back(Extent{E.Begin, Begin});
    if (E.End > End)
      Updated.push_back(Extent{End, E.End});
  }
  Extents.swap(Updated);
  return RemovedBytes;
}

} // namespace extents
} // namespace ompfile

#endif // OMPFILE_EXTENTS_H
