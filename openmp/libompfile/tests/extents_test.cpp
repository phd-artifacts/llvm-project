// The stage extent algebra in ompfile_extents.h.

#include "ompfile_extents.h"
#include "ompfile_test.h"

#include <limits>
#include <vector>

using namespace ompfile::extents;

OMPFILE_TEST(add_merges_adjacent_and_overlapping) {
  std::vector<Extent> E;
  addCoveredExtent(E, 0, 10);
  addCoveredExtent(E, 20, 30);
  OMPFILE_CHECK_EQ(E.size(), 2u);
  addCoveredExtent(E, 10, 20); // touches both: one extent
  OMPFILE_CHECK_EQ(E.size(), 1u);
  OMPFILE_CHECK_EQ(E[0].Begin, 0u);
  OMPFILE_CHECK_EQ(E[0].End, 30u);
  addCoveredExtent(E, 25, 40); // overlaps the tail
  OMPFILE_CHECK_EQ(E.size(), 1u);
  OMPFILE_CHECK_EQ(E[0].End, 40u);
  addCoveredExtent(E, 5, 5); // empty: ignored
  OMPFILE_CHECK_EQ(E.size(), 1u);
}

OMPFILE_TEST(coverage_and_overlap_queries) {
  std::vector<Extent> E;
  addCoveredExtent(E, 0, 10);
  addCoveredExtent(E, 20, 30);
  OMPFILE_CHECK(isCoveredByExtents(E, 0, 10));
  OMPFILE_CHECK(isCoveredByExtents(E, 2, 8));
  OMPFILE_CHECK(!isCoveredByExtents(E, 5, 25)); // the gap [10, 20)
  OMPFILE_CHECK(isCoveredByExtents(E, 7, 7));    // empty range
  OMPFILE_CHECK(overlapsAnyExtent(E, 5, 15));
  OMPFILE_CHECK(!overlapsAnyExtent(E, 10, 20));
  OMPFILE_CHECK(!overlapsAnyExtent(E, 30, 40));
  OMPFILE_CHECK(!overlapsAnyExtent(E, 4, 4));
}

OMPFILE_TEST(remove_splits_and_counts_bytes) {
  std::vector<Extent> E;
  addCoveredExtent(E, 0, 100);
  OMPFILE_CHECK_EQ(removeCoveredRange(E, 40, 60), 20u);
  OMPFILE_CHECK_EQ(E.size(), 2u);
  OMPFILE_CHECK_EQ(E[0].End, 40u);
  OMPFILE_CHECK_EQ(E[1].Begin, 60u);
  OMPFILE_CHECK_EQ(removeCoveredRange(E, 50, 55), 0u); // already a gap
  OMPFILE_CHECK_EQ(removeCoveredRange(E, 30, 70), 20u); // [30,40) + [60,70)
  OMPFILE_CHECK_EQ(E.size(), 2u);
  OMPFILE_CHECK_EQ(E[0].End, 30u);
  OMPFILE_CHECK_EQ(E[1].Begin, 70u);
  OMPFILE_CHECK_EQ(removeCoveredRange(E, 0, 1000), 60u);
  OMPFILE_CHECK(E.empty());
  OMPFILE_CHECK_EQ(removeCoveredRange(E, 0, 10), 0u);
}

OMPFILE_TEST(alignment_and_saturation) {
  const uint64_t Max = std::numeric_limits<uint64_t>::max();
  OMPFILE_CHECK_EQ(saturatingAdd(Max - 1, 5), Max);
  OMPFILE_CHECK_EQ(saturatingAdd(10, 5), 15u);
  OMPFILE_CHECK_EQ(alignDown(4097, 4096), 4096u);
  OMPFILE_CHECK_EQ(alignDown(4097, 0), 4097u);
  OMPFILE_CHECK_EQ(alignUp(4097, 4096), 8192u);
  OMPFILE_CHECK_EQ(alignUp(4096, 4096), 4096u);
  OMPFILE_CHECK_EQ(alignUp(Max - 1, 4096), Max); // saturates
}
