#pragma once

// When the resource initializer submits the uploads it has queued. Games load levels by creating hundreds of
// textures with initial data and draw nothing meanwhile; without a budget every upload waited for the same
// submission and none of its staging memory could be reused (Crysis 2 Remastered: 181 x 64 MB, 11 GB).
// Standard library only: unit-tested natively (tests/unit/upload_budget_test.cpp).

#include <cstddef>

namespace dxmt {

// Bytes of queued upload data after which the queued uploads are submitted before the next one.
constexpr size_t kUploadFlushBytes = 0x8000000; // 128 MB

// True when `size` more bytes would push `pending` queued bytes over `budget`. An upload bigger than the budget
// is queued alone (nothing pending) rather than refused.
constexpr bool
upload_needs_flush(size_t pending, size_t size, size_t budget) {
  return pending != 0 && pending + size > budget;
}

} // namespace dxmt
