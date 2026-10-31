// Native unit tests for the resource initializer's upload budget (src/dxmt/dxmt_upload_budget.hpp).
#include <cstdio>
#include "dxmt_upload_budget.hpp"

using namespace dxmt;

static int failures = 0;
#define CHECK(cond)                                                                                                   \
  do {                                                                                                                \
    if (!(cond)) {                                                                                                    \
      ++failures;                                                                                                     \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                                                   \
    }                                                                                                                 \
  } while (0)

int main() {
  constexpr size_t MB = 1 << 20;
  // nothing queued: never flush, even for an upload bigger than the budget (it goes alone)
  CHECK(!upload_needs_flush(0, 4 * kUploadFlushBytes, kUploadFlushBytes));
  // within the budget: keep batching
  CHECK(!upload_needs_flush(64 * MB, 64 * MB, 128 * MB));
  // the upload would push the queued bytes over the budget: submit what is queued first
  CHECK(upload_needs_flush(64 * MB + 1, 64 * MB, 128 * MB));
  CHECK(upload_needs_flush(100 * MB, 64 * MB, 128 * MB));
  // a level load of 64 MB textures is submitted every two textures, not once at the end
  size_t pending = 0;
  int flushes = 0;
  for (int i = 0; i < 180; i++) {
    if (upload_needs_flush(pending, 64 * MB, kUploadFlushBytes)) {
      flushes++;
      pending = 0;
    }
    pending += 64 * MB;
  }
  CHECK(flushes >= (int)(180 * 64 * MB / kUploadFlushBytes) - 1);
  // the budget stays well under the arena, so a submitted batch can be reused from it
  CHECK(kUploadFlushBytes <= 128 * MB);
  std::printf("%s upload_budget\n", failures ? "FAIL" : "ok  ");
  return failures ? 1 : 0;
}
