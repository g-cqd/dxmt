// Native unit tests for RingBumpStateImpl (src/dxmt/dxmt_ring_bump_state.hpp).
// Build and run: tests/unit/run.sh   (plain clang++ on the host, no Wine or Metal needed)
//
// The GPU is simulated: chunk `seq` is encoded by the CPU while the GPU has completed every chunk up to
// `coherent = seq - lag`; the queue's finishing thread calls free_blocks(coherent) once per chunk.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

static int g_new_block_logs = 0;
#define DXMT_RING_BUMP_LOG_NEW_BLOCK(size) (++g_new_block_logs, (void)(size))
#include "dxmt_ring_bump_state.hpp"

using namespace dxmt;

struct FakeStats {
  int created = 0;
  int live = 0;
  size_t live_bytes = 0;
  size_t peak_bytes = 0;
  int next_id = 1;
};

class FakeAllocator {
public:
  explicit FakeAllocator(FakeStats *stats) : stats_(stats) {}

  class Block {
  public:
    FakeStats *stats = nullptr;
    size_t size = 0;
    int id = 0;

    Block() = default;
    Block(const Block &) = delete;
    Block(Block &&m) : stats(m.stats), size(m.size), id(m.id) { m.stats = nullptr; }
    ~Block() {
      if (stats) {
        stats->live--;
        stats->live_bytes -= size;
      }
    }
  };

  Block allocate(size_t size) {
    Block b;
    b.stats = stats_;
    b.size = size;
    b.id = stats_->next_id++;
    stats_->created++;
    stats_->live++;
    stats_->live_bytes += size;
    if (stats_->live_bytes > stats_->peak_bytes)
      stats_->peak_bytes = stats_->live_bytes;
    return b;
  }

private:
  FakeStats *stats_;
};

constexpr size_t kBlock = 1024;
struct NullMutex {
  void lock() {}
  void unlock() noexcept {}
};
using State = RingBumpStateImpl<FakeAllocator, kBlock, NullMutex>;

static int g_failures = 0;
static int g_checks = 0;
#define CHECK(cond, ...)                                                                                              \
  do {                                                                                                                \
    ++g_checks;                                                                                                       \
    if (!(cond)) {                                                                                                    \
      ++g_failures;                                                                                                   \
      std::printf("  FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond);                                                 \
      std::printf(__VA_ARGS__);                                                                                       \
      std::printf("\n");                                                                                              \
    }                                                                                                                 \
  } while (0)

struct Fixture {
  FakeStats stats;
  State state{FakeAllocator(&stats)};
};

// --- correctness: what the allocator must never do -------------------------------------------------------------

// Two allocations made for chunks the GPU has not finished never overlap, and offsets honour the alignment.
static void test_no_overlap_while_in_flight() {
  Fixture f;
  struct Range { int id; uint64_t begin, end; };
  std::vector<Range> ranges;
  const size_t sizes[] = {100, 300, 7, 1024, 512, 2048, 64, 900, 4096, 1};
  for (int i = 0; i < 40; i++) {
    size_t size = sizes[i % 10];
    size_t alignment = (i % 3 == 0) ? 256 : 16;
    auto [block, offset] = f.state.allocate(100 + i / 4, 50, size, alignment);  // GPU stuck at 50: nothing completes
    CHECK(offset % alignment == 0, "offset %llu not aligned to %zu", (unsigned long long)offset, alignment);
    CHECK(offset + size <= block.size, "allocation [%llu,+%zu) exceeds block of %zu", (unsigned long long)offset, size,
          block.size);
    for (auto &r : ranges)
      if (r.id == block.id)
        CHECK(offset >= r.end || offset + size <= r.begin, "overlap in block %d", block.id);
    ranges.push_back({block.id, offset, offset + size});
  }
}

// A large block (bigger than a regular block) is not handed out again while the GPU may still read it.
static void test_large_block_not_reused_while_in_flight() {
  Fixture f;
  int first = f.state.allocate(10, 5, 4 * kBlock, 16).first.id;
  f.state.free_blocks(5);
  int second = f.state.allocate(11, 5, 4 * kBlock, 16).first.id;
  CHECK(first != second, "block %d reused while chunk 10 is still in flight", first);
}

// A large request gets a block that fits it.
static void test_large_block_fits_request() {
  Fixture f;
  auto [b1, o1] = f.state.allocate(1, 0, 3 * kBlock / 2, 16);
  CHECK(o1 + 3 * kBlock / 2 <= b1.size, "1.5-block request got a block of %zu", b1.size);
  auto [b2, o2] = f.state.allocate(2, 0, 3 * kBlock, 16);
  CHECK(o2 + 3 * kBlock <= b2.size, "3-block request got a block of %zu", b2.size);
}

// free_blocks(~0) (context or queue teardown) releases everything.
static void test_teardown_releases_everything() {
  Fixture f;
  for (int i = 0; i < 30; i++)
    f.state.allocate(1 + i, 0, (i % 2) ? 4 * kBlock : 300, 16);
  f.state.free_blocks(~0ull);
  CHECK(f.stats.live == 0, "%d blocks still alive after teardown", f.stats.live);
}

// The deferred-context pattern: seq 1 and coherent 0 forever, then teardown. Nothing is reused or lost early.
static void test_deferred_context_pattern() {
  Fixture f;
  std::set<std::pair<int, uint64_t>> seen;
  for (int i = 0; i < 50; i++) {
    auto [block, offset] = f.state.allocate(1, 0, (i % 5 == 0) ? 3 * kBlock : 200, 16);
    CHECK(seen.insert({block.id, offset}).second, "same (block %d, offset %llu) returned twice", block.id,
          (unsigned long long)offset);
  }
  f.state.free_blocks(~0ull);
  CHECK(f.stats.live == 0, "%d blocks alive after teardown", f.stats.live);
}

// --- the 2026-10-31 Crysis 2 Remastered leak: large uploads ----------------------------------------------------

// Steady streaming: every chunk uploads one large texture (64 MB in the game) plus small data, GPU two chunks
// behind. The large blocks must be recycled: a handful of block creations, not one per upload.
static void test_large_uploads_are_recycled() {
  Fixture f;
  const uint64_t lag = 2;
  for (uint64_t seq = 10; seq < 2010; seq++) {
    uint64_t coherent = seq - lag;
    f.state.allocate(seq, coherent, 200, 16);
    f.state.allocate(seq, coherent, 2 * kBlock, 16);
    f.state.allocate(seq, coherent, 300, 16);
    f.state.free_blocks(coherent);
  }
  CHECK(f.stats.created <= 12, "%d blocks created for 2000 large uploads (expected reuse)", f.stats.created);
  CHECK(g_new_block_logs <= 12, "%d 'forced to allocate' logs", g_new_block_logs);
}

// A burst of consecutive large uploads behind an idle regular block (the game's level streaming). Finished
// large blocks must be released or recycled once the GPU is done, not held until that regular block expires.
static void test_large_uploads_not_held_behind_idle_block() {
  Fixture f;
  const uint64_t lag = 2;
  f.state.allocate(9, 0, 100, 16);  // one regular block, then only large uploads
  for (uint64_t seq = 10; seq < 1010; seq++) {
    uint64_t coherent = seq - lag;
    for (int k = 0; k < 3; k++)
      f.state.allocate(seq, coherent, 2 * kBlock, 16);
    f.state.free_blocks(coherent);
  }
  // in flight: (lag + 1) chunks x 3 uploads x 2 blocks, plus the regular block, plus a small recycled pool
  const size_t bound = (lag + 1) * 3 * 2 * kBlock + kBlock + kLargePoolCapBlocks * kBlock;
  CHECK(f.stats.peak_bytes <= bound, "peak %zu bytes alive, bound %zu (finished uploads held back)",
        f.stats.peak_bytes, bound);
}

// After a spike of large uploads the recycled pool is trimmed: bounded right away, empty after the lifetime.
static void test_large_pool_trimmed_after_spike() {
  Fixture f;
  for (int i = 0; i < 64; i++)
    f.state.allocate(100, 99, 2 * kBlock, 16);  // 64 uploads in one chunk
  f.state.free_blocks(100);                     // chunk 100 done
  CHECK(f.stats.live_bytes <= kLargePoolCapBlocks * kBlock + kBlock, "%zu bytes kept right after the spike",
        f.stats.live_bytes);
  for (uint64_t c = 101; c < 101 + 2 * kStagingBlockLifetime + 2; c++)
    f.state.free_blocks(c);
  CHECK(f.stats.live == 0, "%d blocks still alive long after the spike", f.stats.live);
}

// A recycled large block is reused for a later upload of the same size once the GPU finished with it.
static void test_large_block_reused_after_completion() {
  Fixture f;
  int first = f.state.allocate(10, 5, 2 * kBlock, 16).first.id;
  f.state.free_blocks(10);
  int second = f.state.allocate(12, 11, 2 * kBlock, 16).first.id;
  CHECK(first == second, "finished block %d not reused (got %d)", first, second);
  CHECK(f.stats.created == 1, "%d blocks created", f.stats.created);
}

// A recycled block is only reused for a request it can hold.
static void test_large_block_reuse_respects_size() {
  Fixture f;
  f.state.allocate(10, 5, 2 * kBlock, 16);
  f.state.free_blocks(10);
  auto [block, offset] = f.state.allocate(12, 11, 5 * kBlock, 16);
  CHECK(offset + 5 * kBlock <= block.size, "5-block request served by a block of %zu", block.size);
}

// --- regular blocks: the ring ----------------------------------------------------------------------------------

// Steady state: the regular blocks form a ring of about (lag + 1) blocks; no growth over time.
static void test_ring_steady_state_is_bounded() {
  Fixture f;
  const uint64_t lag = 3;
  for (uint64_t seq = 10; seq < 5010; seq++) {
    uint64_t coherent = seq - lag;
    for (int k = 0; k < 6; k++)
      f.state.allocate(seq, coherent, 100, 16);  // ~0.6 block per chunk
    f.state.free_blocks(coherent);
  }
  CHECK(f.stats.created <= (int)lag + 3, "%d regular blocks created in steady state", f.stats.created);
}

// After a spike (the GPU stalled while the CPU kept encoding) the extra regular blocks are trimmed within the
// lifetime, not one block per lifetime.
static void test_ring_trimmed_after_spike() {
  Fixture f;
  for (int k = 0; k < 40 * 8; k++)
    f.state.allocate(100, 50, kBlock / 8, 16);  // 40 blocks for one chunk, GPU stalled
  CHECK(f.stats.live >= 40, "spike setup: only %d blocks", f.stats.live);
  for (uint64_t c = 100; c < 100 + 2 * kStagingBlockLifetime + 2; c++)
    f.state.free_blocks(c);
  CHECK(f.stats.live <= 1, "%d regular blocks still alive %zu chunks after the spike", f.stats.live,
        2 * kStagingBlockLifetime + 2);
}

// Regular blocks still in use are never released by trimming.
static void test_ring_keeps_blocks_in_flight() {
  Fixture f;
  auto [block, offset] = f.state.allocate(500, 400, 100, 16);
  int id = block.id;
  for (uint64_t c = 0; c < 499; c++)
    f.state.free_blocks(c);  // the GPU never reaches chunk 500
  CHECK(f.stats.live == 1, "in-flight block released (%d alive)", f.stats.live);
  auto [again, off2] = f.state.allocate(500, 499, 100, 16);
  CHECK(again.id == id && off2 > offset, "same chunk should keep bumping in block %d", id);
}

int main() {
  struct { const char *name; void (*fn)(); } tests[] = {
      {"no_overlap_while_in_flight", test_no_overlap_while_in_flight},
      {"large_block_not_reused_while_in_flight", test_large_block_not_reused_while_in_flight},
      {"large_block_fits_request", test_large_block_fits_request},
      {"teardown_releases_everything", test_teardown_releases_everything},
      {"deferred_context_pattern", test_deferred_context_pattern},
      {"large_uploads_are_recycled", test_large_uploads_are_recycled},
      {"large_uploads_not_held_behind_idle_block", test_large_uploads_not_held_behind_idle_block},
      {"large_pool_trimmed_after_spike", test_large_pool_trimmed_after_spike},
      {"large_block_reused_after_completion", test_large_block_reused_after_completion},
      {"large_block_reuse_respects_size", test_large_block_reuse_respects_size},
      {"ring_steady_state_is_bounded", test_ring_steady_state_is_bounded},
      {"ring_trimmed_after_spike", test_ring_trimmed_after_spike},
      {"ring_keeps_blocks_in_flight", test_ring_keeps_blocks_in_flight},
  };
  int failed_tests = 0;
  for (auto &t : tests) {
    int before = g_failures;
    g_new_block_logs = 0;
    t.fn();
    bool ok = g_failures == before;
    failed_tests += !ok;
    std::printf("%s %s\n", ok ? "ok  " : "FAIL", t.name);
  }
  std::printf("%d/%zu tests passed, %d checks\n", (int)(sizeof(tests) / sizeof(tests[0])) - failed_tests,
              sizeof(tests) / sizeof(tests[0]), g_checks);
  return failed_tests ? 1 : 0;
}
