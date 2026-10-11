#pragma once

#include <atomic>
#include <cstdint>

namespace dxmt {

class Allocation {
public:
  virtual ~Allocation(){};

  void incRef();
  void decRef();

  bool
  checkRetained(uint64_t seq_id) {
    // FIXME: is a compare-and-swap necessary?
    if (seq_id == last_retained_seq_id)
      return true;
    last_retained_seq_id = seq_id;
    return false;
  }

private:
  std::atomic<uint32_t> refcount_ = {0u};
  uint64_t last_retained_seq_id = 0;
};

class AllocationRefTracking {
public:
  AllocationRefTracking();

  bool track(Allocation *allocation);

  void addStorage(void *ptr, size_t length);

  void clear();

private:
  template <size_t Size = 1> struct RefAddChunk {
    RefAddChunk *next_chunk;
    size_t capacity;
    size_t size;
    Allocation *allocations[Size];
  };
  RefAddChunk<29> chunk_placed;
  RefAddChunk<> *chunk_last = nullptr;
};

/**
 * Bounds the CPU-visible copies a resource makes by renaming: a WRITE_DISCARD map
 * of a dynamic resource, or a write map of a staging resource the GPU still reads.
 * A retired copy is only reused once the GPU has finished with it, and nothing in
 * the rename path flushes, so a burst of maps without a flush grows the pool by one
 * copy per map.
 *
 * In a 32-bit process every copy is placed in the 4 GB address space, so a 4 MB
 * staging buffer mapped a few hundred times during a load exhausts it. There the
 * renamer must wait for the GPU to retire its oldest copy once it holds
 * kPerResource bytes of copies, or once all renamers together hold kTotal bytes.
 * 64-bit processes are not bounded.
 */
class RenameBudget {
public:
#ifdef __i386__
  static constexpr uint64_t kPerResource = 64ull << 20;
  static constexpr uint64_t kTotal = 512ull << 20;
#else
  static constexpr uint64_t kPerResource = ~0ull;
  static constexpr uint64_t kTotal = ~0ull;
#endif

  /**
   * Whether a resource that already holds `held` bytes of copies may make another
   * copy of `length` bytes instead of waiting. The first copy is always allowed.
   */
  bool
  allows(uint64_t held, uint64_t length) const {
    if (held == 0)
      return true;
    if (held + length > kPerResource)
      return false;
    return total_.load(std::memory_order_relaxed) + length <= kTotal;
  }

  void
  acquire(uint64_t length) {
    total_.fetch_add(length, std::memory_order_relaxed);
  }

  void
  release(uint64_t length) {
    total_.fetch_sub(length, std::memory_order_relaxed);
  }

  static RenameBudget &instance();

private:
  std::atomic<uint64_t> total_ = {0};
};

} // namespace dxmt
