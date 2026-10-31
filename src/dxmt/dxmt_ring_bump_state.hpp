#pragma once

// Sequence-tracked bump allocator over blocks from an Allocator (Metal buffers, host memory). Only the C++ standard
// library and util_math.hpp are needed here, so the logic is unit-tested natively (tests/unit).
// Define DXMT_RING_BUMP_LOG_NEW_BLOCK(size) before including to log blocks that could not be reused.

#include "util_math.hpp"
#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <queue>
#include <utility>

#ifndef DXMT_RING_BUMP_LOG_NEW_BLOCK
#define DXMT_RING_BUMP_LOG_NEW_BLOCK(size) ((void)(size))
#endif

namespace dxmt {

constexpr size_t kStagingBlockLifetime = 300;
// Finished large blocks (bigger than BlockSize) kept for reuse, in units of BlockSize (32 MB blocks: 512 MB).
constexpr size_t kLargePoolCapBlocks = 16;

template <typename Allocator, size_t BlockSize, class mutex> class RingBumpStateImpl {

public:

  static constexpr size_t block_size = BlockSize;

  RingBumpStateImpl(Allocator &&allocator) : allocator_(std::move(allocator)) {}

  std::pair<typename Allocator::Block &, uint64_t>
  allocate(uint64_t seq_id, uint64_t coherent_id, size_t size, size_t alignment);

  void free_blocks(uint64_t coherent_id);

private:
  struct Allocation {
    size_t allocated_size;
    size_t total_size;
    uint64_t last_used_seq_id;
    uint64_t inc_time_to_live;
    Allocator::Block block;
  };

  Allocation &allocate_or_reuse_block(uint64_t seq_id, uint64_t coherent_id, size_t block_size);

  // Requests bigger than BlockSize (texture uploads of 64 MB and more) get a whole block from a pool of
  // power-of-two size classes. A block returns to the pool once the GPU has finished the last chunk that used it
  // and is reused by the next request of its class; the pool is capped and idle blocks expire. These blocks
  // never enter the FIFO, so they can't be held back behind a regular block that is still waiting to expire.
  struct LargeBlock {
    size_t total_size;
    uint64_t last_used_seq_id;
    uint64_t idle_time;
    Allocator::Block block;
  };

  std::pair<typename Allocator::Block &, uint64_t>
  allocate_large(uint64_t seq_id, uint64_t coherent_id, size_t size);

  // Moves in-flight large blocks whose last chunk is done (last use < done_before) to the pool.
  void reclaim_large(uint64_t done_before);

  static constexpr size_t
  large_size_class(size_t size) {
    size_t cls = BlockSize;
    while (cls < size)
      cls <<= 1;
    return cls;
  }

  static constexpr size_t large_pool_cap = kLargePoolCapBlocks * BlockSize;

  std::list<LargeBlock> large_in_flight_;
  std::list<LargeBlock> large_pool_; // oldest first
  size_t large_pool_bytes_ = 0;

  std::pair<typename Allocator::Block &, uint64_t>
  suballocate(Allocation &allocation, size_t size, size_t alignment) {
    auto offset = align(allocation.allocated_size, alignment);
    allocation.allocated_size = offset + size;
    return {allocation.block, offset};
  };

  std::queue<Allocation> fifo;
  mutex mutex_;
  Allocator allocator_;
};

template <typename Allocator, size_t BlockSize, class mutex>
std::pair<typename Allocator::Block &, uint64_t>
RingBumpStateImpl<Allocator, BlockSize, mutex>::allocate(
    uint64_t seq_id, uint64_t coherent_id, size_t size, size_t alignment
) {
  std::lock_guard<mutex> lock(mutex_);
  if (size > BlockSize)
    return allocate_large(seq_id, coherent_id, size);
  while (!fifo.empty()) {
    auto &latest = fifo.back();
    if ((align(latest.allocated_size, alignment) + size) > latest.total_size) {
      break;
    }
    latest.last_used_seq_id = seq_id;
    return suballocate(latest, size, alignment);
  }
  return suballocate(
      allocate_or_reuse_block(
          seq_id, coherent_id, std::max(size, BlockSize) // in case required size is larger than block size
      ),
      size, alignment
  );
};

template <typename Allocator, size_t BlockSize, class mutex>
void
RingBumpStateImpl<Allocator, BlockSize, mutex>::free_blocks(uint64_t coherent_id) {
  std::lock_guard<mutex> lock(mutex_);
  if (coherent_id == ~0ull) {
    large_in_flight_.clear();
    large_pool_.clear();
    large_pool_bytes_ = 0;
  } else {
    for (auto &pooled : large_pool_)
      pooled.idle_time++;
    reclaim_large(coherent_id + 1); // every chunk up to coherent_id is done
    while (!large_pool_.empty() &&
           (large_pool_.front().idle_time > kStagingBlockLifetime || large_pool_bytes_ > large_pool_cap)) {
      large_pool_bytes_ -= large_pool_.front().total_size;
      large_pool_.pop_front();
    }
  }
  while (!fifo.empty()) {
    auto &front = fifo.front();
    if (front.last_used_seq_id > coherent_id)
      break;
    auto expired = (coherent_id - front.last_used_seq_id) > kStagingBlockLifetime ||
                   front.inc_time_to_live > kStagingBlockLifetime || coherent_id == -1ull;
    auto adhoc = front.total_size != BlockSize;
    if (expired || adhoc) {
      // can be deallocated
      fifo.pop();
      continue;
    }
    front.inc_time_to_live++;
    break;
  }
};

template <typename Allocator, size_t BlockSize, class mutex>
RingBumpStateImpl<Allocator, BlockSize, mutex>::Allocation &
RingBumpStateImpl<Allocator, BlockSize, mutex>::allocate_or_reuse_block(
    uint64_t seq_id, uint64_t coherent_id, size_t block_size
) {
  while (!fifo.empty()) {
    auto &front = fifo.front();
    if (front.last_used_seq_id < coherent_id) {
      if (front.total_size != BlockSize) {
        fifo.pop();
        continue;
      } else if (front.total_size >= block_size) {
        front.last_used_seq_id = seq_id;
        front.allocated_size = 0;
        front.inc_time_to_live = 0;
        fifo.push(std::move(front));
        fifo.pop();
        return fifo.back();
      }
      DXMT_RING_BUMP_LOG_NEW_BLOCK(block_size);
    }
    break;
  }
  fifo.push({
      .allocated_size = 0,
      .total_size = block_size,
      .last_used_seq_id = seq_id,
      .inc_time_to_live = 0,
      .block = allocator_.allocate(block_size),
  });
  return fifo.back();
};

template <typename Allocator, size_t BlockSize, class mutex>
void
RingBumpStateImpl<Allocator, BlockSize, mutex>::reclaim_large(uint64_t done_before) {
  for (auto it = large_in_flight_.begin(); it != large_in_flight_.end();) {
    auto next = std::next(it);
    if (it->last_used_seq_id < done_before) {
      it->idle_time = 0;
      large_pool_bytes_ += it->total_size;
      large_pool_.splice(large_pool_.end(), large_in_flight_, it);
    }
    it = next;
  }
}

template <typename Allocator, size_t BlockSize, class mutex>
std::pair<typename Allocator::Block &, uint64_t>
RingBumpStateImpl<Allocator, BlockSize, mutex>::allocate_large(uint64_t seq_id, uint64_t coherent_id, size_t size) {
  // same rule as the regular blocks: reuse only what a chunk before coherent_id used
  reclaim_large(coherent_id);
  const size_t cls = large_size_class(size);
  for (auto it = large_pool_.rbegin(); it != large_pool_.rend(); ++it) { // most recently used first
    if (it->total_size != cls)
      continue;
    auto pos = std::prev(it.base());
    pos->last_used_seq_id = seq_id;
    pos->idle_time = 0;
    large_pool_bytes_ -= pos->total_size;
    large_in_flight_.splice(large_in_flight_.end(), large_pool_, pos);
    return {large_in_flight_.back().block, 0};
  }
  DXMT_RING_BUMP_LOG_NEW_BLOCK(cls);
  large_in_flight_.push_back({
      .total_size = cls,
      .last_used_seq_id = seq_id,
      .idle_time = 0,
      .block = allocator_.allocate(cls),
  });
  return {large_in_flight_.back().block, 0};
}

} // namespace dxmt
