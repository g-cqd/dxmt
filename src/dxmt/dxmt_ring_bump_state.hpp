#pragma once

// Sequence-tracked staging allocator over buffers from an Allocator (Metal buffers, host memory).
//
// Every allocation belongs to a chunk (seq_id). `coherent_id` passed to allocate() is the last chunk the GPU has
// finished; memory used by a chunk before it may be handed out again. free_blocks(coherent_id) is called as chunks
// complete (every chunk up to coherent_id is done); free_blocks(~0) releases everything.
//
// - Requests up to BlockSize go to a ring: one buffer, carved from head to tail, wrapping around. A ring that is
//   full is replaced by one twice as big (the old one is released when its last chunk is done) instead of waiting
//   for the GPU, which the callers can't do safely here. A ring that stays under a quarter full for
//   kStagingBlockLifetime chunks is replaced by a smaller one, and an empty idle ring is released.
// - Larger requests (texture uploads of 64 MB and more) go to an arena: one ring of kLargeArenaBlocks * BlockSize
//   created on the first large request and released when idle. When the arena is full or the request is bigger
//   than it, a whole block comes from a pool of power-of-two size classes, reused once the GPU is done with it,
//   capped at kLargePoolCapBlocks * BlockSize and trimmed when idle.
//
// Only the C++ standard library and util_math.hpp are needed here, so the logic is unit-tested natively
// (tests/unit/run.sh). Define DXMT_RING_BUMP_LOG_NEW_BLOCK(size) before including to log buffer creations.

#include "util_math.hpp"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <utility>

#ifndef DXMT_RING_BUMP_LOG_NEW_BLOCK
#define DXMT_RING_BUMP_LOG_NEW_BLOCK(size) ((void)(size))
#endif

namespace dxmt {

constexpr size_t kStagingBlockLifetime = 300;
// Finished large blocks kept for reuse, in units of BlockSize (32 MB blocks: 512 MB).
constexpr size_t kLargePoolCapBlocks = 16;
// Arena for large requests, in units of BlockSize (32 MB blocks: 256 MB); created on the first large request.
constexpr size_t kLargeArenaBlocks = 8;

template <typename Allocator, size_t BlockSize, class mutex> class RingBumpStateImpl {

public:
  static constexpr size_t block_size = BlockSize;

  RingBumpStateImpl(Allocator &&allocator) : allocator_(std::move(allocator)) {}

  std::pair<typename Allocator::Block &, uint64_t>
  allocate(uint64_t seq_id, uint64_t coherent_id, size_t size, size_t alignment);

  void free_blocks(uint64_t coherent_id);

private:
  // One buffer carved as a ring. `regions` are the live ranges in allocation order; a chunk may own several
  // (one per wrap). The live part runs from regions.front().begin to regions.back().end, wrapping at capacity.
  struct Ring {
    struct Region {
      uint64_t seq;
      size_t begin, end;
    };
    size_t capacity;
    typename Allocator::Block block;
    std::deque<Region> regions;
    size_t peak_used = 0; // since the last shrink check
    uint64_t idle_time = 0;

    bool
    empty() const {
      return regions.empty();
    }
    // Allocations don't always arrive in chunk order (CommandQueue::Retain allocates for the chunk it is given),
    // so the ring is in use until the newest chunk that ever allocated in it is done.
    uint64_t newest_seq = 0;
    uint64_t
    last_seq() const {
      return regions.empty() ? 0 : newest_seq;
    }
    size_t
    used() const {
      if (regions.empty())
        return 0;
      size_t tail = regions.front().begin, head = regions.back().end;
      return head > tail ? head - tail : capacity - tail + head;
    }
    // Drops the regions of chunks before done_before.
    void
    retire(uint64_t done_before) {
      while (!regions.empty() && regions.front().seq < done_before)
        regions.pop_front();
    }
    // Places `size` bytes for chunk `seq`; false when they don't fit.
    bool
    place(uint64_t seq, size_t size, size_t alignment, size_t &offset) {
      size_t need = size ? size : 1; // a zero-size allocation still owns a byte, so regions never collapse
      size_t at;
      if (regions.empty()) {
        if (need > capacity)
          return false;
        at = 0;
      } else {
        size_t tail = regions.front().begin, head = regions.back().end;
        if (head > tail) { // live [tail, head): free [head, capacity) then [0, tail)
          at = align(head, alignment);
          if (at + need > capacity) {
            if (need > tail)
              return false;
            at = 0;
          }
        } else { // live wraps (or the ring is full): free [head, tail)
          at = align(head, alignment);
          if (at + need > tail)
            return false;
        }
      }
      if (seq > newest_seq)
        newest_seq = seq;
      if (!regions.empty() && regions.back().seq == seq && at >= regions.back().end)
        regions.back().end = at + need;
      else
        regions.push_back({seq, at, at + need});
      size_t u = used();
      if (u > peak_used)
        peak_used = u;
      idle_time = 0;
      offset = at;
      return true;
    }
  };

  struct LargeBlock {
    size_t total_size;
    uint64_t last_used_seq_id;
    uint64_t idle_time;
    Allocator::Block block;
  };

  std::unique_ptr<Ring>
  new_ring(size_t capacity) {
    DXMT_RING_BUMP_LOG_NEW_BLOCK(capacity);
    return std::unique_ptr<Ring>(new Ring{.capacity = capacity, .block = allocator_.allocate(capacity)});
  }

  // Moves `ring` to the retired list (released when its last chunk is done), or drops it if nothing uses it.
  void
  retire_ring(std::unique_ptr<Ring> ring) {
    if (ring && !ring->empty())
      retired_.push_back(std::move(ring));
  }

  void
  release_retired(uint64_t done_before) {
    for (auto it = retired_.begin(); it != retired_.end();) {
      if ((*it)->last_seq() < done_before)
        it = retired_.erase(it);
      else
        ++it;
    }
  }

  std::pair<typename Allocator::Block &, uint64_t> allocate_regular(uint64_t seq_id, size_t size, size_t alignment);

  std::pair<typename Allocator::Block &, uint64_t>
  allocate_large(uint64_t seq_id, uint64_t coherent_id, size_t size, size_t alignment);

  // Moves in-flight pool blocks whose last chunk is before done_before back to the pool.
  void reclaim_large(uint64_t done_before);

  static constexpr size_t
  pow2_at_least(size_t size, size_t from) {
    size_t cls = from;
    while (cls < size)
      cls <<= 1;
    return cls;
  }

  static constexpr size_t large_pool_cap = kLargePoolCapBlocks * BlockSize;
  static constexpr size_t arena_capacity = kLargeArenaBlocks * BlockSize;

  std::unique_ptr<Ring> ring_;
  size_t next_ring_capacity_ = BlockSize;
  uint64_t shrink_window_ = 0; // free_blocks calls since the last shrink check
  std::list<std::unique_ptr<Ring>> retired_;

  std::unique_ptr<Ring> arena_;

  std::list<LargeBlock> large_in_flight_;
  std::list<LargeBlock> large_pool_; // oldest first
  size_t large_pool_bytes_ = 0;

  mutex mutex_;
  Allocator allocator_;
};

template <typename Allocator, size_t BlockSize, class mutex>
std::pair<typename Allocator::Block &, uint64_t>
RingBumpStateImpl<Allocator, BlockSize, mutex>::allocate(
    uint64_t seq_id, uint64_t coherent_id, size_t size, size_t alignment
) {
  std::lock_guard<mutex> lock(mutex_);
  // chunks before coherent_id are done (callers that never call free_blocks rely on this)
  if (ring_)
    ring_->retire(coherent_id);
  release_retired(coherent_id);
  if (size > BlockSize)
    return allocate_large(seq_id, coherent_id, size, alignment);
  return allocate_regular(seq_id, size, alignment);
}

template <typename Allocator, size_t BlockSize, class mutex>
std::pair<typename Allocator::Block &, uint64_t>
RingBumpStateImpl<Allocator, BlockSize, mutex>::allocate_regular(uint64_t seq_id, size_t size, size_t alignment) {
  size_t offset;
  if (ring_ && ring_->place(seq_id, size, alignment, offset))
    return {ring_->block, offset};
  // full (or none yet): a bigger ring takes over; the current one is released once its chunks are done
  size_t capacity = next_ring_capacity_;
  if (ring_)
    capacity = std::max(capacity, ring_->capacity * 2);
  capacity = pow2_at_least(size, capacity); // a fresh ring places at offset 0, aligned for any alignment
  retire_ring(std::move(ring_));
  ring_ = new_ring(capacity);
  next_ring_capacity_ = BlockSize;
  ring_->place(seq_id, size, alignment, offset);
  return {ring_->block, offset};
}

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
RingBumpStateImpl<Allocator, BlockSize, mutex>::allocate_large(
    uint64_t seq_id, uint64_t coherent_id, size_t size, size_t alignment
) {
  // the arena first
  if (size + alignment <= arena_capacity) {
    if (!arena_)
      arena_ = new_ring(arena_capacity);
    arena_->retire(coherent_id);
    size_t offset;
    if (arena_->place(seq_id, size, alignment, offset))
      return {arena_->block, offset};
  }
  // arena full or too small: a whole block from the pool (same reuse rule as the ring)
  reclaim_large(coherent_id);
  const size_t cls = pow2_at_least(size, BlockSize);
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

template <typename Allocator, size_t BlockSize, class mutex>
void
RingBumpStateImpl<Allocator, BlockSize, mutex>::free_blocks(uint64_t coherent_id) {
  std::lock_guard<mutex> lock(mutex_);
  if (coherent_id == ~0ull) {
    // Deferred contexts tear down after every command list: the next list starts with a ring that holds what
    // this one needed, instead of growing again from BlockSize.
    size_t needed = ring_ ? ring_->used() : 0;
    for (auto &retired : retired_)
      needed += retired->used();
    next_ring_capacity_ = pow2_at_least(needed, BlockSize);
    ring_.reset();
    retired_.clear();
    arena_.reset();
    large_in_flight_.clear();
    large_pool_.clear();
    large_pool_bytes_ = 0;
    return;
  }
  const uint64_t done_before = coherent_id + 1; // every chunk up to coherent_id is done
  release_retired(done_before);

  // ring: release when idle, hand over to a smaller one when oversized
  if (ring_) {
    ring_->retire(done_before);
    if (++ring_->idle_time > kStagingBlockLifetime && ring_->empty()) {
      ring_.reset();
      next_ring_capacity_ = BlockSize;
    }
  }
  if (ring_) {
    if (++shrink_window_ >= kStagingBlockLifetime) {
      shrink_window_ = 0;
      if (ring_->capacity > BlockSize && ring_->peak_used * 4 <= ring_->capacity) {
        next_ring_capacity_ = pow2_at_least(ring_->peak_used * 2, BlockSize);
        retire_ring(std::move(ring_)); // the next allocation creates the smaller ring
      } else {
        ring_->peak_used = ring_->used();
      }
    }
  }

  // arena: release when idle
  if (arena_) {
    arena_->retire(done_before);
    if (++arena_->idle_time > kStagingBlockLifetime && arena_->empty())
      arena_.reset();
  }

  // pool
  for (auto &pooled : large_pool_)
    pooled.idle_time++;
  reclaim_large(done_before);
  while (!large_pool_.empty() &&
         (large_pool_.front().idle_time > kStagingBlockLifetime || large_pool_bytes_ > large_pool_cap)) {
    large_pool_bytes_ -= large_pool_.front().total_size;
    large_pool_.pop_front();
  }
}

} // namespace dxmt
