#pragma once
#include "dxmt_buffer.hpp"
#include "dxmt_texture.hpp"
#include <queue>

namespace dxmt {

class DynamicBuffer {
public:
  void incRef();
  void decRef();

  Rc<BufferAllocation> allocate(uint64_t coherent_seq_id);
  /**
   * Whether a WRITE_DISCARD map on the immediate context must wait for the GPU before
   * allocate(): the oldest retired allocation is still in use and the rename budget
   * allows no new one. Sets *wait_seq_id to the sequence id that frees it.
   */
  bool mustWaitToAllocate(uint64_t coherent_seq_id, uint64_t *wait_seq_id);

  /**
   * The sequence id that frees the oldest retired allocation, or 0 if none is retired.
   */
  uint64_t oldestRetiredSeqId();
  void updateImmediateName(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation, uint32_t suballocation, bool owned_by_command_list);
  void recycle(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation);
  uint32_t nextSuballocation();

  Rc<BufferAllocation>
  immediateName() {
    return name_;
  }

  uint32_t
  immediateSuballocation() {
    return name_suballocation_;
  }

  void *
  immediateMappedMemory() {
    return name_->mappedMemory(name_suballocation_);
  }

  DynamicBuffer(Buffer *buffer, Flags<BufferAllocationFlag> flags);
  ~DynamicBuffer();

  struct QueueEntry {
    Rc<BufferAllocation> allocation;
    uint64_t will_free_at;
  };

  /**
   * readonly
   */
  Buffer *buffer;

private:
  Flags<BufferAllocationFlag> flags_;
  std::atomic<uint32_t> refcount_ = {0u};
  std::queue<QueueEntry> fifo;
  dxmt::mutex mutex_;
  Rc<BufferAllocation> name_;
  uint32_t name_suballocation_ = 0;
  bool owned_by_command_list_ = false;
  // bytes allocated by allocate(), held in RenameBudget
  uint64_t renamed_bytes_ = 0;
};

class DynamicLinearTexture {
public:
  void incRef();
  void decRef();

  Rc<TextureAllocation> allocate(uint64_t coherent_seq_id);
  /**
   * Whether a WRITE_DISCARD map on the immediate context must wait for the GPU before
   * allocate(): the oldest retired allocation is still in use and the rename budget
   * allows no new one. Sets *wait_seq_id to the sequence id that frees it.
   */
  bool mustWaitToAllocate(uint64_t coherent_seq_id, uint64_t *wait_seq_id);

  /**
   * The sequence id that frees the oldest retired allocation, or 0 if none is retired.
   */
  uint64_t oldestRetiredSeqId();
  void updateImmediateName(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation, bool owned_by_command_list);
  void recycle(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation);

  Rc<TextureAllocation>
  immediateName() {
    return name_;
  };

  void *
  mappedMemory() {
    return name_->mappedMemory;
  }

  DynamicLinearTexture(Texture *buffer, Flags<TextureAllocationFlag> flags);
  ~DynamicLinearTexture();

  struct QueueEntry {
    Rc<TextureAllocation> allocation;
    uint64_t will_free_at;
  };
  /**
   * readonly
   */
  Texture *texture;

private:
  Flags<TextureAllocationFlag> flags_;
  std::atomic<uint32_t> refcount_ = {0u};
  std::queue<QueueEntry> fifo;
  dxmt::mutex mutex_;
  Rc<TextureAllocation> name_;
  bool owned_by_command_list_ = false;
  // bytes allocated by allocate(), held in RenameBudget
  uint64_t renamed_bytes_ = 0;
};

} // namespace dxmt
