#include "dxmt_dynamic.hpp"
#include "dxmt_texture.hpp"

namespace dxmt {
DynamicBuffer::DynamicBuffer(Buffer *buffer, Flags<BufferAllocationFlag> flags) :
    buffer(buffer),
    flags_(flags),
    name_(buffer->current()) {}

DynamicBuffer::~DynamicBuffer() {
  RenameBudget::instance().release(renamed_bytes_);
}

void
DynamicBuffer::incRef() {
  refcount_.fetch_add(1u, std::memory_order_acquire);
};

void
DynamicBuffer::decRef() {
  if (refcount_.fetch_sub(1u, std::memory_order_release) == 1u)
    delete this;
};

Rc<BufferAllocation>
DynamicBuffer::allocate(uint64_t coherent_seq_id) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  Rc<BufferAllocation> ret;
  for (;;) {
    if (fifo.empty()) {
      break;
    }
    auto entry = fifo.front();
    if (entry.will_free_at > coherent_seq_id) {
      break;
    }
    ret = std::move(entry.allocation);
    fifo.pop();
    break;
  }
  if (!ret.ptr()) {
    ret = buffer->allocate(flags_);
    // one without CPU memory is dropped by the caller and does not count
    if (ret->mappedMemory(0)) {
      renamed_bytes_ += ret->allocatedLength();
      RenameBudget::instance().acquire(ret->allocatedLength());
    }
  }
  return ret;
}

bool
DynamicBuffer::mustWaitToAllocate(uint64_t coherent_seq_id, uint64_t *wait_seq_id) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (fifo.empty() || fifo.front().will_free_at <= coherent_seq_id)
    return false;
  if (RenameBudget::instance().allows(renamed_bytes_, name_->allocatedLength()))
    return false;
  *wait_seq_id = fifo.front().will_free_at;
  return true;
}

uint64_t
DynamicBuffer::oldestRetiredSeqId() {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  return fifo.empty() ? 0 : fifo.front().will_free_at;
}

void
DynamicBuffer::updateImmediateName(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation, uint32_t suballocation, bool owned_by_command_list) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (!owned_by_command_list_)
    fifo.push(QueueEntry{.allocation = std::move(name_), .will_free_at = current_seq_id});
  name_ = std::move(allocation);
  name_suballocation_ = suballocation;
  owned_by_command_list_ = owned_by_command_list;
}

void
DynamicBuffer::recycle(uint64_t current_seq_id, Rc<BufferAllocation> &&allocation) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (owned_by_command_list_) {
    if (name_.ptr() == allocation.ptr()) {
      owned_by_command_list_ = false;
      auto _ = std::move(allocation);
      return;
    }
  }
  fifo.push(QueueEntry{.allocation = std::move(allocation), .will_free_at = current_seq_id});
}

uint32_t
DynamicBuffer::nextSuballocation() {
  if (name_->hasSuballocatoin(name_suballocation_ + 1)) {
    return ++name_suballocation_;
  }
  return 0;
}

DynamicLinearTexture::DynamicLinearTexture(Texture *texture, Flags<TextureAllocationFlag> flags) :
    texture(texture),
    flags_(flags),
    name_(texture->current()) {}

DynamicLinearTexture::~DynamicLinearTexture() {
  RenameBudget::instance().release(renamed_bytes_);
}

void
DynamicLinearTexture::incRef() {
  refcount_.fetch_add(1u, std::memory_order_acquire);
};

void
DynamicLinearTexture::decRef() {
  if (refcount_.fetch_sub(1u, std::memory_order_release) == 1u)
    delete this;
};

Rc<TextureAllocation>
DynamicLinearTexture::allocate(uint64_t coherent_seq_id) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  Rc<TextureAllocation> ret;
  for (;;) {
    if (fifo.empty()) {
      break;
    }
    auto entry = fifo.front();
    if (entry.will_free_at > coherent_seq_id) {
      break;
    }
    ret = std::move(entry.allocation);
    fifo.pop();
    break;
  }
  if (!ret.ptr()) {
    ret = texture->allocate(flags_);
    // one without CPU memory is dropped by the caller and does not count
    if (ret->mappedMemory) {
      renamed_bytes_ += texture->bytesPerImage();
      RenameBudget::instance().acquire(texture->bytesPerImage());
    }
  }
  return ret;
}

bool
DynamicLinearTexture::mustWaitToAllocate(uint64_t coherent_seq_id, uint64_t *wait_seq_id) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (fifo.empty() || fifo.front().will_free_at <= coherent_seq_id)
    return false;
  if (RenameBudget::instance().allows(renamed_bytes_, texture->bytesPerImage()))
    return false;
  *wait_seq_id = fifo.front().will_free_at;
  return true;
}

uint64_t
DynamicLinearTexture::oldestRetiredSeqId() {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  return fifo.empty() ? 0 : fifo.front().will_free_at;
}

void
DynamicLinearTexture::updateImmediateName(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation, bool owned_by_command_list) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (!owned_by_command_list_)
    fifo.push(QueueEntry{.allocation = std::move(name_), .will_free_at = current_seq_id});
  name_ = std::move(allocation);
  owned_by_command_list_ = owned_by_command_list;
}

void
DynamicLinearTexture::recycle(uint64_t current_seq_id, Rc<TextureAllocation> &&allocation) {
  std::lock_guard<dxmt::mutex> lock(mutex_);
  if (owned_by_command_list_) {
    if (name_.ptr() == allocation.ptr()) {
      owned_by_command_list_ = false;
      auto _ = std::move(allocation);
      return;
    }
  }
  fifo.push(QueueEntry{.allocation = std::move(allocation), .will_free_at = current_seq_id});
}

} // namespace dxmt
