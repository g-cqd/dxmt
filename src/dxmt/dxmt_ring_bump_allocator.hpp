#pragma once

#include "Metal.hpp"
#include "log/log.hpp"
#include "thread.hpp"
#include "util_math.hpp"
#include <mutex>
#include <queue>

#define DXMT_RING_BUMP_LOG_NEW_BLOCK(size) WARN("forced to allocate new block of size ", size)
#include "dxmt_ring_bump_state.hpp"

namespace dxmt {

constexpr size_t kStagingBlockSize = 0x2000000; // 32MB
constexpr size_t kStagingBlockSizeForDeferredContext = 0x200000; // 2MB

template <typename Allocator, size_t BlockSize = kStagingBlockSize, class mutex = dxmt::mutex>
using RingBumpState = RingBumpStateImpl<Allocator, BlockSize, mutex>;

class GpuPrivateBufferBlockAllocator {
public:
  GpuPrivateBufferBlockAllocator(WMT::Device device, WMTResourceOptions block_options) {
    device_ = device;
    buffer_info_.memory.set(nullptr);
    buffer_info_.options = block_options;
  }

  class Block {
  public:
    WMT::Reference<WMT::Buffer> buffer;
    uint64_t gpu_address;

    Block() = default;
    Block(const Block &copy) = delete;
    Block(Block &&move) = default;
  };

  Block
  allocate(size_t block_size) {
    Block block{};
    buffer_info_.length = block_size;
    block.buffer = device_.newBuffer(buffer_info_);
    block.gpu_address = buffer_info_.gpu_address;
    return block;
  };

private:
  WMT::Device device_;
  WMTBufferInfo buffer_info_;
};

class StagingBufferBlockAllocator {
public:
  StagingBufferBlockAllocator(WMT::Device device, WMTResourceOptions block_options, bool placed_buffer = true) {
    device_ = device;
    buffer_info_ = block_options;
    placed_buffer_ = placed_buffer;
  }

  class Block {
  public:
    WMT::Reference<WMT::Buffer> buffer;
    uint64_t gpu_address;
    void *mapped_address;

    ~Block() {
      if (mapped_address) {
        free(mapped_address);
        mapped_address = nullptr;
      }
    };

    Block() = default;

    Block(const Block &) = delete;
    Block(Block &&move) {
      buffer = std::move(move.buffer);
      gpu_address = move.gpu_address;
      mapped_address = move.mapped_address;
      move.mapped_address = nullptr;
    };
  };

  Block
  allocate(size_t block_size) {
    Block block{};
    block.mapped_address = placed_buffer_ ? malloc(block_size) : nullptr;
    WMTBufferInfo info;
    info.options = buffer_info_;
    info.memory.set(block.mapped_address);
    info.length = block_size;
    block.buffer = device_.newBuffer(info);
    block.gpu_address = info.gpu_address;
    return block;
  };

private:
  WMT::Device device_;
  WMTResourceOptions buffer_info_;
  bool placed_buffer_;
};

class HostBufferBlockAllocator {
public:
  class Block {
  public:
    void *ptr;

    Block() = default;

    Block(const Block &) = delete;
    Block(Block &&move) {
      ptr = move.ptr;
      move.ptr = nullptr;
    };

    ~Block() {
      if (ptr) {
        free(ptr);
        ptr = nullptr;
      }
    };
  };

  Block
  allocate(size_t block_size) {
    Block block{};
    block.ptr = malloc(block_size);
    return block;
  };
};

} // namespace dxmt