// Regression test: repeated write-maps of a large staging resource that the GPU
// still reads must not exhaust a 32-bit address space.
//
// Crysis 3 (32-bit) loads a level by writing a 4 MB staging buffer and copying it
// to a default buffer hundreds of times without presenting or flushing. Each
// write-map while the GPU still owes a copy made DXMT rename the staging resource
// to a fresh 4 MB CPU allocation that was only recycled after the GPU finished;
// nothing flushed, so the pool grew until _aligned_malloc failed and Map wrote
// 4 MB to address 0.
//
// This test repeats that pattern (default 20 bursts of 300 map/write/unmap/copy
// cycles of a 4 MB staging buffer, Flush only between bursts), counts committed
// private 4 MB regions in its own address space, then reads the destination back
// to check that every partial write landed and the rest of the data survived.
//
// usage: dx11_staging_rename [bursts] [maps_per_burst] [max_4mb_regions]
// Exit 0: every map succeeded, the data is correct, and the number of 4 MB
// regions stayed at or below max_4mb_regions (default 64, i.e. 256 MB).
// Exit 1: a check failed. Exit 3: the process faulted (stock DXMT on i386).
#include <d3d11.h>
#include <windows.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static const UINT kSize = 4u << 20;
static const UINT kStride = 4096; // one dword written per map, kStride apart

static LONG WINAPI on_fault(EXCEPTION_POINTERS *info) {
  std::printf("FAULT code=0x%08lx at %p (access %s 0x%p)\n", (unsigned long)info->ExceptionRecord->ExceptionCode,
              info->ExceptionRecord->ExceptionAddress,
              info->ExceptionRecord->NumberParameters > 1 && info->ExceptionRecord->ExceptionInformation[0] ? "write"
                                                                                                             : "read",
              info->ExceptionRecord->NumberParameters > 1 ? (void *)info->ExceptionRecord->ExceptionInformation[1]
                                                          : nullptr);
  std::fflush(stdout);
  ExitProcess(3);
  return EXCEPTION_EXECUTE_HANDLER;
}

struct AddressSpace {
  unsigned regions4mb;
  unsigned long long committed_mb;
  unsigned long long largest_free_mb;
};

static AddressSpace measure() {
  AddressSpace r = {};
  MEMORY_BASIC_INFORMATION m;
  ULONG_PTR limit = sizeof(void *) == 4 ? 0xffffffffu : (ULONG_PTR)0x7fffffffffffull;
  ULONG_PTR a = 0x10000;
  unsigned long long committed = 0, largest = 0;
  while (a < limit && VirtualQuery((void *)a, &m, sizeof m)) {
    unsigned long long size = m.RegionSize;
    if (m.State == MEM_FREE) {
      if (size > largest)
        largest = size;
    } else if (m.State == MEM_COMMIT) {
      committed += size;
      if (m.Type == MEM_PRIVATE && size >= kSize && size <= kSize + 0x10000)
        r.regions4mb++;
    }
    ULONG_PTR next = (ULONG_PTR)m.BaseAddress + m.RegionSize;
    if (next <= a)
      break;
    a = next;
  }
  r.committed_mb = committed >> 20;
  r.largest_free_mb = largest >> 20;
  return r;
}

int main(int argc, char **argv) {
  SetUnhandledExceptionFilter(on_fault);
  int bursts = argc > 1 ? std::atoi(argv[1]) : 20;
  int maps = argc > 2 ? std::atoi(argv[2]) : 300;
  unsigned max_regions = argc > 3 ? (unsigned)std::atoi(argv[3]) : 64;
  int failures = 0;

  ID3D11Device *device = nullptr;
  ID3D11DeviceContext *context = nullptr;
  D3D_FEATURE_LEVEL level = {};
  HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                 &device, &level, &context);
  std::printf("D3D11CreateDevice hr=0x%08lx (%d-bit) bursts=%d maps/burst=%d max 4MB regions=%u\n",
              (unsigned long)hr, (int)sizeof(void *) * 8, bursts, maps, max_regions);
  if (FAILED(hr))
    return 2;

  D3D11_BUFFER_DESC sd = {};
  sd.ByteWidth = kSize;
  sd.Usage = D3D11_USAGE_STAGING;
  sd.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
  ID3D11Buffer *upload = nullptr;
  hr = device->CreateBuffer(&sd, nullptr, &upload);
  if (FAILED(hr)) {
    std::printf("FAIL CreateBuffer(staging write) hr=0x%08lx\n", (unsigned long)hr);
    return 1;
  }
  sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  ID3D11Buffer *readback = nullptr;
  hr = device->CreateBuffer(&sd, nullptr, &readback);
  if (FAILED(hr)) {
    std::printf("FAIL CreateBuffer(staging read) hr=0x%08lx\n", (unsigned long)hr);
    return 1;
  }
  D3D11_BUFFER_DESC dd = {};
  dd.ByteWidth = kSize;
  dd.Usage = D3D11_USAGE_DEFAULT;
  dd.BindFlags = D3D11_BIND_SHADER_RESOURCE;
  ID3D11Buffer *target = nullptr;
  hr = device->CreateBuffer(&dd, nullptr, &target);
  if (FAILED(hr)) {
    std::printf("FAIL CreateBuffer(default) hr=0x%08lx\n", (unsigned long)hr);
    return 1;
  }

  D3D11_MAPPED_SUBRESOURCE mapped = {};
  hr = context->Map(upload, 0, D3D11_MAP_WRITE, 0, &mapped);
  if (FAILED(hr) || !mapped.pData) {
    std::printf("FAIL initial Map hr=0x%08lx pData=%p\n", (unsigned long)hr, mapped.pData);
    return 1;
  }
  std::memset(mapped.pData, 0xab, kSize);
  context->Unmap(upload, 0);

  AddressSpace start = measure();
  std::printf("start: 4MB regions %u, committed %llu MB, largest free %llu MB\n", start.regions4mb,
              start.committed_mb, start.largest_free_mb);

  const UINT slots = kSize / kStride;
  unsigned peak = start.regions4mb;
  double worst_burst_ms = 0, total_ms = 0;
  unsigned long long last_value = 0;
  for (int b = 0; b < bursts && !failures; b++) {
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < maps; i++) {
      hr = context->Map(upload, 0, D3D11_MAP_WRITE, 0, &mapped);
      if (FAILED(hr) || !mapped.pData) {
        std::printf("FAIL Map burst %d map %d hr=0x%08lx pData=%p\n", b, i, (unsigned long)hr, mapped.pData);
        failures++;
        break;
      }
      unsigned value = (unsigned)(b * maps + i) + 1;
      // partial write: everything else must be preserved by a rename
      ((volatile unsigned *)mapped.pData)[(value % slots) * (kStride / 4)] = value;
      last_value = value;
      context->Unmap(upload, 0);
      context->CopyResource(target, upload);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    total_ms += ms;
    if (ms > worst_burst_ms)
      worst_burst_ms = ms;
    AddressSpace now = measure();
    if (now.regions4mb > peak)
      peak = now.regions4mb;
    std::printf("burst %2d: %8.2f ms (%.3f ms/map), 4MB regions %u, committed %llu MB, largest free %llu MB\n", b,
                ms, ms / maps, now.regions4mb, now.committed_mb, now.largest_free_mb);
    std::fflush(stdout);
    context->Flush();
  }

  if (!failures) {
    context->CopyResource(readback, target);
    hr = context->Map(readback, 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr) || !mapped.pData) {
      std::printf("FAIL readback Map hr=0x%08lx\n", (unsigned long)hr);
      failures++;
    } else {
      // expected: slot s holds the last value v with v % slots == s, or 0xabababab if never written
      const unsigned *data = (const unsigned *)mapped.pData;
      unsigned bad = 0;
      for (UINT s = 0; s < slots; s++) {
        unsigned long long v = last_value;
        while (v > 0 && v % slots != s)
          v--;
        unsigned want = v ? (unsigned)v : 0xababababu;
        if (data[s * (kStride / 4)] != want || data[s * (kStride / 4) + 1] != 0xababababu)
          bad++;
      }
      context->Unmap(readback, 0);
      std::printf("%s readback: %u of %u slots wrong\n", bad ? "FAIL" : "PASS", bad, slots);
      if (bad)
        failures++;
    }
  }

  bool bounded = peak <= max_regions;
  std::printf("%s peak 4MB regions %u (start %u, limit %u)\n", bounded ? "PASS" : "FAIL", peak, start.regions4mb,
              max_regions);
  if (!bounded)
    failures++;
  std::printf("time: total %.1f ms, worst burst %.1f ms, mean %.3f ms/map\n", total_ms, worst_burst_ms,
              bursts && maps ? total_ms / ((double)bursts * maps) : 0.0);
  std::printf("%s\n", failures ? "RESULT FAIL" : "RESULT PASS");
  readback->Release();
  target->Release();
  upload->Release();
  context->Release();
  device->Release();
  return failures ? 1 : 0;
}
