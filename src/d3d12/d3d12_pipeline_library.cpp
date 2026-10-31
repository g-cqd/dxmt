/*
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 */

// ID3D12PipelineLibrary1: pipeline states stored and looked up by name.
//
// Some games require D3D12_SHADER_CACHE_SUPPORT_LIBRARY and abort without it (HITMAN 3). This library keeps the
// pipelines stored during the process and returns them by name; it persists nothing: Serialize writes an empty,
// versioned blob, and a blob that isn't ours is refused with D3D12_ERROR_DRIVER_VERSION_MISMATCH, which applications
// handle by creating an empty library and compiling their pipelines again (the Metal shader cache still applies).

#include "com/com_pointer.hpp"
#include "d3d12_device.hpp"
#include "d3d12_device_child.hpp"
#include "thread.hpp"
#include <cstring>
#include <map>
#include <mutex>
#include <string>

namespace dxmt {

namespace {

constexpr uint32_t kLibraryMagic = 0x4c50584d; // "MXPL"
constexpr uint32_t kLibraryVersion = 1;

struct LibraryBlobHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t pipeline_count;
};

} // namespace

class MTLD3D12PipelineLibraryImpl : public MTLD3D12DeviceChild<ID3D12PipelineLibrary1> {
public:
  MTLD3D12PipelineLibraryImpl(MTLD3D12Device *pDevice) : MTLD3D12DeviceChild<ID3D12PipelineLibrary1>(pDevice) {}

  HRESULT
  STDMETHODCALLTYPE
  QueryInterface(REFIID riid, void **ppvObject) {
    if (ppvObject == nullptr)
      return E_POINTER;

    *ppvObject = nullptr;

    if (riid == __uuidof(IUnknown) || riid == __uuidof(ID3D12Object) || riid == __uuidof(ID3D12DeviceChild) ||
        riid == __uuidof(ID3D12PipelineLibrary) || riid == __uuidof(ID3D12PipelineLibrary1)) {
      *ppvObject = ref(this);
      return S_OK;
    }

    if (logQueryInterfaceError(__uuidof(ID3D12PipelineLibrary1), riid)) {
      WARN("D3D12PipelineLibrary: Unknown interface query ", str::format(riid));
    }

    return E_NOINTERFACE;
  }

  HRESULT STDMETHODCALLTYPE
  StorePipeline(LPCWSTR pName, ID3D12PipelineState *pPipeline) {
    if (!pName || !pPipeline)
      return E_INVALIDARG;
    std::lock_guard<dxmt::mutex> lock(mutex_);
    auto [it, inserted] = pipelines_.try_emplace(pName, pPipeline);
    return inserted ? S_OK : E_INVALIDARG; // a name can be stored once
  }

  HRESULT STDMETHODCALLTYPE
  LoadGraphicsPipeline(LPCWSTR pName, const D3D12_GRAPHICS_PIPELINE_STATE_DESC *pDesc, REFIID riid, void **ppPipelineState) {
    return Load(pName, riid, ppPipelineState);
  }

  HRESULT STDMETHODCALLTYPE
  LoadComputePipeline(LPCWSTR pName, const D3D12_COMPUTE_PIPELINE_STATE_DESC *pDesc, REFIID riid, void **ppPipelineState) {
    return Load(pName, riid, ppPipelineState);
  }

  HRESULT STDMETHODCALLTYPE
  LoadPipeline(LPCWSTR pName, const D3D12_PIPELINE_STATE_STREAM_DESC *pDesc, REFIID riid, void **ppPipelineState) {
    return Load(pName, riid, ppPipelineState);
  }

  SIZE_T STDMETHODCALLTYPE
  GetSerializedSize() {
    return sizeof(LibraryBlobHeader);
  }

  HRESULT STDMETHODCALLTYPE
  Serialize(void *pData, SIZE_T DataSizeInBytes) {
    if (!pData || DataSizeInBytes < sizeof(LibraryBlobHeader))
      return E_INVALIDARG;
    LibraryBlobHeader header = {kLibraryMagic, kLibraryVersion, 0};
    std::memcpy(pData, &header, sizeof(header));
    return S_OK;
  }

private:
  HRESULT
  Load(LPCWSTR pName, REFIID riid, void **ppPipelineState) {
    if (ppPipelineState)
      *ppPipelineState = nullptr;
    if (!pName)
      return E_INVALIDARG;
    std::lock_guard<dxmt::mutex> lock(mutex_);
    auto it = pipelines_.find(pName);
    if (it == pipelines_.end())
      return E_INVALIDARG; // not stored: the application compiles it and stores it
    if (!ppPipelineState)
      return S_FALSE;
    return it->second->QueryInterface(riid, ppPipelineState);
  }

  dxmt::mutex mutex_;
  std::map<std::wstring, Com<ID3D12PipelineState>> pipelines_;
};

HRESULT
CreatePipelineLibrary(MTLD3D12Device *pDevice, const void *pBlob, SIZE_T BlobLength, REFIID riid, void **ppLibrary) {
  if (ppLibrary)
    *ppLibrary = nullptr;
  if (BlobLength) {
    LibraryBlobHeader header;
    if (!pBlob || BlobLength < sizeof(header))
      return D3D12_ERROR_DRIVER_VERSION_MISMATCH;
    std::memcpy(&header, pBlob, sizeof(header));
    if (header.magic != kLibraryMagic || header.version != kLibraryVersion)
      return D3D12_ERROR_DRIVER_VERSION_MISMATCH;
  }
  auto library = Com(new MTLD3D12PipelineLibraryImpl(pDevice));
  if (!ppLibrary)
    return S_FALSE;
  return library->QueryInterface(riid, ppLibrary);
}

} // namespace dxmt
