// Regression test: CreateRenderTargetView on a buffer resource.
//
// Frostbite (Battlefield 3 / 4) creates a D3D11_RTV_DIMENSION_BUFFER view on a
// buffer created with D3D11_BIND_RENDER_TARGET (R32G32B32A32_FLOAT, 8192
// elements, as logged by the BF3 proxy) and shows a fatal "DirectX function ...
// failed with E_INVALIDARG" dialog if it fails. Stock DXMT v0.80 returns
// E_INVALIDARG for every buffer RTV. Exit code 0 means every check passed.
#define COBJMACROS
#include <d3d11.h>
#include <cstdio>

static int g_failures = 0;

static void check(const char *what, HRESULT got, HRESULT want) {
  bool ok = got == want;
  if (!ok)
    g_failures++;
  std::printf("%s %-58s hr=0x%08lx (want 0x%08lx)\n", ok ? "PASS" : "FAIL", what, (unsigned long)got,
              (unsigned long)want);
}

static ID3D11Buffer *make_buffer(ID3D11Device *device, UINT bind) {
  D3D11_BUFFER_DESC bd = {};
  bd.ByteWidth = 8192 * 16;
  bd.Usage = D3D11_USAGE_DEFAULT;
  bd.BindFlags = bind;
  ID3D11Buffer *buffer = nullptr;
  HRESULT hr = device->CreateBuffer(&bd, nullptr, &buffer);
  if (FAILED(hr)) {
    std::printf("FAIL CreateBuffer bind=0x%x hr=0x%08lx\n", bind, (unsigned long)hr);
    g_failures++;
  }
  return buffer;
}

int main() {
  ID3D11Device *device = nullptr;
  ID3D11DeviceContext *context = nullptr;
  D3D_FEATURE_LEVEL level = {};
  HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION,
                                 &device, &level, &context);
  std::printf("D3D11CreateDevice hr=0x%08lx level=0x%x (%d-bit)\n", (unsigned long)hr, level,
              (int)sizeof(void *) * 8);
  if (FAILED(hr))
    return 2;

  D3D11_RENDER_TARGET_VIEW_DESC rd = {};
  rd.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
  rd.ViewDimension = D3D11_RTV_DIMENSION_BUFFER;
  rd.Buffer.FirstElement = 0;
  rd.Buffer.NumElements = 8192;

  // 1. The Frostbite call: must succeed and return a usable view.
  ID3D11Buffer *rtBuffer = make_buffer(device, D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE);
  ID3D11RenderTargetView *view = nullptr;
  if (rtBuffer) {
    check("buffer RTV, R32G32B32A32_FLOAT x 8192 elements", device->CreateRenderTargetView(rtBuffer, &rd, &view),
          S_OK);
  }
  if (view) {
    D3D11_RENDER_TARGET_VIEW_DESC got = {};
    view->GetDesc(&got);
    std::printf("     view desc: format=%u dimension=%u\n", got.Format, got.ViewDimension);
    // Exercise the view on the GPU: bind, clear, draw nothing, flush.
    const float colour[4] = {0.25f, 0.5f, 0.75f, 1.0f};
    D3D11_VIEWPORT vp = {0, 0, 8192, 1, 0, 1};
    context->OMSetRenderTargets(1, &view, nullptr);
    context->RSSetViewports(1, &vp);
    context->ClearRenderTargetView(view, colour);
    context->OMSetRenderTargets(0, nullptr, nullptr);
    context->Flush();
    std::printf("PASS bound, cleared and flushed the view\n");
    // The second view on the same buffer must also work (the game creates several).
    ID3D11RenderTargetView *second = nullptr;
    check("second buffer RTV on the same buffer", device->CreateRenderTargetView(rtBuffer, &rd, &second), S_OK);
    if (second)
      second->Release();
    view->Release();
  }

  // 2. Invalid calls stay invalid.
  if (rtBuffer) {
    check("buffer RTV with a NULL description", device->CreateRenderTargetView(rtBuffer, nullptr, &view), E_INVALIDARG);
    D3D11_RENDER_TARGET_VIEW_DESC unknown = rd;
    unknown.Format = DXGI_FORMAT_UNKNOWN;
    check("buffer RTV with DXGI_FORMAT_UNKNOWN", device->CreateRenderTargetView(rtBuffer, &unknown, &view),
          E_INVALIDARG);
    D3D11_RENDER_TARGET_VIEW_DESC tex2d = rd;
    tex2d.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    check("TEXTURE2D view dimension on a buffer", device->CreateRenderTargetView(rtBuffer, &tex2d, &view),
          E_INVALIDARG);
    rtBuffer->Release();
  }
  ID3D11Buffer *srvOnly = make_buffer(device, D3D11_BIND_SHADER_RESOURCE);
  if (srvOnly) {
    check("buffer RTV without BIND_RENDER_TARGET", device->CreateRenderTargetView(srvOnly, &rd, &view), E_INVALIDARG);
    srvOnly->Release();
  }

  context->Release();
  ULONG left = device->Release();
  std::printf("device released, refcount %lu\n", left);
  std::printf("%s: %d failure(s)\n", g_failures ? "FAILED" : "OK", g_failures);
  return g_failures ? 1 : 0;
}
