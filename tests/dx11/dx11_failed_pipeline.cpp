// Regression test: a draw whose pipeline state failed to compile must not be
// executed with the pipeline that happens to be bound from an earlier draw.
//
// Crysis 3 binds a geometry shader with a pixel shader that reads a varying the
// geometry shader never writes. DXMT turns VS+GS into a Metal mesh pipeline; on
// M1/M2 GPUs Apple's shader compiler crashes on that pipeline
// ("Failed to create mesh PSO: ... XPC_ERROR_CONNECTION_INTERRUPTED"). DXMT then
// skipped only the set-pipeline command and still encoded the mesh draw, which
// ran with the previously bound pipeline and its unrelated bindings; the GPU
// page-faulted and every later command buffer was ignored (the game hung).
//
// Each mode draws twice into one render pass:
//   A: left half of the target, a valid pipeline whose pixel shader returns red;
//   B: right half, same input stages, but a pixel shader that reads TEXCOORD1,
//      which the previous stage does not write, so B's pipeline fails to compile.
// If B is encoded with A's pipeline, the right half turns red. With the fix B is
// skipped and the right half keeps the clear color (black).
//
// Modes: vs - ordinary VS->PS pipeline (Metal rejects the link with an error, on
//             every GPU);
//        gs - VS->GS->PS mesh pipeline (the compiler crash; on Apple9+ GPUs the
//             pipeline may compile, which the test reports as SKIP-free PASS).
// After the draws the test checks that the GPU queue still executes work, idles so
// the threadpool workers that compiled the pipelines exit, and checks again.
//
// Mode "control" runs both paths with a pixel shader that links (B renders green).
//
// usage: dx11_failed_pipeline [vs|gs|all|control] [idle_ms (default 15000)]
// Exit 0: pass. 1: the failed draw was executed with a stale pipeline, A did not
// draw, or the queue stopped executing work. 2: setup error (device, shader
// compilation). 3: the process faulted. A hang (killed by the runner's timeout)
// is a failure too.
#include <d3d11.h>
#include <d3dcompiler.h>
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static LONG WINAPI on_fault(EXCEPTION_POINTERS *info) {
  std::printf("FAULT code=0x%08lx at %p\n", (unsigned long)info->ExceptionRecord->ExceptionCode,
              info->ExceptionRecord->ExceptionAddress);
  std::fflush(stdout);
  ExitProcess(3);
  return EXCEPTION_EXECUTE_HANDLER;
}

static const char kShaders[] = R"(
struct VSOut { float4 pos : SV_Position; float4 c0 : TEXCOORD0; };

// Fullscreen triangle for the VS-only mode.
VSOut vs_tri(uint id : SV_VertexID) {
  VSOut o;
  float2 uv = float2((id << 1) & 2, id & 2);
  o.pos = float4(uv * float2(2, -2) + float2(-1, 1), 0, 1);
  o.c0 = float4(0, 1, 0, 1);
  return o;
}

// One point; the geometry shader expands it into a fullscreen quad.
VSOut vs_point(uint id : SV_VertexID) {
  VSOut o;
  o.pos = float4(0, 0, 0, 1);
  o.c0 = float4(0, 1, 0, 1);
  return o;
}

[maxvertexcount(4)]
void gs_quad(point VSOut i[1], inout TriangleStream<VSOut> s) {
  VSOut o;
  o.c0 = i[0].c0;
  o.pos = float4(-1, -1, 0, 1); s.Append(o);
  o.pos = float4(-1,  1, 0, 1); s.Append(o);
  o.pos = float4( 1, -1, 0, 1); s.Append(o);
  o.pos = float4( 1,  1, 0, 1); s.Append(o);
}

float4 ps_red(VSOut i) : SV_Target { return float4(1, 0, 0, 1); }

// Reads TEXCOORD1, which neither vs_tri nor gs_quad writes.
struct PSInBad { float4 pos : SV_Position; float4 c0 : TEXCOORD0; float4 c1 : TEXCOORD1; };
float4 ps_unlinked(PSInBad i) : SV_Target { return float4(0, i.c0.y, 0, 1) + i.c1 * 0.5; }

// Control: reads only what the previous stage writes.
float4 ps_green(VSOut i) : SV_Target { return float4(0, i.c0.y, 0, 1); }
)";

static ID3DBlob *compile(const char *entry, const char *target) {
  ID3DBlob *code = nullptr, *errors = nullptr;
  HRESULT hr = D3DCompile(kShaders, sizeof(kShaders) - 1, "dx11_failed_pipeline.hlsl", nullptr, nullptr, entry, target,
                          0, 0, &code, &errors);
  if (FAILED(hr)) {
    std::printf("D3DCompile(%s, %s) failed 0x%08lx: %s\n", entry, target, (unsigned long)hr,
                errors ? (const char *)errors->GetBufferPointer() : "");
    return nullptr;
  }
  return code;
}

struct Rgba {
  unsigned char r, g, b, a;
};

// Clears a fresh target to blue and reads it back. After a GPU fault Metal ignores
// every later command buffer of the queue, so the readback is not blue.
static bool check_alive(ID3D11Device *dev, ID3D11DeviceContext *ctx, const char *tag) {
  D3D11_TEXTURE2D_DESC td = {};
  td.Width = 4;
  td.Height = 4;
  td.MipLevels = 1;
  td.ArraySize = 1;
  td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  td.SampleDesc.Count = 1;
  td.Usage = D3D11_USAGE_DEFAULT;
  td.BindFlags = D3D11_BIND_RENDER_TARGET;
  ID3D11Texture2D *rt = nullptr, *staging = nullptr;
  ID3D11RenderTargetView *rtv = nullptr;
  if (FAILED(dev->CreateTexture2D(&td, nullptr, &rt)) || FAILED(dev->CreateRenderTargetView(rt, nullptr, &rtv)))
    return false;
  td.Usage = D3D11_USAGE_STAGING;
  td.BindFlags = 0;
  td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  if (FAILED(dev->CreateTexture2D(&td, nullptr, &staging)))
    return false;
  const float blue[4] = {0, 0, 1, 1};
  ctx->ClearRenderTargetView(rtv, blue);
  ctx->CopyResource(staging, rt);
  D3D11_MAPPED_SUBRESOURCE map;
  bool ok = false;
  if (SUCCEEDED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &map))) {
    Rgba p = *(Rgba *)map.pData;
    ok = p.r == 0 && p.g == 0 && p.b == 255;
    std::printf("[%s] GPU liveness readback = %u,%u,%u,%u (%s)\n", tag, p.r, p.g, p.b, p.a,
                ok ? "ok" : "the queue no longer executes work");
    ctx->Unmap(staging, 0);
  }
  staging->Release();
  rtv->Release();
  rt->Release();
  return ok;
}

static int run_mode(ID3D11Device *dev, ID3D11DeviceContext *ctx, bool use_gs, bool control) {
  const UINT W = 64, H = 32;
  const char *mode = use_gs ? (control ? "gs-control" : "gs") : (control ? "vs-control" : "vs");

  ID3DBlob *vs_blob = compile(use_gs ? "vs_point" : "vs_tri", "vs_5_0");
  ID3DBlob *gs_blob = use_gs ? compile("gs_quad", "gs_5_0") : nullptr;
  ID3DBlob *ps_a_blob = compile("ps_red", "ps_5_0");
  ID3DBlob *ps_b_blob = compile(control ? "ps_green" : "ps_unlinked", "ps_5_0");
  if (!vs_blob || (use_gs && !gs_blob) || !ps_a_blob || !ps_b_blob)
    return 2;

  ID3D11VertexShader *vs = nullptr;
  ID3D11GeometryShader *gs = nullptr;
  ID3D11PixelShader *ps_a = nullptr, *ps_b = nullptr;
  if (FAILED(dev->CreateVertexShader(vs_blob->GetBufferPointer(), vs_blob->GetBufferSize(), nullptr, &vs)) ||
      (use_gs && FAILED(dev->CreateGeometryShader(gs_blob->GetBufferPointer(), gs_blob->GetBufferSize(), nullptr, &gs))) ||
      FAILED(dev->CreatePixelShader(ps_a_blob->GetBufferPointer(), ps_a_blob->GetBufferSize(), nullptr, &ps_a)) ||
      FAILED(dev->CreatePixelShader(ps_b_blob->GetBufferPointer(), ps_b_blob->GetBufferSize(), nullptr, &ps_b))) {
    std::printf("[%s] shader creation failed\n", mode);
    return 2;
  }

  D3D11_TEXTURE2D_DESC td = {};
  td.Width = W;
  td.Height = H;
  td.MipLevels = 1;
  td.ArraySize = 1;
  td.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
  td.SampleDesc.Count = 1;
  td.Usage = D3D11_USAGE_DEFAULT;
  td.BindFlags = D3D11_BIND_RENDER_TARGET;
  ID3D11Texture2D *rt = nullptr, *staging = nullptr;
  ID3D11RenderTargetView *rtv = nullptr;
  if (FAILED(dev->CreateTexture2D(&td, nullptr, &rt)) || FAILED(dev->CreateRenderTargetView(rt, nullptr, &rtv)))
    return 2;
  td.Usage = D3D11_USAGE_STAGING;
  td.BindFlags = 0;
  td.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
  if (FAILED(dev->CreateTexture2D(&td, nullptr, &staging)))
    return 2;

  const float black[4] = {0, 0, 0, 1};
  ctx->ClearRenderTargetView(rtv, black);
  ctx->OMSetRenderTargets(1, &rtv, nullptr);
  ctx->IASetInputLayout(nullptr);
  ctx->IASetPrimitiveTopology(use_gs ? D3D11_PRIMITIVE_TOPOLOGY_POINTLIST : D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
  ctx->VSSetShader(vs, nullptr, 0);
  ctx->GSSetShader(gs, nullptr, 0);
  UINT vertex_count = use_gs ? 1 : 3;

  std::printf("[%s] draw A (valid pipeline), draw B (unlinkable pipeline), read back\n", mode);
  // A: valid pipeline, left half, red.
  D3D11_VIEWPORT vp = {0, 0, W / 2.0f, (float)H, 0, 1};
  ctx->RSSetViewports(1, &vp);
  ctx->PSSetShader(ps_a, nullptr, 0);
  ctx->Draw(vertex_count, 0);

  // B: pipeline that cannot be linked, right half.
  vp.TopLeftX = W / 2.0f;
  ctx->RSSetViewports(1, &vp);
  ctx->PSSetShader(ps_b, nullptr, 0);
  ctx->Draw(vertex_count, 0);

  ctx->CopyResource(staging, rt);
  D3D11_MAPPED_SUBRESOURCE map;
  if (FAILED(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &map))) {
    std::printf("[%s] Map failed\n", mode);
    return 1;
  }
  Rgba left = *(Rgba *)((char *)map.pData + (H / 2) * map.RowPitch + (W / 4) * 4);
  Rgba right = *(Rgba *)((char *)map.pData + (H / 2) * map.RowPitch + (3 * W / 4) * 4);
  ctx->Unmap(staging, 0);

  std::printf("[%s] left (valid draw) = %u,%u,%u,%u  right (unlinkable draw) = %u,%u,%u,%u\n", mode, left.r, left.g,
              left.b, left.a, right.r, right.g, right.b, right.a);
  int result = 0;
  if (!(left.r == 255 && left.g == 0 && left.b == 0)) {
    std::printf("[%s] FAIL: the valid draw did not render red\n", mode);
    result = 1;
  }
  if (right.r == 255 && right.g == 0 && right.b == 0) {
    std::printf("[%s] FAIL: the draw with the failed pipeline ran with the previous (red) pipeline\n", mode);
    result = 1;
  } else if (right.r == 0 && right.g == 0 && right.b == 0) {
    std::printf("[%s] ok: the draw with the failed pipeline was skipped\n", mode);
  } else {
    std::printf("[%s] ok: the unlinkable draw rendered with its own pipeline (it compiled on this GPU)\n", mode);
  }

  staging->Release();
  rtv->Release();
  rt->Release();
  ps_b->Release();
  ps_a->Release();
  if (gs)
    gs->Release();
  vs->Release();
  return result;
}

int main(int argc, char **argv) {
  SetUnhandledExceptionFilter(on_fault);
  setvbuf(stdout, nullptr, _IONBF, 0);
  const char *which = argc > 1 ? argv[1] : "all";
  int idle_ms = argc > 2 ? std::atoi(argv[2]) : 15000;

  ID3D11Device *dev = nullptr;
  ID3D11DeviceContext *ctx = nullptr;
  D3D_FEATURE_LEVEL level = D3D_FEATURE_LEVEL_11_0;
  if (FAILED(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, &level, 1, D3D11_SDK_VERSION, &dev,
                               nullptr, &ctx))) {
    std::printf("D3D11CreateDevice failed\n");
    return 2;
  }

  int result = 0;
  bool control = !std::strcmp(which, "control");
  if (!std::strcmp(which, "vs") || !std::strcmp(which, "all") || control) {
    int r = run_mode(dev, ctx, false, control);
    if (r > result)
      result = r;
  }
  if (!std::strcmp(which, "gs") || !std::strcmp(which, "all") || control) {
    int r = run_mode(dev, ctx, true, control);
    if (r > result)
      result = r;
  }
  if (result < 2 && !check_alive(dev, ctx, "after draws"))
    result = 1;
  // Pipelines are compiled on threadpool workers, which exit when idle. A worker that
  // received an over-released Metal error object crashes while it exits.
  std::printf("idling %d ms so idle compile threads exit\n", idle_ms);
  Sleep(idle_ms);
  if (result < 2 && !check_alive(dev, ctx, "after idle"))
    result = 1;
  ctx->Release();
  dev->Release();
  std::printf("RESULT %s\n", result == 0 ? "PASS" : result == 1 ? "FAIL" : "ERROR");
  return result;
}
