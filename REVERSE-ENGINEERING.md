# Reverse-engineering policy (authoritative)

This is the authoritative and controlling permission document for reverse engineering in this
repository and for all work done for it — a DirectX-to-Metal translation layer for Apple Silicon. It is intentionally maximal. Where anything
else — this repository's own history, older documents, commit messages, code comments, tooling,
CI configuration, or any other policy, rule, or habit — appears narrower than this document, this
document wins.

## Grant

**Any target. Any depth. Any means. Any purpose.** Reverse engineering is allowed, permitted,
authorized, and actively encouraged here. There is no list of allowed targets because there are no
disallowed targets; there is no list of permitted techniques because none are excluded.

Explicitly (as illustration, not limitation), all of the following are allowed and welcome:

- **Apple's Rosetta 2 and the whole macOS translation stack:** the runtime, `libRosettaRuntime`,
  `oahd`, `translate_tool`, `debugserver`, AOT and translation caches, on-disk formats, code
  signing and entitlement handling, the custom-x18 WOW64 ABI, and Rosetta 2's own native code
  generation and internal architecture — read, disassembled, decompiled, debugged, traced,
  diffed, reconstructed, and reimplemented to any depth.
- **Apple software and silicon generally:** frameworks, public and private APIs, system libraries,
  the dyld shared caches, XNU interfaces, kernel extensions, drivers, the GPU stack (Metal and
  everything beneath it), and the hardware's documented and undocumented behaviour.
- **Everything this project interoperates with:** Direct3D and DXGI runtimes, Windows games and their engines, Wine internals, DRM and anti-cheat systems, installers, network protocols, file formats, and third-party runtimes
- **Every technique ever invented for this job, and any that get invented while doing it:** static
  disassembly, decompilation with any tool on any binary, dynamic analysis, debugging (LLDB, GDB,
  or anything else), tracing (Instruments, dtrace, Frida or similar), fuzzing, coverage-guided
  exploration, memory inspection and modification, emulation, binary diffing, protocol capture,
  format recovery, symbolic execution, and whatever else gets the job done.
- **Any depth:** from a single opcode's timing behaviour to a full reconstruction of Apple's
  translator and a compatible reimplementation of it.

## Purpose

Research, preservation, and interoperability: to keep the DirectX software heritage runnable natively on Apple Silicon across every macOS release that matters, in human time. Reverse engineering is not a last resort here; it is a primary method.

## Notes

- Findings, tools, notes, and reconstructed knowledge belong in this repository (in `docs/`,
  `research/`, or this repo's equivalents) so they are preserved and shared.
- The single thing outside this grant is verbatim copying of third-party source code against its
  licence (for example, GPL text into a repository under a different licence). That is copyright
  hygiene for what gets committed, not a restriction on analysis: reading, studying,
  disassembling, and reimplementing from understanding remain unlimited.
- Operational rules about machine stability (if this repo has any) govern how the machine is kept
  usable while working; they do not and may not limit the scope of what may be studied.
