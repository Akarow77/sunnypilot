# M2 Mac accelerator 0.2.2 — bench research preview

Tested on a MacBook Air M2 and comma 3X. This release preserves the Apple Silicon
accelerator research and native macOS launcher while removing the live-camera
hook that interfered with local model timing. **Not a road-use release.**

## Results and outlook

- Mac-only Big Model inference using Core ML CPU + Neural Engine: 6,000 runs at
  20Hz over five minutes, mean 25.69ms, p99 27.18ms, maximum 39.53ms.
- That benchmark excludes live camera capture and USB. A real-camera experiment
  averaged 67.46ms from copy-start to output; 33/37 frames exceeded 55ms and none
  qualified. Compute performance is promising; end-to-end integration is unproven.
- Production modeld now matches the fetched official master baseline. Live capture
  activation is blocked, and standalone workers require normal timesharing.
- A recovery-operation parameter ownership error caused a later manager startup
  failure and was corrected. The test device was restored to official master;
  manager/UI startup and offroad state were verified. A separate earlier reboot
  remains unexplained. Neither rollback nor these tests establish driving safety.

## Download and setup

The ZIP contains the native arm64 **Sunnypilot Mac Accelerator.app** for macOS 15+.
It defaults to Mac-only localhost mode. Optional USB mode is for synthetic bench
diagnostics only. Server Ready does not mean driving readiness.

The app is ad-hoc signed, **not Apple-notarized**, and requires the matching source
checkout, Python environments, source model and converted Core ML package. It does
not bundle models, Python, credentials, personal paths or driving logs. Do not
disable Gatekeeper globally; build from inspected source if preferred.

- [Preparation and benchmark instructions](https://github.com/Akarow77/sunnypilot/blob/mac-accelerator-v0.2.2/tools/mac_accelerator/README.md)
- [M2 progress, measurements and research roadmap](https://github.com/Akarow77/sunnypilot/blob/mac-accelerator-v0.2.2/tools/mac_accelerator/PROJECT_STATUS.md)
- [Stability findings and recovery details](https://github.com/Akarow77/sunnypilot/blob/mac-accelerator-v0.2.2/tools/mac_accelerator/REVIEW_2026-09-16.md)

Only the app ZIP and its SHA-256 checksum are release assets. This is an independent
research project, not an official Apple, comma.ai or sunnypilot product.
