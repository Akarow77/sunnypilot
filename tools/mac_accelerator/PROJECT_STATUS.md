# M2 Mac accelerator: progress and outlook

Status: 2026-09-16 · Research preview 0.2.2 · Tested on MacBook Air M2 + comma 3X

## 요약

Apple Silicon을 sunnypilot의 외부 추론 장치로 사용하는 연구입니다. **M2에서
Big Model 연산 자체를 20Hz로 실행할 가능성은 실측으로 확인했습니다.** Core ML의
CPU + Neural Engine 경로에서 5분간 6,000회 실행했고 평균 25.69ms, 최대
39.53ms였습니다. 이 결과는 후속 개발을 진행할 근거가 됩니다.

그러나 **실차 제어용 가속기는 완성되지 않았습니다.** 실제 3X 카메라 입력을
복사하고 USB로 전송한 실험은 55ms 기준을 지속적으로 만족하지 못했습니다.
로컬 모델을 지연시키는 동기식 GPU 읽기 코드는 제거했으며, 라이브 활성화 도구도
차단했습니다. 공개 앱은 Mac에서 서버와 벤치마크를 실행하는 연구용 런처입니다.

## What works

- Native arm64 macOS launcher, server status and logs, localhost test mode.
- Big Model conversion and persistent Core ML CPU + Neural Engine worker.
- Authenticated USB-NCM transport, model identity checking, frame IDs, checksums,
  finite-output checks, bounded timeouts, and optional lossless compression.
- Recorded-route replay, backend comparison, synthetic transport tests, and
  failure-injection/log-analysis tools.
- Chestnut-inspired service-state UI experiments. The Mac service is a user-space
  accelerator; it does not enumerate as a native PCIe GPU.

## Measurements and their limits

Different rows measure different parts of the system. They are not interchangeable.

| Experiment on M2 Air / 3X | Samples | Mean | p99 | Maximum | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Mac-only Core ML CPU + NE, paced 20Hz, 5 min | 6,000 | 25.69ms | 27.18ms | 39.53ms | No measured inference over 50ms; excludes camera/USB |
| Offroad USB, full-size random input, uncompressed | 160 | 48.23ms | 58.97ms | 64.10ms | Request-to-validated-response; excludes live readback |
| Offroad USB, full-size random input, Zstd | 160 | 51.05ms | 63.52ms | 71.88ms | Compression did not help this payload |
| Parked real-camera shadow, uncompressed | 37 | 67.46ms | 112.65ms | 117.07ms | Copy-start to output, including queue; 33/37 over 55ms |

The 37-frame live run had **zero qualified frames** and terminated after a QCOM
readback took 10.003ms. Camera-EOF-to-output averaged 73.54ms. Two earlier Zstd
live probes had 40/40 and 17/17 deadline misses and stopped at 13.101ms and
17.414ms readback overruns. Remote outputs were never published as vehicle controls.

The synthetic USB diagnostics used a 500ms stop timeout to collect latency
distributions. Their successful completion does not mean they passed 50ms/55ms
qualification. A later 80-frame synthetic check explicitly measured 34/80 over
the 50ms period. Short, repeated synthetic inputs do not represent road accuracy.

The five-minute Mac benchmark is retained as aggregate data in
[results/m2-coreml-5min.json](results/m2-coreml-5min.json).
The detailed [stability review](REVIEW_2026-09-16.md) records the live failures.
Raw driving logs, images, GPS/CAN data, device identifiers, and authentication keys
are not included in this release.

## Why the device failed, and what changed

1. **Local timing interference:** the old live hook performed synchronous
   `tensor.numpy()` before the local policy. The budget check happened after the
   delay, so it could not protect that frame. Production modeld now matches the
   fetched upstream baseline, and the live enable helper refuses to run.
2. **A confirmed later boot failure:** a recovery command wrote a parameter as
   root with mode 0600. The comma account could not copy it during boot logging,
   causing `Manager failed to start`. Restoring the specific file's ownership
   fixed the access fault. This was a recovery-operation error.
3. **Earlier whole-device reboot:** a reboot was observed while the shadow worker
   was absent. USB/power/kernel causes remain unproven. The absence of a running
   worker does not exclude every indirect effect of prior changes.

The test device was returned to official master for recovery. The research branch
retains the Mac tools and evidence. A code rollback is not a guarantee against an
unresolved hardware or OS fault.

## Outlook: promising compute, unresolved integration

M2 inference performance justifies continued bench research. Success is not assured:
remaining work concerns end-to-end timing, interference with local execution,
conversion fidelity, and hardware reliability. No evidence supports assigning a
numerical probability of success or promising a road-ready date.

Next steps, in order:

1. Establish a repeatable, stable 3X baseline with standard software and power;
   collect boot/power evidence if resets recur, without road experiments.
2. Explore asynchronous camera capture or shared memory **outside** the local model
   loop. Confirm whether QCOM and macOS APIs actually support the proposed path;
   zero-copy is a research question, not an implemented capability.
3. Measure realistic image payloads, queue age, and scheduling under sustained
   thermal load. Preserve model temporal ordering; dropping frames silently is
   not an acceptable substitute for 20Hz inference.
4. Expand original-model/Core ML comparisons over diverse recorded routes and
   long recurrent sequences. Curvature differences alone do not measure quality.
5. Run disconnect, process-failure, and stale-frame tests on a bench. Live capture
   can only return after isolation and timing are independently demonstrated.

## App and source

Download the **0.2.2 research preview** from the
[GitHub release](https://github.com/Akarow77/sunnypilot/releases/tag/mac-accelerator-v0.2.2).
The app is an ad-hoc signed, non-notarized launcher for macOS 15+ on Apple Silicon.
It needs this checkout, the Python environments and separately prepared model
artifacts described in the [setup guide](README.md). It is not a self-contained
model download or a device installer. Start with the default Mac-only mode.

Only MacBook Air M2 and comma 3X were tested here. Other M-series chips, macOS
versions, vehicles, and cables have not been qualified. This is an independent
experiment, not an official Apple, comma.ai, or sunnypilot product.
