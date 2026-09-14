# sunnypilot Mac accelerator experiment

This directory contains a bench-only prototype for using an Apple Silicon Mac as a
sunnypilot inference worker. It targets the comma 3X and the standard 20 Hz driving
model. It includes a policy-only Metal artifact, a checksummed TCP worker, USB-NCM
link setup, and a dependency-free synthetic client that runs on the 3X.

Do not use this experiment to control a vehicle. The live camera/modeld integration
and on-device fallback are not implemented. The current client sends synthetic
warped tensors and measures the shadow inference path only.

## Prepare

```bash
uv sync --frozen --all-extras
source .venv/bin/activate
git lfs install --local
git lfs pull --include=openpilot/selfdrive/modeld/models/driving_supercombo.onnx
```

## Compile and validate the Metal artifact

```bash
tools/mac_accelerator/compile_model.sh
```

The compiler performs an exact output and state-buffer comparison after serializing
and loading the JIT. `JIT=2` is intentional: Metal graph replay produced incorrect
round-trip results on the tested M2, while the ungrouped JIT passed exact validation.

Compile the policy-only worker artifact used by the USB prototype:

```bash
tools/mac_accelerator/compile_policy.sh
```

The official big model can also be compiled, but the tested M2 Air cannot run it at
20 Hz:

```bash
git lfs pull --include=openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx
MODEL="$PWD/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx" \
OUTPUT="$PWD/tools/mac_accelerator/artifacts/big_driving_policy_metal.pkl" \
  tools/mac_accelerator/compile_policy.sh
```

## Run with a comma 3X

Keep the 3X powered through OBD-C and connect its auxiliary port to the Mac with a
USB 3 cable. Enable ADB on the 3X. In the first Mac terminal, run:

```bash
tools/mac_accelerator/run_server.sh
```

The server binds only to the Mac's USB link-local address, not Wi-Fi. In a second
terminal, run the synthetic 20 Hz client on the 3X:

```bash
tools/mac_accelerator/run_3x_smoke_test.sh
```

The setup is intentionally transient. The 3X `usb0` interface is raised through
ADB and returns to its normal state after a reboot. The two small client files are
copied only to `/tmp/mac_accelerator` on the 3X.

## Sustained benchmark

```bash
tools/mac_accelerator/run_benchmark.sh --duration 300 \
  --json-output tools/mac_accelerator/artifacts/m2-5min.json
```

The report includes p50/p95/p99 latency, the number of runs exceeding the model's
50 ms period, and first-half versus second-half means as a basic fanless-Mac thermal
slowdown signal.

## Compare Metal with the CPU reference

Compile the same model for the CPU and compare it with Metal:

```bash
tools/mac_accelerator/compile_cpu_reference.sh
DEV=METAL JIT=2 .venv/bin/python tools/mac_accelerator/compare_backends.py
```

This feeds the same 3X-sized frames and policy inputs through both recurrent runners
and reports absolute and normalized output error for every frame.

## Replay a local route

Copy a completed segment's `fcamera.hevc`, `ecamera.hevc`, and `rlog.zst` into one
directory, then run:

```bash
DEV=METAL JIT=2 .venv/bin/python tools/mac_accelerator/route_model_benchmark.py \
  /path/to/segment --artifact /path/to/driving_tinygrad.pkl --frames 200
```

The replay streams both HEVC cameras, rebuilds the padded 3X NV12 buffers, and uses
the recorded device type, sensor, right-hand-drive state, and calibrated transform.

## Transport sizing

Sending both padded 3X NV12 frames would require about 149.4 MB/s at 20 Hz, before
protocol overhead. The intended split is therefore to perform the camera warp on the
comma 3X and send the two `6x128x256` model tensors instead. That payload is 393,216
bytes per inference, or about 7.9 MB/s at 20 Hz.

The tested 10 Gbps-rated cable negotiated at 5 Gbps because the 3X UDC reports both
`current_speed` and `maximum_speed` as `super-speed`. USB-NCM measured about 460 Mbps
for one 3X-to-Mac TCP stream and 964 Mbps total with four streams. Cable rating is
therefore not the limiting factor for the 63 Mbps model-input requirement.

## Safety boundary and next stage

The prototype uses monotonically increasing frame IDs, session resets, capture
timestamps, fixed payload sizes, CRC32, finite-output checks, and one request in
flight. The next stage must connect live 3X warps in shadow mode and reject results
over the deadline on the 3X. The local model must remain the control source whenever
the worker is late, warming up, disconnected, or invalid.

See [PROTOCOL.md](PROTOCOL.md) for the proposed split, message contents, and staged
failure-testing plan. [CHESTNUT_DESIGN.md](CHESTNUT_DESIGN.md) maps sunnypilot's
Chestnut loading, state reporting, and latched small-model fallback onto a Mac worker.

## Tested baseline

The repository was updated to official sunnypilot commit `c57f9a7`. The original
full-model baseline below was recorded at commit `3c24eeea2518` on a MacBook Air M2
(16 GB). Network and 3X preprocessing latency are not included in those rows.

| Run | Mean | p99 | Max | Over 50 ms | First half | Second half |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 60 seconds | 8.60 ms | 11.48 ms | 72.84 ms | 1 / 6,971 | 8.67 ms | 8.53 ms |
| 5 minutes | 11.06 ms | 19.67 ms | 176.52 ms | 15 / 27,086 | 9.20 ms | 12.92 ms |

The five-minute run slowed by about 40.5% from its first half to its second half.
Although p99 remained below the 50 ms model period, the long outliers require stale
result rejection and a local fallback.

Actual 3X USB-NCM policy-only synthetic shadow results, after five warm-up frames:

| Run | Mean | p99 | Max | Over 50 ms |
| --- | ---: | ---: | ---: | ---: |
| 1 minute / 1,200 frames | 33.61 ms | 48.31 ms | 49.28 ms | 0 / 1,200 |
| 5 minutes / 6,000 frames | 36.16 ms | 49.99 ms | 76.80 ms | 60 / 6,000 |
| Post-reboot / 200 frames | 32.89 ms | 57.40 ms | 76.53 ms | 4 / 200 |

The long run missed the 50 ms period on 1% of frames, and the post-reboot sample also
had misses. This is enough for a shadow experiment with local fallback, but it is not
safe or deterministic enough to become the vehicle's control source.

Actual calibrated route replay, using the same 200 camera-frame pairs:

| Model | Output floats | Mean | p99 | Max | Over 50 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small | 2,576 | 12.79 ms | 28.25 ms | 35.43 ms | 0 / 200 |
| Big | 18,452 | 114.87 ms | 118.26 ms | 147.28 ms | 200 / 200 |

Both produced finite outputs, but the official big model is not real-time on the
MacBook Air M2. Compiling successfully is not sufficient to use it as a 20 Hz driving
model.
