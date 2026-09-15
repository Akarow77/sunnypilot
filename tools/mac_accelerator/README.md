# sunnypilot Mac accelerator experiment

This directory contains a bench-only prototype for using an Apple Silicon Mac as a
sunnypilot inference worker. It targets the comma 3X and the standard 20 Hz driving
model. It includes Metal and Core ML/ANE workers plus authenticated and checksummed
USB-NCM transport. Requests are always lossless; the live Big Model shadow sends
uncompressed uint8 warps, while Zstd remains available for synthetic diagnostics.
Live Big Model shadow uses NumPy on the 3X; the framing layer itself has only
standard-library dependencies.

Do not use this experiment to control a vehicle. Live camera/modeld integration is
implemented only as a non-controlling shadow, and the sustained deadline requirement
has not passed. The client recognizes the Mac as a user-space accelerator, sends the
same warped tensors used by the local model, and exercises a fail-closed shadow path.
A Mac cannot enumerate
as a native PCIe/USB GPU on the 3X; the authenticated service identity is the honest
equivalent.

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

The official big model can also be compiled for Metal, but the tested M2 Air cannot
run that backend at 20 Hz:

```bash
git lfs pull --include=openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx
MODEL="$PWD/openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx" \
OUTPUT="$PWD/tools/mac_accelerator/artifacts/big_driving_policy_metal.pkl" \
  tools/mac_accelerator/compile_policy.sh
```

## Compile the Big Model for Core ML/ANE

Core ML is the only tested backend that runs the official Big Model below 50 ms on
the M2 Air itself. Create the isolated, pinned conversion environment and compile:

```bash
tools/mac_accelerator/setup_coreml_env.sh
tools/mac_accelerator/compile_coreml.sh
```

The converter promotes the ONNX graph to consistent FP32 storage to work around a
Core ML convolution import mismatch, then requests Core ML FLOAT16 lowering. Keep
the original ONNX and its SHA-256 as the protocol identity.

## Run with a comma 3X

Keep the 3X powered through OBD-C and connect its auxiliary port to the Mac with a
USB 3 cable. Enable ADB on the 3X. In the first Mac terminal, run:

```bash
tools/mac_accelerator/run_server.sh
```

The Metal server binds only to the Mac's USB link-local address, not Wi-Fi. In a
second terminal, run the synthetic 20 Hz client on the 3X:

```bash
tools/mac_accelerator/run_3x_smoke_test.sh
```

For the Big Model Core ML/ANE worker, first generate a private 32-byte shared key,
then start the server:

```bash
openssl rand -out /path/outside-the-repository/mac-accelerator.key 32
AUTH_KEY_FILE=/path/outside-the-repository/mac-accelerator.key \
  tools/mac_accelerator/run_coreml_server.sh
```

From another terminal, run the 3X test with the expected SHA printed by
`shasum -a 256 openpilot/selfdrive/modeld/models/big_driving_supercombo.onnx`:

```bash
AUTH_KEY_FILE=/path/outside-the-repository/mac-accelerator.key \
EXPECTED_BACKEND=COREML_ANE EXPECTED_OUTPUT_FLOATS=18452 \
EXPECTED_MODEL_SHA256=<64-character SHA-256> COMPRESSION=zstd-1 \
SYNTHETIC_RANDOM_PREFIX_BYTES=250000 WARMUP_FRAMES=40 \
QUALIFICATION_FRAMES=20 tools/mac_accelerator/run_3x_smoke_test.sh
```

The setup is intentionally transient. The 3X `usb0` interface is raised through
ADB and returns to its normal state after a reboot. The client files are copied
only to `/tmp/mac_accelerator` on the 3X.

## Run the live, non-controlling shadow

This requires a sunnypilot Tinygrad model bundle with a separated warp and policy.
The currently tested TSFM 20 Hz bundle has that layout. With the 3X off-road and the
Mac app running, configure the authenticated USB endpoint and persistent 3X key:

```bash
tools/mac_accelerator/enable_live_shadow.sh
```

On the next on-road transition, `modeld_tinygrad` takes a snapshot immediately after
the QCOM camera warp and before running the local policy. A bounded, nonblocking mmap
mailbox feeds a separate process, so USB transfer and Mac inference overlap with the
local policy. The worker runs on 3X core 2 at FIFO priority 5; modeld remains priority
54 on core 7, and controls/planning retain their higher dedicated priorities. The
local TSFM model remains the sole publisher of control-related model messages.

The live path deliberately uses uncompressed requests. On the connected 3X, a
full-size high-entropy 393 KiB input measured 48.23 ms mean / 54.05 ms p95
uncompressed, versus 51.05 ms mean / 56.30 ms p95 with Zstd. Real parked camera
warps made Zstd preparation take 12--13 ms on core 2, so compression increased
latency and CPU contention despite reducing bytes on the wire.

Two parked live-camera Zstd probes never became eligible: the first recorded
40/40 copy-to-output deadline misses (92.14 ms mean, 110.63 ms p95) and stopped
when one QCOM readback reached 13.10 ms; the second recorded 17/17 misses
(109.03 ms mean, 192.64 ms p95) and stopped at a 17.41 ms readback. These
measurements are the reason the live default is uncompressed and the automatic
run remains disabled until another parked qualification is explicitly enabled.

The shadow process never publishes remote output. A malformed response, disconnect,
or active copy-to-output deadline miss stops the shadow session. Initial cold
frames remain in loading state until at least 66 contiguous frames have populated
the Big Model temporal context and 66 consecutive results meet both timing limits.
Missing camera frames reset recurrent queues and readiness; duplicate/backward,
stale, uncalibrated, or unsynchronized inputs fail the session.

**The GPU-to-host readback is still synchronous.** The initial real-device probe is
restricted to a parked car with lateral control inactive and uses a 10 ms snapshot
budget so the actual QCOM readback cost can be measured. Moving above 0.5 m/s or
activating lateral control stops frame delivery and the worker fails closed. The
budget disables future snapshots after an overrun; it cannot prevent or undo the
first slow copy. Capturing before the local policy minimizes output age but no longer
has the same-frame local curvature available for live comparison; recorded-route
backend validation remains separate. Do not treat this measurement mode as a driving
configuration. Real-device A/B timing remains required.

The worker waits in loading state without a fixed deadline until the first calibrated
frame arrives, connects only for that first frame, detects a subsequent 500 ms source
stall, and emits a heartbeat at most four times per second. The UI rejects missing,
future, or more-than-two-second-old heartbeats. Logs stop at 32 MiB per session;
startup refuses a directory already at 256 MiB (existing logs are not deleted).
These are diagnostic thresholds, not a vehicle-control safety specification.

Per-frame timing and local-versus-Big curvature are written on the 3X under
`/data/media/0/mac_accelerator_shadow/live-*.jsonl`. After a run, copy a log and run:

```bash
.venv/bin/python tools/mac_accelerator/summarize_live_shadow.py /path/to/live-log.jsonl
```

`copyToOutputMs` includes snapshot, queue, compression, transfer, inference, and
validation. `captureToOutputMs` additionally includes time since camera EOF; it is
not the same as sensor exposure latency. Only qualified frames above 5 m/s enter
the curvature summary. This compares raw Big Model curvature with the local
postprocessed action: difference is not an accuracy score or proof of superiority.

Disable the next run only while off-road:

```bash
tools/mac_accelerator/disable_live_shadow.sh
```

## macOS app and dedicated qualification

Build the native arm64 launcher app with the installed Command Line Tools:

```bash
tools/mac_accelerator/build_macos_app.sh
open "tools/mac_accelerator/dist/Sunnypilot Mac Accelerator.app"
```

The app creates a mode-0600 shared key under the user's Application Support
directory, keeps the Mac awake, limits auxiliary math-library threads, and starts
the Core ML/ANE worker. It shows server and authenticated 3X connection status but
does not publish vehicle-control outputs.

The launcher supervises the scoped USB-NCM endpoint. If a 3X reboot or cable
reconnection changes the macOS interface index, it raises `usb0`, rediscovers both
link-local addresses, and restarts the server on the new scope. An unchanged-link
model/server failure remains stopped instead of entering a restart loop.

The app requires macOS 15 or newer for the converted model. Select **Mac-only test
(localhost, no 3X)** before Start to avoid ADB/USB setup. Server Ready / Client
connected are service states, not driving readiness or a latency qualification.

For an end-to-end Mac-only test including the independent shadow process:

```bash
.coreml-venv/bin/python tools/mac_accelerator/local_shadow_benchmark.py \
  --frames 300 --output-dir tools/mac_accelerator/artifacts/local-test-001
```

Use a new output directory each run. `--drop-at 100` injects a missing camera
frame; `--stall-at 220` pauses the producer for 700 ms; `--disconnect-at 100`
terminates the test server. Failure injections intentionally exit nonzero and must
be checked against the recorded reason. Synthetic camera timestamps and curvature
cannot validate a real driving model. Exit zero indicates plumbing completed;
`sustainedDeadlinePass` separately records strict deadline completion/readiness.

The current handshake hashes the source ONNX, not the converted Core ML package.
It authenticates the peer's source-model declaration, **not conversion provenance**.
Do not replace the package independently of its source/metadata; reproducible
artifact provenance and longer numerical validation remain outstanding.

To test the hypothesis that other Mac workloads caused latency spikes, quit heavy
applications and run the paced five-minute qualification:

```bash
tools/mac_accelerator/run_coreml_qualification.sh
```

The command fails on any inference over 50 ms and reports first-half versus
second-half latency plus scheduler lateness. A passing Mac-only run is necessary
but not sufficient: the 3X warp, compression, USB round trip, full-route accuracy,
disconnect injection, and local fallback must also pass.

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

Protocol v2 uses mutually authenticated HMAC-SHA256 discovery, per-request and
per-response HMAC tags, monotonically increasing frame IDs, session resets, capture
timestamps, fixed uncompressed sizes, CRC32, finite-output checks, optional lossless
Zstd, an absolute client deadline, and one request in flight. Loading allows a
separate cold-start timeout but requires consecutive in-deadline frames before the
accelerator becomes ready. Any active failure is latched until reset.

The live stage reuses the separated Tinygrad warp without running a second QCOM warp,
starts transfer before the local policy, and measures copy-to-output plus
camera-EOF-to-output timing. The local model remains
the only control source while the worker is late, warming up, disconnected, or
invalid. Shared CPU, memory bandwidth, and synchronous readback effects still need
real-device qualification.

On macOS the persistent Core ML server uses CPU + Neural Engine, requests the highest
public `user-interactive` QoS, verifies that QoS took effect, enables Core ML's
`FastPrediction` specialization, and is launched under a latency-critical process
activity. This is a scheduler hint, not a hard real-time guarantee.

When the comma-side test is running, `MacAccelerator*` parameters map its state to
the existing Chestnut icons: loading pulses, ready/active is green, and a latched
failure is orange. The model manager still knows it is not physical Chestnut, so a
green shadow-ready icon cannot silently select a Chestnut model or vehicle-control
path.

See [PROTOCOL.md](PROTOCOL.md) for the proposed split, message contents, and staged
failure-testing plan. [CHESTNUT_DESIGN.md](CHESTNUT_DESIGN.md) maps sunnypilot's
Chestnut loading, state reporting, and latched small-model fallback onto a Mac worker.

## Tested baseline

The repository was updated to official sunnypilot commit `63a2a38`. The original
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

Those rows use the tinygrad Metal backend. The Big Model is not real-time with that
backend on the MacBook Air M2.

The Core ML/ANE conversion materially improves inference speed on the same Mac:

| Big Model test | Mean | p99 | Max | Missed 50 ms |
| --- | ---: | ---: | ---: | ---: |
| Mac only, CPU + Neural Engine / 5 min at 20 Hz | 25.69 ms | 27.18 ms | 39.53 ms | 0 / 6,000 |
| 3X USB, Zstd synthetic / 400 frames | 38.68 ms | 41.50 ms | 51.53 ms | 0 / 400 |
| 3X USB, full high-entropy input, uncompressed / 160 frames | 48.23 ms | 58.97 ms | 64.10 ms | not recorded |
| 3X USB, full high-entropy input, Zstd / 160 frames | 51.05 ms | 63.52 ms | 71.88 ms | not recorded |
| Warm Mac + 3X USB diagnostic | 46.88 ms | 53.51 ms | 53.77 ms | observed |
| FP16 response + full validation / 200 frames | 40.77 ms | 44.29 ms | 44.48 ms | 0 / 200 |
| Recorded route, Metal warp + ANE / 3,150 frames | 21.75 ms | 23.56 ms | 55.72 ms | 1 / 3,150 |

FP16 preserves the Core ML model's native output precision while reducing the Big
Model response from about 73.8 KB to 36.9 KB. Reusing the comma-side finite-check and
FP32 conversion buffers also removes per-frame validation allocations. These changes
reduced mean output validation from 3.31 ms to about 0.35 ms.

They do not make the link deterministic. One extended attempt latched failure after
541 measured frames at 50.30 ms. A second attempt failed after 110 measured frames at
50.12 ms; the Mac log identified a 52.97 ms Core ML inference and 53.94 ms total
service time on that frame. The single-frame fail-closed behavior worked in both
cases.

The recorded-route row replays both saved 1928x1208 camera streams, applies the
logged calibration and action-delay inputs, and preserves recurrent state across
segments of the same route. A repeat over the 1,950-frame moving route measured
21.76 ms mean, 24.08 ms p99, and two frames above 55 ms. This demonstrates that
Mac-local camera warp plus ANE inference is normally well inside one model period,
but rare scheduling/ANE outliers remain even without USB transport.

At speeds of at least 5 m/s, 590 camera-frame-aligned outputs were compared with the
recorded local Small Model. Big-versus-Small desired-curvature correlation was 0.927,
with 0.00350 mean absolute error, 0.01546 p95 absolute error, and 0.05386 maximum
absolute error. These are different models rather than equivalent backends, so the
comparison describes behavioral divergence and does not qualify the Big Model for
vehicle control.

An exact five-frame recorded-route temporal comparison against the original PyTorch
model measured 0.584% maximum overall normalized RMSE. Individual output heads ranged
up to about 2.17%, so more route coverage is required before judging equivalence.

Core ML/ANE can execute the Big Model itself at 20 Hz, but the fanless M2 Air plus
USB transport, scheduling, and 3X warp does not sustain a hard 50 ms
capture-to-output deadline. A parked live-camera run also accumulated queue delay
and failed its 55 ms qualification before any remote output became eligible. The
current result is suitable only for continued off-road shadow testing; it is not a
vehicle-control accelerator.
