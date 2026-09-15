# Remote inference protocol v2

This is the implemented bench protocol between a comma 3X and an Apple Silicon
Mac. It is not approved for on-road control.

## Split point

The comma 3X retains camera acquisition, calibration, frame selection, and the
tinygrad camera warp. It sends two contiguous `uint8[6,128,256]` tensors to the
Mac. The Mac owns the recurrent image, desire, and feature queues and runs the
driving policy.

Keeping the warp on the 3X reduces the 20 Hz payload from about 149.4 MB/s of
padded NV12 data to 393,216 bytes per inference before compression, or about
7.9 MB/s at 20 Hz. The Core ML Big Model response preserves its native 18,452
float16 values (about 36.9 KB); the 3X converts these to float32 before parsing.

## Framing and discovery

Every message starts with the 40-byte, network-order `SPMA` header defined in
`transport.py`: protocol version, message type, flags, session ID, frame ID,
capture monotonic timestamp, payload length, and CRC32. Payloads are capped at
4 MiB.

The client begins each connection with a nonce-bearing JSON `HELLO`. The server
returns an identity containing its device type, backend, model checkpoint,
source-model SHA-256, input/output shapes, output slices, frame skip, output length
and dtype, request length, and supported compression modes. The client rejects any identity that differs from
its configured expectations.

The source hash is currently a server declaration computed from `--onnx`; it is
not cryptographically bound to `--model` (the Core ML package). Authentication
does not by itself establish conversion provenance or numerical equivalence.

When a shared key is configured, both sides prove possession with HMAC-SHA256
over canonical handshake JSON. Every inference request and response also carries
an HMAC-SHA256 tag covering the protocol metadata and payload. The key must be at
least 32 bytes and is never stored in the repository.

## Inference request and response

The uncompressed request is exactly 393,264 bytes: two warped image tensors,
eight desire-pulse floats, two traffic-convention floats, and two action-delay
floats. `FLAG_ZSTD_REQUEST` losslessly compresses this request with Zstd level 1;
the declared uncompressed size remains fixed. `FLAG_RESET` resets recurrent
state and must be acknowledged by the response.

The response contains server receive, inference-start, and inference-end
monotonic timestamps followed by the model's declared little-endian output. The
header echoes the request session, frame, capture timestamp, and accepted flags.
Monotonic clocks on different machines are not compared; the client measures the
complete request round trip with its own clock.

Only one request may be in flight. Frame IDs must be sequential at the client and
strictly increasing at the server. A newer camera frame may replace an unsent old
frame. The shadow preserves real camera frame IDs separately from sequential
wire IDs, resetting recurrent queues and qualification after any camera gap.
Duplicate/backward frames are rejected. Big Model requires `frame_skip=2` and
66 contiguous context frames before comparison/readiness.

## Readiness and failure behavior

Loading has a separate cold-start timeout. The accelerator becomes ready only
after a configured number of consecutive frames finish within the active
deadline. Once active, a bad identity, malformed length, checksum or HMAC error,
stale response, non-finite output, disconnect, or deadline miss latches the
client into the failed state until an explicit reset or ignition cycle.

Network/compression/logging work runs in a separate process on core 2 at FIFO priority
5, below modeld and the control/planning priorities. The optional host snapshot starts
after the QCOM warp and before the local policy so remote work can overlap the local
policy. Readback is still synchronous: the 8 ms snapshot tripwire prevents subsequent
copies, not the first overrun. No zero-interference claim is established. Current USB and Mac-only tests
do not establish a sustained hard deadline, so the implementation remains shadow-only.

Receive timeouts use one absolute deadline across header, payload, and fragmented
reads. Active live checks include the full snapshot-to-response age (default 55 ms)
and camera EOF age (150 ms). Warm-up allows a 500 ms individual request but requires
66 consecutive in-limit frames. A 500 ms input stall fails the worker; a 2 s stale
heartbeat prevents the 3X UI remaining green after worker death. Terminal failures
are logged separately from completed-frame deadline misses so missing responses
are not mistaken for successful frames. These are diagnostic, not safety, limits.

## Development stages

1. Loopback server/client on the Mac with synthetic warped inputs. Completed.
2. Recorded-route Core ML versus PyTorch validation. Completed for a short sample.
3. Wired comma 3X discovery and synthetic USB-NCM inference. Completed off-road.
4. Live camera-warp shadow integration. Implemented for sunnypilot's separated
   Tinygrad warp/policy bundles; fault-injection qualification remains pending.
5. Sustained thermal and full-route qualification. Not passed.
6. Any control-path proposal requires a separate safety design and review.
