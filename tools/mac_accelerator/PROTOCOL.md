# Remote inference protocol outline

This is the proposed bench protocol for a comma 3X and an Apple Silicon Mac. It is
not approved for on-road control.

## Split point

The comma 3X keeps camera acquisition, calibration, frame selection, and the tinygrad
camera warp. It sends two contiguous `uint8[6,128,256]` tensors to the Mac. The Mac
keeps the recurrent image, desire, and feature queues and runs the driving policy.

Keeping the warp on the 3X reduces the 20 Hz payload from about 149.4 MB/s of padded
NV12 data to about 7.9 MB/s. The Mac returns the model-dependent float32 output:
2,576 values (about 10.3 KB) for the small model or 18,452 values (about 73.8 KB) for
the tested official big model.

## Request

Every request should contain:

- protocol magic and version;
- session ID, monotonically increasing frame ID, and capture monotonic timestamp;
- payload length and checksum;
- the two warped image tensors;
- desire pulse, traffic convention, and action-delay inputs;
- a reset flag for reconnects and model changes.

Prototype v1 uses a 40-byte network-order header (`SPMA`, version, type, flags,
session ID, frame ID, capture timestamp, payload length, CRC32). The request payload
is exactly 393,216 bytes of `uint8[2,6,128,256]`, followed by twelve little-endian
float32 values: eight desire pulses, two traffic-convention values, and two action
delay values.

Only one request may be in flight. A newer frame supersedes an unsent older frame;
queues must never be advanced for a dropped or out-of-order request.

## Response

Every response should contain:

- matching session and frame IDs;
- server receive, inference-start, and inference-end monotonic timestamps;
- model identity and protocol version;
- output length, checksum, and the model output tensor;
- a reset acknowledgement when requested.

The prototype response starts with three network-order monotonic timestamps (receive,
inference start, inference end), followed by the little-endian float32 model output.
The header echoes the request session, frame, and capture timestamp. Monotonic clocks
on the two machines are not compared directly; the client measures the round trip on
its own clock.

Before live integration, add a handshake that binds a session to the model hash,
input/output shapes, backend, and deadline. The current synthetic client supplies the
expected output length out of band and is not sufficient for automatic model changes.

## Failure behavior

The 3X must reject responses from an old session, duplicate or out-of-order frame IDs,
non-finite outputs, malformed lengths, checksum failures, and results that miss their
deadline. A remote disconnect must never block `modeld`.

For early testing, the normal 3X model must continue running as the control source.
The Mac output is shadow-only and logged for comparison. Promotion beyond shadow mode
requires route replay, fault injection, sustained thermal testing, and an explicit
fallback design review.

## Development stages

1. Loopback server/client on the Mac with synthetic warped inputs.
2. Route replay with recorded 3X frames and side-by-side local/remote outputs.
3. Wired bench connection to a powered 3X, remote output still shadow-only.
4. Inject packet loss, delays, disconnects, restarts, and corrupted payloads.
5. Decide whether the measured benefit justifies any further integration.
