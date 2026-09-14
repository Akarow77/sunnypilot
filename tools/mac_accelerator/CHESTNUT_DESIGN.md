# Chestnut execution pattern applied to a Mac worker

The current sunnypilot `modeld` and `modeld_v2` Chestnut paths establish the safety
pattern this experiment should follow. Chestnut is detected separately from model
availability, reports hardware health, warms the big model under a 60 second loading
limit, and keeps a small model instance available even after the big model starts.
Any runtime exception latches `ChestnutModelError`, clears `ChestnutActive`, drops the
failed frame, and switches permanently to the small model for that drive.

## Directly reusable behavior

- Treat physical presence, readiness, loading, active, and failed as distinct states.
- Warm the accelerator before marking it active.
- Keep the local small model initialized while the accelerator is active.
- Never retry a failed accelerator while driving; retry only after an explicit reset
  or ignition cycle.
- Drop the frame that triggered fallback instead of publishing a questionable result.
- Expose health continuously. Chestnut publishes power, PCIe, temperature, clock,
  usage, and fan data at 10 Hz.
- Notify `selfdrived` when the active model disappears so it can soft-disable.

## Mac-specific readiness

The Mac has no Chestnut VID/PID, PCIe LTSSM, or 12 V telemetry. Its readiness check
must instead require all of the following:

1. the comma USB gadget is negotiated at SuperSpeed;
2. USB-NCM is active on both ends;
3. the worker handshake matches protocol version, model hash, input/output shapes,
   and Metal backend;
4. warm-up completes within 60 seconds;
5. a configured number of shadow frames pass CRC, identity, finite-output, and
   deadline checks.

The health message should include round-trip, upload, inference and response timing,
deadline misses, reconnect count, model hash, and Mac thermal-pressure state. A single
malformed, stale, disconnected, non-finite, or over-deadline response must latch the
Mac worker as failed when it is being considered as an active source.

## Important architectural difference

Chestnut is a PCIe GPU made visible through the USB enclosure and is driven inside the
same `modeld` process. The Mac prototype is a networked process with additional copy,
transport, scheduling, and clock-domain failure modes. It cannot be marked active
merely because it follows the Chestnut UI/state naming.

For now the Mac worker remains shadow-only. Live integration should run the small
model as the published source, send the same warped frames to the Mac, compare both
outputs, and exercise the fallback latch without changing vehicle actuation.
