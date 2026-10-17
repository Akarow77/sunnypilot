"""
GPU on LAN: the chestnut's big driving model, on a computer on the local network.

Speaks the KServe v2 / Open Inference Protocol over HTTP with the binary tensor data
extension, and the Triton sequence extension (sequence_id / sequence_start) for the
model's temporal state. The inputs are the driving model's non-state inputs (new_img, desire,
traffic_convention, action_t); the server keeps the state_* tensors, like Triton's implicit
sequence state.

modeld uses it the way it uses the chestnut: it connects while modeld starts (Big Model
Loading), then the big model drives every frame instead of the on-device one: modeld warps the
frame, sends it and waits. A slow server means dropped camera frames (Driving Model Lagging); an
error or a stall makes modeld fall back to the on-device model for the rest of the drive (Big
Model Failed).
"""
import http.client
import ipaddress
import json
import math
import random
import re
import socket
import time

import numpy as np

from openpilot.common.swaglog import cloudlog
from openpilot.system.hardware import usb_network

DEFAULT_PORT = 8000  # KServe / Triton HTTP default
USB_TARGET = "usb"  # server is whatever is on the other end of the USB Ethernet link (see usb_network)
MODEL_NAME = "driving_policy"
TIMEOUT_S = 3.0  # like the chestnut's HCQDEV_WAIT_TIMEOUT_MS
CONNECT_RETRY_S = 1.0
MAX_RESPONSE_BYTES = 1 << 20
MAX_HEADER_BYTES = 64 << 10
INPUTS = {"new_img": ("UINT8", 2 * 6 * 128 * 256), "desire": ("FP32", 8),
          "traffic_convention": ("FP32", 2), "action_t": ("FP32", 2)}
REQUIRED_SLICES = {"lane_lines", "lane_lines_prob", "road_edges", "meta", "desire_pred", "pose",
                   "wide_from_device_euler", "road_transform", "plan", "lead", "lead_prob", "desire_state"}


def parse_host(target: str) -> tuple[str, int]:
  target = target.strip()
  port = str(DEFAULT_PORT)
  if target.startswith('['):  # [v6]:port
    host, end, suffix = target[1:].partition(']')
    if not end or (suffix and not suffix.startswith(':')):
      raise ValueError("invalid bracketed IPv6 address")
    ipaddress.IPv6Address(host)
    if suffix:
      port = suffix[1:]
  elif target.count(':') == 1:
    host, port = target.split(':')
  else:
    host = target
    if ':' in host:
      ipaddress.IPv6Address(host)
  if not host or len(host) > 253 or (':' not in host and re.fullmatch(r"[A-Za-z0-9_.-]+", host) is None):
    raise ValueError("enter a host or IP address, not a URL")
  if not port.isascii() or not port.isdigit() or not 1 <= int(port) <= 65535:
    raise ValueError("port must be between 1 and 65535")
  return host, int(port)


def _tensor_size(shape) -> int:
  if not isinstance(shape, list) or not shape or len(shape) > 8 or any(type(n) is not int or n <= 0 for n in shape):
    raise ValueError("invalid tensor shape")
  size = math.prod(shape)
  if size > MAX_RESPONSE_BYTES:
    raise ValueError("tensor is too large")
  return size


def validate_metadata(meta: dict) -> tuple[dict[str, list[int]], dict[str, slice], int]:
  if meta.get("name") != MODEL_NAME or meta.get("parameters", {}).get("stateful") is not True:
    raise ValueError("expected a stateful driving_policy model")
  shapes = {}
  for tensor in meta["inputs"]:
    name = tensor["name"]
    if name not in INPUTS or name in shapes or (tensor["datatype"], _tensor_size(tensor["shape"])) != INPUTS[name]:
      raise ValueError("incompatible model inputs")
    shapes[name] = tensor["shape"]
  if set(shapes) != INPUTS.keys() or shapes["new_img"] != [2, 6, 128, 256]:
    raise ValueError("incompatible camera input layout")
  outputs = meta["outputs"]
  if len(outputs) != 1 or outputs[0]["name"] != "outputs" or outputs[0]["datatype"] != "FP32":
    raise ValueError("incompatible model outputs")
  size = _tensor_size(outputs[0]["shape"])
  if outputs[0]["shape"] != [1, size] or size * 4 > MAX_RESPONSE_BYTES - MAX_HEADER_BYTES:
    raise ValueError("invalid model output shape")
  raw_slices = json.loads(meta["parameters"]["output_slices"])
  if not isinstance(raw_slices, dict) or not REQUIRED_SLICES.issubset(raw_slices):
    raise ValueError("missing driving output slices")
  slices = {}
  for name, bounds in raw_slices.items():
    if not isinstance(bounds, list) or len(bounds) != 2 or any(type(n) is not int for n in bounds):
      raise ValueError("invalid output slice")
    lo, hi = bounds
    if not 0 <= lo < hi <= size:
      raise ValueError("output slice outside tensor")
    slices[name] = slice(lo, hi)
  ordered = sorted((s.start, s.stop) for s in slices.values())
  if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:], strict=False)):
    raise ValueError("overlapping output slices")
  return shapes, slices, size


def encode_infer_request(seq: int, sequence_id: int, start: bool, tensors: dict[str, np.ndarray]) -> tuple[bytes, int]:
  datatypes = {np.dtype(np.uint8): "UINT8", np.dtype(np.float32): "FP32"}
  header = {
    "id": str(seq),
    "parameters": {"sequence_id": sequence_id, "sequence_start": start, "sequence_end": False},
    "inputs": [{"name": k, "shape": list(v.shape), "datatype": datatypes[v.dtype],
                "parameters": {"binary_data_size": v.nbytes}} for k, v in tensors.items()],
    "outputs": [{"name": "outputs", "parameters": {"binary_data": True}}],
  }
  header_bytes = json.dumps(header, separators=(',', ':')).encode()
  # A single owned body snapshot: no per-tensor .tobytes(), payload join and
  # header concatenation copies. HTTP still owns stable bytes during the request.
  return b''.join([header_bytes, *(memoryview(np.ascontiguousarray(v)).cast('B') for v in tensors.values())]), len(header_bytes)


def decode_infer_response(body: bytes, header_len: int) -> tuple[dict, np.ndarray]:
  if not 0 < header_len <= min(len(body), MAX_HEADER_BYTES) or len(body) > MAX_RESPONSE_BYTES:
    raise ValueError("invalid response header length")
  header = json.loads(body[:header_len])
  if len(header["outputs"]) != 1:
    raise ValueError("expected one output tensor")
  out = header["outputs"][0]
  if out.get("name") != "outputs" or out.get("datatype") != "FP32":
    raise ValueError("expected FP32 outputs")
  size = out["parameters"]["binary_data_size"]
  if type(size) is not int or size <= 0 or size != _tensor_size(out["shape"]) * 4 or header_len + size != len(body):
    raise ValueError("output byte count does not match shape or body")
  if header.get("parameters", {}).get("non_finite", False):
    raise ValueError("server rejected non-finite model output")
  # a copy: the parser works in place, and frombuffer over bytes is read-only
  outputs = np.frombuffer(body, dtype='<f4', count=size // 4, offset=header_len).copy()
  if not np.all(np.isfinite(outputs)):
    raise ValueError("model output not finite")
  return header, outputs


def read_response(response: http.client.HTTPResponse) -> bytes:
  length = response.getheader("Content-Length")
  if length is not None and not 0 <= int(length) <= MAX_RESPONSE_BYTES:
    raise ValueError("response exceeds size limit")
  data = response.read(MAX_RESPONSE_BYTES + 1)
  if len(data) > MAX_RESPONSE_BYTES or (length is not None and len(data) != int(length)):
    raise ValueError("invalid response body size")
  if response.status != 200:
    raise ConnectionError(f"gpu on lan: HTTP {response.status}")
  return data


def connect_gpu_on_lan(target: str, timeout: float):
  """Invalid settings or an unavailable server must leave the local model usable."""
  client = None
  try:
    client = GpuOnLan(target)
    if client.connect(timeout):
      return client
  except Exception:
    cloudlog.exception("gpu on lan: connection setup failed")
  if client is not None:
    client.close()
  return None


class GpuOnLan:
  def __init__(self, target: str):
    host, self.port = parse_host(target.strip() or USB_TARGET)
    self.usb = host.lower() == USB_TARGET
    self.host: str | None = None if self.usb else host
    self.output_slices: dict[str, slice] | None = None
    self.model_name: str | None = None  # what the server says it runs, e.g. "Cinque Terre Model v2"
    self._input_shapes: dict[str, list[int]] = {}  # the server model's, from its metadata
    self._output_size = 0
    self._conn: http.client.HTTPConnection | None = None
    self._seq = 0
    self._sequence_id = random.getrandbits(63)

  def connect(self, timeout: float) -> bool:
    """Blocks until the server answers with its model (it loads and warms it up before it listens), or timeout."""
    deadline = time.monotonic() + timeout
    while True:
      try:
        self._conn = self._connect()
        return True
      except Exception as e:
        self.close()
        if time.monotonic() > deadline:
          cloudlog.exception(f"gpu on lan: no server at {'usb' if self.usb else self.host}:{self.port}: {e}")
          return False
        time.sleep(CONNECT_RETRY_S)

  def close(self) -> None:
    if self._conn is not None:
      self._conn.close()
      self._conn = None

  def infer(self, new_img: np.ndarray, desire: np.ndarray, traffic_convention: np.ndarray, action_t: np.ndarray) -> np.ndarray:
    """The big model's raw outputs for this frame (slice with output_slices). Raises if the server
    fails or stalls for TIMEOUT_S."""
    assert self._conn is not None
    if new_img.dtype != np.uint8 or list(new_img.shape) != self._input_shapes["new_img"]:
      raise ValueError("camera warp does not match remote model")
    tensors = {
      "new_img": new_img,
      "desire": desire.astype(np.float32, copy=False),
      "traffic_convention": traffic_convention.astype(np.float32, copy=False),
      "action_t": action_t.astype(np.float32, copy=False),
    }
    # a model bundle may lay out the same inputs differently, e.g. desire (1, 8) vs (8,)
    for name, shape in self._input_shapes.items():
      if name in tensors:
        tensors[name] = tensors[name].reshape(shape)
        if tensors[name].dtype.kind == 'f' and not np.all(np.isfinite(tensors[name])):
          raise ValueError("model input not finite")
    seq = self._seq
    body, header_len = encode_infer_request(self._seq, self._sequence_id, self._seq == 0, tensors)
    self._seq += 1
    started = time.monotonic()
    self._conn.request("POST", f"/v2/models/{MODEL_NAME}/infer", body=body, headers={
      "Content-Type": "application/octet-stream",
      "Inference-Header-Content-Length": str(header_len),
    })
    resp = self._conn.getresponse()
    data = read_response(resp)
    header, outputs = decode_infer_response(data, int(resp.getheader("Inference-Header-Content-Length")))
    if header.get("id") != str(seq) or header.get("model_name") != MODEL_NAME or outputs.size != self._output_size:
      raise ValueError("response does not match this request or model")
    if time.monotonic() - started > TIMEOUT_S:
      raise TimeoutError("gpu on lan: stale response")
    return outputs

  def _resolve(self) -> str:
    if self.host is not None and not self.usb:
      return self.host
    # the link itself is brought up from settings (UsbNetworkEnabled), not from modeld
    peer = usb_network.peer_address()
    if peer is None:
      raise ConnectionError("no laptop on the USB network")
    self.host = peer
    return peer

  def _connect(self) -> http.client.HTTPConnection:
    conn = http.client.HTTPConnection(self._resolve(), self.port, timeout=TIMEOUT_S)
    self._conn = conn  # also close connections whose metadata/handshake fails
    conn.request("GET", "/v2/health/ready")
    resp = conn.getresponse()
    read_response(resp)
    conn.request("GET", f"/v2/models/{MODEL_NAME}")
    meta = json.loads(read_response(conn.getresponse()))
    self._input_shapes, self.output_slices, self._output_size = validate_metadata(meta)
    self.model_name = meta["parameters"].get("model_name")
    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    cloudlog.warning(f"gpu on lan: connected to {self.host}:{self.port}, model {meta['parameters'].get('model_checkpoint')}")
    return conn
