import json
import copy
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from openpilot.selfdrive.modeld.gpu_on_lan import parse_host, encode_infer_request, decode_infer_response, DEFAULT_PORT
from openpilot.selfdrive.modeld import gpu_on_lan as lan


def test_parse_host():
  assert parse_host("192.168.43.100") == ("192.168.43.100", DEFAULT_PORT)
  assert parse_host(" mac.local:9000 ") == ("mac.local", 9000)
  assert parse_host("[fe80::1]:8001") == ("fe80::1", 8001)
  assert parse_host("[fe80::1]") == ("fe80::1", DEFAULT_PORT)


def test_request_layout():
  tensors = {"img": np.arange(12, dtype=np.uint8).reshape(1, 3, 4), "action_t": np.array([[0.25, 0.5]], dtype=np.float32)}
  body, header_len = encode_infer_request(5, 42, True, tensors)
  header = json.loads(body[:header_len])
  assert header["id"] == "5"
  assert header["parameters"] == {"sequence_id": 42, "sequence_start": True, "sequence_end": False}
  assert [(i["name"], i["datatype"], i["shape"], i["parameters"]["binary_data_size"]) for i in header["inputs"]] == \
    [("img", "UINT8", [1, 3, 4], 12), ("action_t", "FP32", [1, 2], 8)]
  assert body[header_len:header_len + 12] == bytes(range(12))
  assert np.frombuffer(body[header_len + 12:], dtype=np.float32).tolist() == [0.25, 0.5]


def test_request_is_owned_snapshot_of_noncontiguous_and_readonly_views():
  backing = np.arange(24, dtype=np.uint8).reshape(4, 6)
  image = backing[:, ::2]
  image.flags.writeable = False
  expected = image.tobytes()
  body, header_len = encode_infer_request(0, 1, True, {'img': image})
  backing[:] = 0
  assert body[header_len:] == expected


def test_decode_response():
  outputs = np.array([1., 2., 3.], dtype=np.float32)
  header = json.dumps({"model_name": "driving_policy", "parameters": {"sequence_warm": True},
                       "outputs": [{"name": "outputs", "shape": [1, 3], "datatype": "FP32",
                                    "parameters": {"binary_data_size": 12}}]}).encode()
  parsed_header, parsed = decode_infer_response(header + outputs.tobytes(), len(header))
  assert parsed_header["parameters"]["sequence_warm"]
  np.testing.assert_array_equal(parsed, outputs)



def test_usb_target():
  from openpilot.selfdrive.modeld.gpu_on_lan import GpuOnLan
  for target, usb, host, port in (("", True, None, DEFAULT_PORT), ("usb", True, None, DEFAULT_PORT),
                                  ("USB:9000", True, None, 9000), ("10.0.0.2:8001", False, "10.0.0.2", 8001)):
    c = GpuOnLan(target)
    assert (c.usb, c.host, c.port) == (usb, host, port), target


def test_decoded_outputs_writable():
  outputs = np.ones(2, dtype=np.float32)
  header = json.dumps({"outputs": [{"name": "outputs", "shape": [1, 2], "datatype": "FP32",
                                    "parameters": {"binary_data_size": outputs.nbytes}}]}).encode()
  _, decoded = decode_infer_response(header + outputs.tobytes(), len(header))
  decoded -= 1  # the parser's softmax does this


@pytest.mark.parametrize("target", ["", "http://mac:8000", "mac/path", "mac:", ":8000", "mac:0", "mac:65536", "[::1", "[::1]oops", "mac:-1"])
def test_bad_hosts(target):
  with pytest.raises(ValueError):
    parse_host(target)


def metadata():
  lengths = {"lane_lines": 528, "lane_lines_prob": 8, "road_edges": 264, "meta": 55, "desire_pred": 32,
             "pose": 12, "wide_from_device_euler": 6, "road_transform": 12, "plan": 990, "lead": 144,
             "lead_prob": 3, "desire_state": 8, "action": 4, "pad": 2}
  slices, offset = {}, 0
  for name, size in lengths.items():
    slices[name] = [offset, offset + size]
    offset += size
  shapes = {"new_img": [2, 6, 128, 256], "desire": [8], "traffic_convention": [1, 2], "action_t": [1, 2]}
  return {"name": lan.MODEL_NAME, "parameters": {"stateful": True, "output_slices": json.dumps(slices), "model_checkpoint": "test"},
          "inputs": [{"name": n, "datatype": lan.INPUTS[n][0], "shape": shape} for n, shape in shapes.items()],
          "outputs": [{"name": "outputs", "datatype": "FP32", "shape": [1, offset]}]}


def test_current_server_metadata_and_parsers():
  from openpilot.selfdrive.modeld.parse_model_outputs import Parser as StockParser
  from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser as SunnyParser
  _, slices, size = lan.validate_metadata(metadata())
  assert size == 2068  # hidden_state remains on the Mac
  for parser in (StockParser(), SunnyParser()):
    out = np.zeros(size, dtype=np.float32)
    result = parser.parse_outputs({k: out[np.newaxis, v] for k, v in slices.items()})
    assert result['plan'].shape == (1, 33, 15)
    assert result['pose'].shape == (1, 6)


@pytest.mark.parametrize("bad", ["input_shape", "input_dtype", "missing_input", "duplicate_input", "output_dtype", "slice_bounds",
                                 "slice_overlap", "missing_slice", "stateful"])
def test_incompatible_metadata(bad):
  meta = metadata()
  if bad == 'input_shape':
    meta['inputs'][0]['shape'] = [2, 6, 256, 128]
  elif bad == 'input_dtype':
    meta['inputs'][0]['datatype'] = 'FP32'
  elif bad == 'missing_input':
    meta['inputs'].pop()
  elif bad == 'duplicate_input':
    meta['inputs'].append(copy.deepcopy(meta['inputs'][0]))
  elif bad == 'output_dtype':
    meta['outputs'][0]['datatype'] = 'FP16'
  elif bad == 'stateful':
    meta['parameters']['stateful'] = False
  else:
    slices = json.loads(meta['parameters']['output_slices'])
    if bad == 'missing_slice':
      del slices['plan']
    else:
      slices['plan'] = [-1, 5] if bad == 'slice_bounds' else [0, 5]
    meta['parameters']['output_slices'] = json.dumps(slices)
  with pytest.raises(ValueError):
    lan.validate_metadata(meta)


def response(seq='0', mode='ok'):
  out = np.zeros(2068, dtype=np.float32)
  if mode == 'nan':
    out[0] = np.nan
  header = {"model_name": lan.MODEL_NAME, "id": seq if mode != 'wrong_id' else '999',
            "parameters": {"non_finite": mode == 'non_finite'},
            "outputs": [{"name": "outputs", "shape": [1, len(out)], "datatype": "FP32",
                         "parameters": {"binary_data_size": out.nbytes}}]}
  if mode == 'wrong_shape':
    header['outputs'][0]['shape'] = [1, 1]
  raw = json.dumps(header).encode()
  body = raw + out.tobytes()
  return (body[:-1] if mode == 'truncated' else body), len(raw)


@pytest.mark.parametrize('mode', ['nan', 'non_finite', 'wrong_shape', 'truncated'])
def test_invalid_output(mode):
  body, size = response(mode=mode)
  with pytest.raises(ValueError):
    decode_infer_response(body, size)


@pytest.mark.parametrize('header_len', [-1, 0, 1000000])
def test_header_limits(header_len):
  with pytest.raises(ValueError):
    decode_infer_response(b'{}', header_len)


@pytest.fixture
def server():
  class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *_):
      pass

    def send(self, data, status=200, header_len=None):
      self.send_response(status)
      self.send_header('Content-Length', str(len(data)))
      if header_len is not None:
        self.send_header('Inference-Header-Content-Length', str(header_len))
      self.end_headers()
      try:
        self.wfile.write(data)
      except BrokenPipeError:
        pass  # timeout tests deliberately close the client

    def do_GET(self):
      meta = metadata()
      if self.server.mode == 'bad_metadata':
        meta['parameters']['stateful'] = False
      self.send(b'' if self.path.endswith('/ready') else json.dumps(meta).encode())

    def do_POST(self):
      body = self.rfile.read(int(self.headers['Content-Length']))
      hdr = json.loads(body[:int(self.headers['Inference-Header-Content-Length'])])
      self.server.requests.append(hdr)
      if self.server.mode == 'stall':
        threading.Event().wait(.15)
      if self.server.mode == 'disconnect':
        self.close_connection = True
        return
      out, size = response(hdr['id'], self.server.mode)
      self.send(out, 503 if self.server.mode == 'http_error' else 200, size)

  srv = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
  srv.mode, srv.requests = 'ok', []
  thread = threading.Thread(target=srv.serve_forever, daemon=True)
  thread.start()
  try:
    yield srv
  finally:
    srv.shutdown()
    srv.server_close()
    thread.join(timeout=2)


def run_frame(client):
  return client.infer(np.zeros((2, 6, 128, 256), dtype=np.uint8), np.zeros(8, dtype=np.float32),
                      np.array([1, 0], dtype=np.float32), np.array([.1, .1], dtype=np.float32))


def test_http_roundtrip_and_state_sequence(server):
  client = lan.connect_gpu_on_lan(f'127.0.0.1:{server.server_port}', 0)
  assert client is not None
  try:
    for _ in range(2):
      assert run_frame(client).shape == (2068,)
    first, second = server.requests
    assert first['parameters']['sequence_start'] is True
    assert second['parameters']['sequence_start'] is False
    assert first['parameters']['sequence_id'] == second['parameters']['sequence_id']
    assert [first['id'], second['id']] == ['0', '1']
  finally:
    client.close()


@pytest.mark.parametrize('mode', ['wrong_id', 'nan', 'non_finite', 'wrong_shape', 'truncated', 'http_error', 'disconnect'])
def test_http_failures_do_not_return_predictions(server, mode):
  client = lan.connect_gpu_on_lan(f'127.0.0.1:{server.server_port}', 0)
  assert client is not None
  server.mode = mode
  try:
    with pytest.raises((ValueError, ConnectionError, OSError, lan.http.client.HTTPException)):
      run_frame(client)
  finally:
    client.close()


def test_bad_setting_keeps_local_path():
  assert lan.connect_gpu_on_lan('mac:not-a-port', 0) is None


def test_bad_metadata_closes_connection(server):
  server.mode = 'bad_metadata'
  client = lan.GpuOnLan(f'127.0.0.1:{server.server_port}')
  assert not client.connect(0)
  assert client._conn is None


def test_stall_raises_instead_of_reusing_output(server, monkeypatch):
  monkeypatch.setattr(lan, 'TIMEOUT_S', .03)
  client = lan.connect_gpu_on_lan(f'127.0.0.1:{server.server_port}', 0)
  assert client is not None
  server.mode = 'stall'
  try:
    with pytest.raises(TimeoutError):
      run_frame(client)
  finally:
    client.close()


def test_bad_camera_layout_rejected_before_upload(server):
  client = lan.connect_gpu_on_lan(f'127.0.0.1:{server.server_port}', 0)
  assert client is not None
  try:
    with pytest.raises(ValueError):
      client.infer(np.zeros((2, 6, 256, 128), dtype=np.uint8), np.zeros(8), np.zeros(2), np.zeros(2))
    assert not server.requests
  finally:
    client.close()
