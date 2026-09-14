#!/usr/bin/env python3
"""Experimental ONNX -> PyTorch -> Core ML conversion for Apple Neural Engine tests."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import AttributeProto, TensorProto, numpy_helper
import torch
from torch import nn
import torch.nn.functional as functional

from onnx2torch import convert
from onnx2torch.node_converters.cast import OnnxCast
from onnx2torch.node_converters.registry import add_converter
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.node_converters.split import OnnxSplit13
from onnx2torch.utils.common import OperationConverterResult, onnx_mapping_from_node


class OnnxGelu(nn.Module):
  def __init__(self, approximate: str):
    super().__init__()
    self.approximate = approximate

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    return functional.gelu(value, approximate=self.approximate)


class OnnxReduceMean18(nn.Module):
  def __init__(self, keepdims: int, noop_with_empty_axes: int):
    super().__init__()
    self.keepdims = bool(keepdims)
    self.noop_with_empty_axes = bool(noop_with_empty_axes)

  def forward(self, value: torch.Tensor, axes: torch.Tensor | None = None) -> torch.Tensor:
    if axes is not None and axes.numel() == 0 and self.noop_with_empty_axes:
      return value
    dims = tuple(range(value.ndim)) if axes is None or axes.numel() == 0 else tuple(int(x) for x in axes)
    return torch.mean(value, dim=dims, keepdim=self.keepdims)


@add_converter(operation_type='Cast', version=19)
def convert_cast(node, graph):
  del graph
  return OperationConverterResult(
    torch_module=OnnxCast(node.attributes.get('to')),
    onnx_mapping=onnx_mapping_from_node(node=node),
  )


@add_converter(operation_type='Gelu', version=20)
def convert_gelu(node, graph):
  del graph
  approximate = node.attributes.get('approximate', 'none')
  if isinstance(approximate, bytes):
    approximate = approximate.decode()
  return OperationConverterResult(
    torch_module=OnnxGelu('tanh' if approximate == 'tanh' else 'none'),
    onnx_mapping=onnx_mapping_from_node(node=node),
  )


@add_converter(operation_type='Identity', version=19)
def convert_identity(node, graph):
  del graph
  return OperationConverterResult(
    torch_module=nn.Identity(),
    onnx_mapping=onnx_mapping_from_node(node=node),
  )


@add_converter(operation_type='ReduceMean', version=18)
def convert_reduce_mean(node, graph):
  del graph
  return OperationConverterResult(
    torch_module=OnnxReduceMean18(node.attributes.get('keepdims', 1),
                                  node.attributes.get('noop_with_empty_axes', 0)),
    onnx_mapping=onnx_mapping_from_node(node=node),
  )


@add_converter(operation_type='Reshape', version=19)
def convert_reshape(node, graph):
  del graph
  return OperationConverterResult(
    torch_module=OnnxReshape(),
    onnx_mapping=onnx_mapping_from_node(node=node),
  )


@add_converter(operation_type='Split', version=18)
def convert_split(node, graph):
  del graph
  return OperationConverterResult(
    torch_module=OnnxSplit13(axis=node.attributes.get('axis', 0), num_splits=len(node.output_values)),
    onnx_mapping=onnx_mapping_from_node(node=node),
  )


class FloatImageWrapper(nn.Module):
  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model

  def forward(self, img, big_img, desire_pulse, traffic_convention, action_t, features_buffer):
    return self.model(img, big_img, desire_pulse, traffic_convention, action_t, features_buffer)


class CoreMLInputWrapper(nn.Module):
  def __init__(self, model: nn.Module):
    super().__init__()
    self.model = model

  def forward(self, img, big_img, desire_pulse, traffic_convention, action_t, features_buffer):
    # Core ML tensor inputs do not support uint8. The ONNX graph immediately casts
    # both image tensors to FP16, so accepting integral-valued FP16 pixels here is
    # numerically identical and avoids an unsupported FP16 -> uint8 -> FP16 path.
    return self.model(img, big_img, desire_pulse, traffic_convention, action_t, features_buffer)


def promote_fp16_graph(graph: onnx.GraphProto) -> None:
  """Promote FP16 storage and declared values so Core ML sees consistent conv types."""
  for value in (*graph.input, *graph.output, *graph.value_info):
    tensor_type = value.type.tensor_type
    if tensor_type.elem_type == TensorProto.FLOAT16:
      tensor_type.elem_type = TensorProto.FLOAT
  for index, initializer in enumerate(graph.initializer):
    if initializer.data_type == TensorProto.FLOAT16:
      promoted = numpy_helper.from_array(numpy_helper.to_array(initializer).astype(np.float32), initializer.name)
      graph.initializer[index].CopyFrom(promoted)
  for node in graph.node:
    if node.op_type == 'Cast':
      for attribute in node.attribute:
        if attribute.name == 'to' and attribute.i == TensorProto.FLOAT16:
          attribute.i = TensorProto.FLOAT
    for attribute in node.attribute:
      if attribute.type == AttributeProto.TENSOR and attribute.t.data_type == TensorProto.FLOAT16:
        promoted = numpy_helper.from_array(numpy_helper.to_array(attribute.t).astype(np.float32), attribute.t.name)
        attribute.t.CopyFrom(promoted)
      elif attribute.type == AttributeProto.GRAPH:
        promote_fp16_graph(attribute.g)
      elif attribute.type == AttributeProto.GRAPHS:
        for child_graph in attribute.graphs:
          promote_fp16_graph(child_graph)


def main() -> None:
  parser = argparse.ArgumentParser(description='Convert a sunnypilot ONNX model to Core ML')
  source = parser.add_mutually_exclusive_group(required=True)
  source.add_argument('--onnx', type=Path)
  source.add_argument('--traced-input', type=Path)
  parser.add_argument('--output', type=Path, required=True)
  parser.add_argument('--torch-only', action='store_true')
  parser.add_argument('--promote-fp32', action='store_true',
                      help='promote the ONNX graph to FP32 before Core ML FLOAT16 lowering')
  args = parser.parse_args()
  if args.promote_fp32 and args.traced_input is not None:
    parser.error('--promote-fp32 requires --onnx')

  started = time.monotonic()
  float_dtype = torch.float32 if args.promote_fp32 else torch.float16
  inputs = (
    torch.zeros((1, 12, 128, 256), dtype=torch.uint8),
    torch.zeros((1, 12, 128, 256), dtype=torch.uint8),
    torch.zeros((1, 33, 8), dtype=float_dtype),
    torch.tensor([[1.0, 0.0]], dtype=float_dtype),
    torch.tensor([[0.1, 0.1]], dtype=float_dtype),
    torch.zeros((1, 32, 32, 512), dtype=float_dtype),
  )
  if args.traced_input is not None:
    traced = torch.jit.load(args.traced_input).eval()
    with torch.inference_mode():
      traced_output = traced(*inputs)
    if not torch.isfinite(traced_output).all() or traced_output.ndim != 2 or traced_output.shape[0] != 1:
      raise RuntimeError(f'invalid loaded trace output: {tuple(traced_output.shape)}')
    print(f'Loaded and validated PyTorch trace: {time.monotonic() - started:.1f}s')
  else:
    model_proto = onnx.load(args.onnx)
    if args.promote_fp32:
      promote_fp16_graph(model_proto.graph)
    for node in model_proto.graph.node:
      if node.domain == 'org.tinygrad' and node.op_type == 'Contiguous':
        node.domain = ''
        node.op_type = 'Identity'
    onnx.checker.check_model(model_proto)
    model = FloatImageWrapper(convert(model_proto).eval()).eval()
    with torch.inference_mode():
      reference = model(*inputs)
    if not torch.isfinite(reference).all() or reference.ndim != 2 or reference.shape[0] != 1:
      raise RuntimeError(f'invalid converted PyTorch output: {tuple(reference.shape)}')
    traced = torch.jit.trace(model, inputs, strict=True)
    with torch.inference_mode():
      traced_output = traced(*inputs)
    if not torch.equal(reference, traced_output):
      raise RuntimeError('traced PyTorch output differs from eager output')
    print(f'PyTorch conversion and exact trace validation: {time.monotonic() - started:.1f}s')
  if args.torch_only:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, args.output)
    return

  import coremltools as ct
  traced = traced.float() if args.promote_fp32 else traced.half()
  with torch.inference_mode():
    half_output = traced(*inputs)
  if not torch.equal(traced_output, half_output):
    error = torch.max(torch.abs(traced_output.float() - half_output.float())).item()
    raise RuntimeError(f'FP16 normalization changed traced output: max_abs={error}')
  coreml_dtype = torch.float32 if args.promote_fp32 else torch.float16
  coreml_numpy_dtype = np.float32 if args.promote_fp32 else np.float16
  coreml_trace_inputs = (inputs[0].to(coreml_dtype), inputs[1].to(coreml_dtype), *inputs[2:])
  traced = torch.jit.trace(CoreMLInputWrapper(traced).eval(), coreml_trace_inputs, strict=True)
  with torch.inference_mode():
    coreml_trace_output = traced(*coreml_trace_inputs)
  if not torch.equal(traced_output, coreml_trace_output):
    raise RuntimeError('Core ML input wrapper changed model output')
  print(f'Core ML input trace exact validation: {time.monotonic() - started:.1f}s')
  coreml_inputs = [
    ct.TensorType(name='img', shape=inputs[0].shape, dtype=coreml_numpy_dtype),
    ct.TensorType(name='big_img', shape=inputs[1].shape, dtype=coreml_numpy_dtype),
    ct.TensorType(name='desire_pulse', shape=inputs[2].shape, dtype=coreml_numpy_dtype),
    ct.TensorType(name='traffic_convention', shape=inputs[3].shape, dtype=coreml_numpy_dtype),
    ct.TensorType(name='action_t', shape=inputs[4].shape, dtype=coreml_numpy_dtype),
    ct.TensorType(name='features_buffer', shape=inputs[5].shape, dtype=coreml_numpy_dtype),
  ]
  coreml_model = ct.convert(
    traced,
    inputs=coreml_inputs,
    convert_to='mlprogram',
    compute_precision=ct.precision.FLOAT16,
    minimum_deployment_target=ct.target.macOS15,
  )
  args.output.parent.mkdir(parents=True, exist_ok=True)
  coreml_model.save(args.output)
  print(f'Core ML package: {args.output} ({time.monotonic() - started:.1f}s total)')


if __name__ == '__main__':
  main()
