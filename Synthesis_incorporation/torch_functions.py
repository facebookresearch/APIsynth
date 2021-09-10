# Copyright 2021 The PyCoder Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Functions and arguments used in the PyCoder project."""

import ast
import collections

import torch
from tf_coder import filter_group


FilterGroup = filter_group.FilterGroup


FunctionInfo = collections.namedtuple(
    'FunctionInfo',
    ['name', 'filter_group', 'weight'])


# Weights for leaf nodes in the AST.

# Constants given by the user.
PROVIDED_CONSTANT_WEIGHT = 7

# Ubiquitous constants: 0, 1, -1.
COMMON_CONSTANT_WEIGHT = 8

# A torch.constant() wrapper around an input primitive.
PRIMITIVE_INPUT_AS_TENSOR_WEIGHT = 5

# Int constants meant to be axis values, chosen based on input tensor ranks.
AXIS_CONSTANT_WEIGHT = 14

# Int constants obtained from input/output tensor shapes.
SHAPE_CONSTANT_WEIGHT = 24

# Weight of constructing a tuple with the output shape.
OUTPUT_SHAPE_TUPLE_WEIGHT = 32

# Input variable nodes (in1, in2, etc.).
INPUT_VARIABLE_WEIGHT = 4

# DTypes with weights to add to the pool of constants.
CONSTANT_DTYPES_AND_WEIGHTS = collections.OrderedDict([
    (torch.int32, 8),
    (torch.float32, 8),
    (torch.bool, 8),
    (torch.int64, 16),
])

# Used in value search to convert primitive inputs (e.g., 3) into scalar tensors
# (e.g., torch.tensor(3)).
CONSTANT_OPERATION_NAME = 'torch.tensor(data)'
INT_OPERATION_NAME = 'IntOperation'
FLOAT_OPERATION_NAME = 'FloatOperation'
BOOL_OPERATION_NAME = 'BoolOperation'


# A list of FunctionInfo namedtuples, each describing one function usable by a
# program synthesizer. Each FunctionInfo's name contains the function name along
# with the names of the arguments for that function, in the order given in the
# function's signature. A function may appear multiple times with different
# lists of usable arguments. This list is ordered, so value search will try
# earlier functions before later ones.

# FunctionInfo name format: "torch.module.function(arg_1, arg_2, arg_3='value')"
# means call the function `torch.module.function` with varying inputs `arg_1` and
# `arg_2`, where `arg_3` is fixed and set to the literal constant `'value'`.
TORCH_FUNCTIONS = [
    # FunctionInfo(name='torch.abs(input)',
    #              filter_group=FilterGroup.NUMERICTENSOR_1,
    #              weight=40),
    FunctionInfo(name='torch.add(input, other)',
                 filter_group=FilterGroup.ELEMENTWISE_COMPARISON_2,
                 weight=28),
    # # FunctionInfo(name='torch.all(input)',
    # #              filter_group=FilterGroup.TENSOR_1,
    # #              weight=40),
    # # FunctionInfo(name='torch.all(input, dim)',
    # #              filter_group=FilterGroup.EXPAND_DIMS_2,
    # #              weight=40),
    FunctionInfo(name='torch.any(input)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=40),
    FunctionInfo(name='torch.any(input, dim)',
                 filter_group=FilterGroup.EXPAND_DIMS_2,
                 weight=40),
    FunctionInfo(name='torch.arange(end)',
                 filter_group=FilterGroup.RANGE_1,
                 weight=28),
    # FunctionInfo(name='torch.arange(start, end, step)',
    #              filter_group=FilterGroup.RANGE_3,
    #              weight=56),
    FunctionInfo(name='torch.argmax(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=32),
    FunctionInfo(name='torch.argmax(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=32),
    # # FunctionInfo(name='torch.argsort(input, dim, descending=True)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
    # #              weight=48),
    # # FunctionInfo(name='torch.argsort(input, dim, descending=False)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
    # #              weight=48),
    FunctionInfo(name='torch.bincount(input)',
                 filter_group=FilterGroup.BINCOUNT_1,
                 weight=40),
    # # FunctionInfo(name='torch.bmm(input, mat2)',
    # #              filter_group=FilterGroup.BMM_2,
    # #              weight=40),
    # # FunctionInfo(name='torch.cat(tensors, dim)',
    # #              filter_group=FilterGroup.CAT_TENSORSEQUENCE_AXIS_2,
    # #              weight=36),
    FunctionInfo(name='torch.cdist(x1, x2)',
                 filter_group=FilterGroup.CDIST_2,
                 weight=48),
    # # FunctionInfo(name='torch.cumsum(input, dim)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
    # #              weight=44),
    FunctionInfo(name='torch.div(input, other)',
                 filter_group=FilterGroup.NE_BROADCASTABLE_2,
                 weight=28),
    FunctionInfo(name='torch.eq(input, other)',
                 filter_group=FilterGroup.ELEMENTWISE_COMPARISON_2,
                 weight=24),
    FunctionInfo(name='torch.eye(n)',
                 filter_group=FilterGroup.EYE_1,
                 weight=40),
    # FunctionInfo(name='torch.eye(n, m)',
    #              filter_group=FilterGroup.EYE_ROWS_COLS_2,
    #              weight=60),
    # # FunctionInfo(name='torch.flatten(input)',
    # #              filter_group=FilterGroup.TENSOR_1,
    # #              weight=23),
    # # FunctionInfo(name='torch.flatten(input, start_dim)',
    # #              filter_group=FilterGroup.EXPAND_DIMS_2,
    # #              weight=23),
    # # FunctionInfo(name='torch.flatten(input, start_dim, end_dim)',
    # #              filter_group=FilterGroup.TRANSPOSE_3,
    # #              weight=23),
    FunctionInfo(name='torch.gather(input, dim, index)',
                 filter_group=FilterGroup.GATHER_3,
                 weight=48),
    # # FunctionInfo(name='torch.ge(input, other)',
    # #              filter_group=FilterGroup.ELEMENTWISE_COMPARISON_2,
    # #              weight=32),
    FunctionInfo(name='torch.gt(input, other)',
                 filter_group=FilterGroup.ELEMENTWISE_COMPARISON_2,
                 weight=24),
    # # FunctionInfo(name='torch.index_select(input, dim, index)',
    # #              filter_group=FilterGroup.INDEX_SELECT_3,
    # #              weight=24),
    # # FunctionInfo(name='torch.le(input, other)',
    # #              filter_group=FilterGroup.ELEMENTWISE_COMPARISON_2,
    # #              weight=32),
    FunctionInfo(name='torch.lt(input, other)',
                 filter_group=FilterGroup.ELEMENTWISE_COMPARISON_2,
                 weight=24),
    # # FunctionInfo(name='torch.logical_and(input, other)',
    # #              filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
    # #              weight=24),
    FunctionInfo(name='torch.masked_select(input, mask)',
                 filter_group=FilterGroup.TENSOR_BOOLTENSOR_2,
                 weight=28),
    FunctionInfo(name='torch.matmul(input, other)',
                 filter_group=FilterGroup.MATMUL_2,
                 weight=24),
    FunctionInfo(name='torch.max(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=24),
    # FunctionInfo(name='torch.max(input, dim)',
                #  filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                #  weight=24),
    # # FunctionInfo(name='torch.maximum(input, other)',
    # #              filter_group=FilterGroup.SAME_SHAPES_NUMERICTENSOR_2,
    # #              weight=24),
    # # FunctionInfo(name='torch.mean(input)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_1,
    # #              weight=40),
    # # FunctionInfo(name='torch.mean(input, dim)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
    # #              weight=40),
    # # FunctionInfo(name='torch.min(input)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_1,
    # #              weight=24),
    FunctionInfo(name='torch.minimum(input, other)',
                 filter_group=FilterGroup.SAME_SHAPES_NUMERICTENSOR_2,
                 weight=32),
    # # FunctionInfo(name='torch.mm(input, mat2)',
    # #              filter_group=FilterGroup.MM_2,
    # #              weight=32),
    FunctionInfo(name='torch.mul(input, other)',
                 filter_group=FilterGroup.ELEMENTWISE_COMPARISON_2,
                 weight=24),
    FunctionInfo(name='torch.ne(input, other)',
                 filter_group=FilterGroup.ELEMENTWISE_COMPARISON_2,
                 weight=24),
    # # FunctionInfo(name='torch.nonzero(input)',
    # #              filter_group=FilterGroup.TENSOR_1,
    # #              weight=24),
    # # FunctionInfo(name='torch.nn.functional.normalize(input, dim)',
    # #              filter_group=FilterGroup.NORMALIZE_2,
    # #              weight=48),
    FunctionInfo(name='torch.nn.functional.one_hot(input, num_classes)',
                 filter_group=FilterGroup.ONE_HOT_2,
                 weight=28),
    # # FunctionInfo(name='torch.nn.functional.pad(input, pad, mode="constant")',
    # #              filter_group=FilterGroup.PAD_2,
    # #              weight=40),
    # # FunctionInfo(name='torch.prod(input, dim)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
    # #              weight=52),
    FunctionInfo(name='torch.repeat_interleave(input, repeats, dim)',
                 filter_group=FilterGroup.REPEAT_3,
                 weight=48),
    FunctionInfo(name='torch.reshape(input, shape)',
                 filter_group=FilterGroup.RESHAPE_2,
                 weight=28),
    FunctionInfo(name='torch.roll(input, shifts, dims)',
                 filter_group=FilterGroup.ROLL_3,
                 weight=48),
    FunctionInfo(name='torch.searchsorted(sorted_sequence, input)',
                 filter_group=FilterGroup.SEARCHSORTED_2,
                 weight=56),
    # # FunctionInfo(name='torch.sort(input, dim)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
    # #              weight=52),
    # # FunctionInfo(name='torch.sort(input, dim, descending=True)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
    # #              weight=60),
    FunctionInfo(name='torch.squeeze(input)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=23),
    FunctionInfo(name='torch.squeeze(input, dim)',
                 filter_group=FilterGroup.SQUEEZE_2,
                 weight=23),
    # # FunctionInfo(name='torch.sqrt(input)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_1,
    # #              weight=56),
    FunctionInfo(name='torch.square(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=28),
    FunctionInfo(name='torch.stack(tensors)',
                 filter_group=FilterGroup.TENSORSEQUENCE_1,
                 weight=36),
    FunctionInfo(name='torch.stack(tensors, dim)',
                 filter_group=FilterGroup.TENSORSEQUENCE_AXIS_2,
                 weight=36),
    # # FunctionInfo(name='torch.std(input, dim)',
    # #              filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
    # #              weight=40),
    # # FunctionInfo(name='torch.sub(input, other)',
    # #              filter_group=FilterGroup.NE_BROADCASTABLE_2,
    # #              weight=28),
    FunctionInfo(name='torch.sum(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=24),
    FunctionInfo(name='torch.sum(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=24),
    # FunctionInfo(name=CONSTANT_OPERATION_NAME,
    #              filter_group=FilterGroup.TENSORIZABLE_1,
    #              weight=24),
    FunctionInfo(name='torch.tensordot(a, b, dims)',
                 filter_group=FilterGroup.TENSORDOT_3,
                 weight=24),
    FunctionInfo(name='torch.tile(input, dims)',
                 filter_group=FilterGroup.TILE_2,
                 weight=28),
    FunctionInfo(name='torch.transpose(input, dim0, dim1)',
                 filter_group=FilterGroup.TRANSPOSE_3,
                 weight=24),
    FunctionInfo(name='torch.where(condition, input, other)',
                 filter_group=FilterGroup.WHERE_TENSOR_3,
                 weight=24),
    FunctionInfo(name='torch.where(condition, self, other)',
                 filter_group=FilterGroup.WHERE_NUMERIC_3,
                 weight=24),
    # # FunctionInfo(name='torch.unique(input)',
    # #              filter_group=FilterGroup.TENSOR_1,
    # #              weight=48),
    FunctionInfo(name='torch.unsqueeze(input, dim)',
                 filter_group=FilterGroup.EXPAND_DIMS_ADDITIONAL_2,
                 weight=22),
    # # FunctionInfo(name='torch.zeros(size)',
    # #              filter_group=FilterGroup.SHAPE_1,
    # #              weight=40),
]

SPARSE_FUNCTIONS = [
]


def parse_function_info_name(function_info):
  """Takes a FunctionInfo and returns (function_name, list_of_args).

  Args:
    function_info: A FunctionInfo namedtuple.

  Returns:
    A tuple (function_name, list_of_args, constant_kwargs), where function_name
    is a string, list_of_args is a list of strings, and constant_kwargs is a
    dict mapping argument names to their constant literal values. For example,
    if the FunctionInfo's name is 'torch.foo.bar(x, axis, baz=True)', then
    this function would return ('torch.foo.bar', ['x', 'axis'], {'baz': True}).

  Raises:
    ValueError: If the FunctionInfo's name is not properly formatted.
  """
  name = function_info.name

  if name.count('(') != 1:
    raise ValueError("The FunctionInfo's name must have exactly one open "
                     "parenthesis.")
  if name.count(')') != 1 or name[-1] != ')':
    raise ValueError("The FunctionInfo's name must have exactly one close "
                     "parenthesis, at the end of the name.")

  open_paren = name.index('(')
  close_paren = name.index(')')
  function_name = name[ : open_paren]
  arg_list = name[open_paren + 1 : close_paren]
  split_by_comma = [arg.strip() for arg in arg_list.split(',')]
  list_of_args = []
  constant_kwargs = collections.OrderedDict()
  for part in split_by_comma:
    if '=' in part:
      kwarg_name, literal_as_string = [x.strip() for x in part.split('=')]
      constant_kwargs[kwarg_name] = ast.literal_eval(literal_as_string)
    else:
      list_of_args.append(part)
  return function_name, list_of_args, constant_kwargs
