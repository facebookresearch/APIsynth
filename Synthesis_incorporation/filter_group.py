# Copyright 2021 The TF-Coder Authors.
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
"""Defines the FilterGroup enum.

Every function used in TF-Coder is associated with one such FilterGroup.
"""

import enum


# LINT.IfChange(FilterGroup)
@enum.unique
class FilterGroup(enum.Enum):
    """A group of similar operations that should have the same filters.

    The number of arguments is crucially important when adding filters, so by
    convention the enum names have the number of arguments at the end.
    """

    # No filters. Even if some filtering might be reasonable, it could be faster
    # to just try all values to avoid the filtering overhead.
    NONE = "NONE"

    #############################
    # Operations with 1 argument.

    # The argument is a shape.
    SHAPE_1 = "SHAPE_1"
    # The argument is a tensor.
    TENSOR_1 = "TENSOR_1"
    # The argument is a sequence of tensors.
    TENSORSEQUENCE_1 = "TENSORSEQUENCE_1"
    # The argument is a float tensor.
    FLOATTENSOR_1 = "FLOATTENSOR_1"
    # The argument is an int or float tensor.
    NUMERICTENSOR_1 = "NUMERICTENSOR_1"
    # The argument is a primitive or tensor.
    PRIMITIVE_OR_TENSOR_1 = "PRIMITIVE_OR_TENSOR_1"

    ################################
    # Operations with 2 arguments.

    # The first argument is a tensor, and the second argument is an int
    # representing an axis, i.e., an int in the range [1, rank_of_tensor).
    TENSOR_AXIS_2 = "TENSOR_AXIS_2"
    # The first argument is an int or float tensor, the second is an axis.
    NUMERICTENSOR_AXIS_2 = "NUMERICTENSOR_AXIS_2"
    # The first argument is a sequence of tensors, the second is an axis.
    TENSORSEQUENCE_AXIS_2 = "TENSORSEQUENCE_AXIS_2"
    # The first argument is a tensor, the second is a boolean tensor.
    TENSOR_BOOLTENSOR_2 = "TENSOR_BOOLTENSOR_2"
    # The two arguments are numeric (int or float) tensors with the same shape.
    SAME_SHAPES_NUMERICTENSOR_2 = "SAME_SHAPES_NUMERICTENSOR_2"
    # The two arguments are numeric (int or float) tensors with the same dtype,
    # and the two tensors are broadcastable.
    SAME_DTYPE_NUMERIC_BROADCASTABLE_2 = "SAME_DTYPE_NUMERIC_BROADCASTABLE_2"
    # The first argument is a numeric tensor, and the second is either a scalar
    # or a tensor. The two arguments are broadcastable.
    ELEMENTWISE_COMPARISON_2 = "ELEMENTWISE_COMPARISON_2"
    # The first argument is a numeric tensor, and the second is either a scalar
    # or a tensor. The two arguments are broadcastable, but the must be different.
    NE_BROADCASTABLE_2 = "NE_BROADCASTABLE_2"

    #########################################
    # Operations with other special handling.

    # The argument contains nonnegative ints with a small maximum.
    BINCOUNT_1 = "BINCOUNT_1"
    # The argument results in a small tensor.
    EYE_1 = "EYE_1"
    # The argument results in a small tensor.
    RANGE_1 = "RANGE_1"
    # The argument is either a primitive or a sequence of primitives
    TENSORIZABLE_1 = "TENSORIZABLE_1"
    # The arguments should be 3-D tensors, and the first argument's
    # third dimension size should be equal to the second argument's
    # second dimension size.
    BMM_2 = "BMM_2"
    # The first argument is a sequence of tensors,
    # the second is an axis in the range [-1, rank_of_tensor-1].
    CAT_TENSORSEQUENCE_AXIS_2 = "CAT_TENSORSEQUENCE_AXIS_2"
    # Both arguments are tensors and have the same shape.
    # The dimensions should be greater than 1.
    CDIST_2 = "CDIST_2"
    # The first argument is a tensor, the second is an axis in the range
    # [-1, rank_of_tensor-1]. Note that this range is slightly different from the
    # TENSOR_AXIS_2 filter.
    EXPAND_DIMS_2 = "EXPAND_DIMS_2"
    # The first argument is a tensor, the second is an axis in the range
    # [-1, rank_of_tensor].
    EXPAND_DIMS_ADDITIONAL_2 = "EXPAND_DIMS_ADDITIONAL_2"
    # The arguments result in a small tensor.
    EYE_ROWS_COLS_2 = "EYE_ROWS_COLS_2"
    # Ensures the tensors are both numeric and have the same dtype and rank.
    MATMUL_2 = "MATMUL_2"
    # Ensures the tensors are both numeric and have the same dtype and rank.
    MM_2 = "MM_2"
    # The first argument is a tensor, the second is an axis in the range
    # [-1, rank_of_tensor-1]. The first argument must be float or int.
    NORMALIZE_2 = "NORMALIZE_2"
    # Ensures that torch.nn.functional.one_hot(indices, num_classes) produces a small result.
    ONE_HOT_2 = "ONE_HOT_2"
    # The first argument must be a tensor, and the second must be a nested int
    # list or int32 tensor of shape [rank_of_arg_1, 2].
    PAD_2 = "PAD_2"
    # The first argument is a tensor, and the second is a tuple.
    RESHAPE_2 = "RESHAPE_2"
    # Ensures that torch.tile(input, multiples) produces a small result.
    TILE_2 = "TILE_2"
    # The first argument is sorted in the last dimension, the second argument is
    # the same dtype and rank, and all dimension lengths match except the last.
    SEARCHSORTED_2 = "SEARCHSORTED_2"
    # The first argument is a tensor with more than 1 squeezable dimension, and
    # the second argument is an int specifying a squeezable dimension.
    SQUEEZE_2 = "SQUEEZE_2"

    # The first argument is a non-scalar tensor, the second is a dimension, and
    # the third is a tensor containing ints suitable for indexing into the first
    # tensor.
    GATHER_3 = "GATHER_3"
    # The first argument is a tensor, the second is a tensor containing ints
    # suitable for indexing into the first tensor on multiple dimensions, and the
    # third is a number of batch dimensions.
    INDEX_SELECT_3 = "INDEX_SELECT_3"
    # The arguments result in a small tensor.
    RANGE_3 = "RANGE_3"
    # The first argument is a tensor, the second argument is either a numeric tensor
    # or an integer, and the third argument is an int specifying the dimension.
    REPEAT_3 = "REPEAT_3"
    # The second and third arguments must be int primitives, lists of ints, or 1D
    # int tensors, and they must have the same shape.
    ROLL_3 = "ROLL_3"
    # The first two arguments are tensors with the same dtype, and the third
    # contains ints of the appropriate shape.
    TENSORDOT_3 = "TENSORDOT_3"
    # The first argument is a tensor, and the second and the third are dimensions
    # to transpose.
    TRANSPOSE_3 = "TRANSPOSE_3"
    # Ensures that the shapes and dtypes for torch.where(condition, tensor, tensor/number) match.
    WHERE_TENSOR_3 = "WHERE_TENSOR_3"
    # Ensures that the shapes and dtypes for torch.where(condition, number, tensor/number) match.
    WHERE_NUMERIC_3 = "WHERE_NUMERIC_3"


# LINT.ThenChange(value_search/operation_filtering.py:add_filters_to_function_operation)
