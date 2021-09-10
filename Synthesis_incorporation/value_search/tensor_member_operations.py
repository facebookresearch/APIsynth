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
"""Defines Operation objects for Python operators."""

import torch
from tf_coder.value_search import operation_base
from tf_coder.value_search import operation_filtering as filtering
from tf_coder.value_search import value

# Weights for Python operations.
SIZE_WEIGHT = 32
INT_WEIGHT = 16
FLOAT_WEIGHT = 16
BOOL_WEIGHT = 16
VIEW_WEIGHT = 28
EXPAND_WEIGHT = 24

# "Docstrings" for Python operations, so they can used for ranking in the same
# way as for TensorFlow operations.

SIZE_DOCSTRING = """
Returns the size of the self tensor. The returned value is a subclass of tuple.
"""
INT_DOCSTRING = """
Cast the self tensor to int.
"""
FLOAT_DOCSTRING = """
Cast the self tensor to float.
"""
BOOL_DOCSTRING = """
Cast the self tensor to bool.
"""
VIEW_DOCSTRINGS = """
Returns a new tensor with the same data as the self tensor but of a different shape.
"""
EXPAND_DOCSTRINGS = """
Returns a new view of the self tensor with singleton dimensions expanded to a larger size.
"""

class SizeOperation(operation_base.Operation):

    def __init__(self):
        metadata = operation_base.OperationMetadata(docstring=SIZE_DOCSTRING)
        super(SizeOperation, self).__init__(
            num_args=1, weight=SIZE_WEIGHT, metadata=metadata)

        self.add_value_filters([filtering.NON_SCALAR_TENSOR_FILTER])

    def apply(self, arg_values, settings):
        """See base class."""
        try:
            return value.OperationValue(arg_values[0].value.size(),
                                        self, arg_values)
        except Exception:  # pylint: disable=broad-except
            return None

    def reconstruct_expression_from_strings(self, arg_strings):
        """See base class."""
        if len(arg_strings) == 1:
            return arg_strings[0] + '.size()'
        else:
            return arg_strings[0] + '.size(' + arg_strings[1] + ')'


class IntOperation(operation_base.Operation):

    def __init__(self):
        metadata = operation_base.OperationMetadata(docstring=INT_DOCSTRING)
        super(IntOperation, self).__init__(
            num_args=1, weight=INT_WEIGHT, metadata=metadata)

        def _non_int_tensor_filter(arg_value):
            """Only keeps values that are non-int tensors."""
            return arg_value.is_tensor and not arg_value.has_int_dtype()

        self.add_value_filters([_non_int_tensor_filter])


    def apply(self, arg_values, settings):
        """See base class."""
        try:
            return value.OperationValue(arg_values[0].value.long(),
                                        self, arg_values)
        except Exception:  # pylint: disable=broad-except
            return None

    def reconstruct_expression_from_strings(self, arg_strings):
        """See base class."""
        return arg_strings[0] + '.long()'

class FloatOperation(operation_base.Operation):

    def __init__(self):
        metadata = operation_base.OperationMetadata(docstring=FLOAT_DOCSTRING)
        super(FloatOperation, self).__init__(
            num_args=1, weight=FLOAT_WEIGHT, metadata=metadata)

        def _non_float_tensor_filter(arg_value):
            """Only keeps values that are non-float tensors."""
            return arg_value.is_tensor and not arg_value.has_float_dtype()
        self.add_value_filters([_non_float_tensor_filter])


    def apply(self, arg_values, settings):
        """See base class."""
        try:
            return value.OperationValue(arg_values[0].value.float(),
                                        self, arg_values)
        except Exception:  # pylint: disable=broad-except
            return None

    def reconstruct_expression_from_strings(self, arg_strings):
        """See base class."""
        return arg_strings[0] + '.float()'

class BoolOperation(operation_base.Operation):

    def __init__(self):
        metadata = operation_base.OperationMetadata(docstring=BOOL_DOCSTRING)
        super(BoolOperation, self).__init__(
            num_args=1, weight=BOOL_WEIGHT, metadata=metadata)

        def _non_bool_tensor_filter(arg_value):
            """Only keeps values that are non-bool tensors."""
            return arg_value.is_tensor and not arg_value.has_bool_dtype()
        self.add_value_filters([_non_bool_tensor_filter])


    def apply(self, arg_values, settings):
        """See base class."""
        try:
            return value.OperationValue(arg_values[0].value.bool(),
                                        self, arg_values)
        except Exception:  # pylint: disable=broad-except
            return None

    def reconstruct_expression_from_strings(self, arg_strings):
        """See base class."""
        return arg_strings[0] + '.bool()'

class ViewOperation(operation_base.Operation):

    def __init__(self):
        metadata = operation_base.OperationMetadata(docstring=VIEW_DOCSTRINGS)
        super(ViewOperation, self).__init__(
            num_args=2, weight=VIEW_WEIGHT, metadata=metadata)

        def _size_compatable_filter(arg_values):
            """The new size must be compatible with its original size."""
            in1, in2 = arg_values
            return torch.prod(torch.tensor(in1.value.shape)) % torch.prod(torch.abs(torch.tensor(in2.value))) == 0
        self.add_value_filters([filtering.TENSOR_FILTER, filtering.SHAPE_FILTER])
        self.set_apply_filter(_size_compatable_filter)


    def apply(self, arg_values, settings):
        """See base class."""
        try:
            return value.OperationValue(arg_values[0].value.view(arg_values[1].value),
                                        self, arg_values)
        except Exception:  # pylint: disable=broad-except
            return None

    def reconstruct_expression_from_strings(self, arg_strings):
        """See base class."""
        return arg_strings[0] + '.view(' + arg_strings[1] + ')'


class ExpandOperation(operation_base.Operation):

    def __init__(self):
        metadata = operation_base.OperationMetadata(docstring=EXPAND_DOCSTRINGS)
        super(ExpandOperation, self).__init__(
            num_args=2, weight=EXPAND_WEIGHT, metadata=metadata)

        def _size_compatable_filter(arg_values):
            """The new size must be compatible with its original size."""
            in1, in2 = arg_values
            in1_dims_len = len(in1.value.shape)
            in2_dims_len = len(in2.value)
            if in1_dims_len > in2_dims_len:
                return False
            for i in range(in1_dims_len, in2_dims_len):
                if (in2.value[i] == -1
                ):
                    return False
            return True
        self.add_value_filters([filtering.TENSOR_FILTER, filtering.SHAPE_FILTER])
        self.set_apply_filter(_size_compatable_filter)


    def apply(self, arg_values, settings):
        """See base class."""
        try:
            return value.OperationValue(arg_values[0].value.expand(arg_values[1].value),
                                        self, arg_values)
        except Exception:  # pylint: disable=broad-except
            return None

    def reconstruct_expression_from_strings(self, arg_strings):
        """See base class."""
        return arg_strings[0] + '.expand(' + arg_strings[1] + ')'
