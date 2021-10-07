# Copyright 2021 The TF-Coder Authors.
# Copyright (c) Facebook, Inc. and its affiliates.
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
"""Exhaustive value search (enumerating by weight of expression)."""

import collections
import keyword
import re
import sys
import timeit
import tokenize
from typing import Any, Dict, List, NamedTuple, Optional, Set, Text, Tuple, Union
import random
from itertools import product

import numpy as np
import six
import torch
from absl import logging
from tf_coder import torch_functions
from tf_coder.benchmarks import benchmark as benchmark_module
from tf_coder.natural_language import description_handler as description_handler_module
from tf_coder.models import prediction_model
from tf_coder.repair import snippet_handler as snippet_handler_module
from tf_coder.value_search import all_operations
from tf_coder.value_search import filtered_values_cache
from tf_coder.value_search import operation_base
from tf_coder.value_search import operation_filtering
from tf_coder.value_search import operation_statistics
from tf_coder.value_search import value as value_module
from tf_coder.value_search import value_search_settings as settings_module

from tf_coder.natural_language import description_handler_factory
from tf_coder.models import prediction_model_factory
from tf_coder.repair import snippet_handler_factory

ValuesByWeight = operation_base.ValuesByWeightDict
DescriptionHandler = description_handler_module.DescriptionHandler
PredictionModel = prediction_model.ClassificationModel
SnippetHandler = snippet_handler_module.SnippetHandler

Solution = NamedTuple(
    "Solution",
    [
        ("value", value_module.Value),
        ("expression", Text),
        ("weight", int),
        ("time", float),
    ],
)

ValueSearchResults = NamedTuple(
    "ValueSearchResults",
    [
        ("solutions", List[Solution]),
        ("total_time", float),
        ("value_set", Set[value_module.Value]),
        ("values_by_weight", ValuesByWeight),
        ("benchmark", benchmark_module.Benchmark),
        ("settings", settings_module.Settings),
        ("statistics", Optional[operation_statistics.OperationStatistics]),
    ],
)


def _suppress_warnings() -> None:
    """Suppress TensorFlow and Numpy warnings."""
    # TensorFlow will produce tons of error logging because we often apply
    # TensorFlow operations with bad arguments. Suppressing logging noticeably
    # improves performance.
    logging.set_verbosity(logging.ERROR)

    # Numpy sometimes produces warnings for overflow, etc., which can be
    # distracting.
    np.seterr(all="ignore")


def _user_inputs(inputs: Union[Dict[Text, Any], List[Any]]) -> List[Any]:
    """Takes the inputs dict or list and extracts the input tensors."""
    if isinstance(inputs, list):
        return inputs
    elif isinstance(inputs, dict):
        return list(inputs.values())
    elif isinstance(inputs, tuple):
        return list(inputs)
    else:
        raise ValueError(
            "inputs must be a list or dict, but is {}".format(type(inputs))
        )


def _contains_sparse(benchmark: benchmark_module.Benchmark) -> bool:
    """Returns whether the benchmark involves SparseTensors."""
    # TODO(kshi): These heuristics are okay, but we should let the user choose if
    # they want to.
    for example in benchmark.examples:
        if isinstance(example.output, torch.Tensor):
            if example.output.is_sparse:
                return True
        for input_object in _user_inputs(example.inputs):
            if isinstance(input_object, torch.Tensor):
                if input_object.is_sparse:
                    return True
    return "sparse" in benchmark.description.lower()


def _add_value_by_weight(
    values_by_weight: ValuesByWeight, value: value_module.Value, weight: int
) -> None:
    """Adds a value of a given weight to values_by_weight."""
    if weight < len(values_by_weight):
        values_by_weight[weight][value] = value


def _constant_exists(constant: Any, constants_so_far: Set[Any]) -> bool:
    """Checks whether a constant exists already."""
    # We can't use the `in` keyword because `True in [1, 2, 3]` evaluates to True!
    # (`True == 1` evaluates to True.)
    return any(
        constant == existing and type(constant) is type(existing)
        for existing in constants_so_far
    )


def _is_valid_name(name: Text) -> bool:
    """Returns whether name is an acceptable Python identifier."""
    # Behavior is slightly different between Python versions, e.g., `await` is a
    # keyword only in PY3, and `print` is keyword only in PY2.
    if name in ["torch", "np"] or keyword.iskeyword(name):
        return False
    if six.PY3:
        return name.isidentifier()
    else:
        return bool(re.match(tokenize.Name + "$", name)) and name not in [
            "True",
            "False",
            "None",
        ]


def _input_names_to_objects(
    inputs_collection: Union[List[Any], Dict[Text, Any]]
) -> Dict[Text, Any]:
    """Returns a mapping from input names to objects, also validating names."""
    if isinstance(inputs_collection, (list, tuple)):
        input_names_to_objects = collections.OrderedDict(
            ("in" + str(i + 1), input_object)
            for i, input_object in enumerate(inputs_collection)
        )
    elif isinstance(inputs_collection, dict):
        for name in inputs_collection:
            if not isinstance(name, six.string_types):
                raise ValueError("The input name {!r} must be a string.".format(name))
            if not _is_valid_name(name):
                raise ValueError(
                    "The input name {!r} is not a valid Python identifier.".format(name)
                )
        input_names_to_objects = inputs_collection
    else:
        raise ValueError(
            "The collection of inputs has the wrong format. It can be "
            "a list of input objects, or a dict mapping string names "
            "to input objects."
        )
    return input_names_to_objects


def _add_constants_and_inputs_and_print(
    values_by_weight: ValuesByWeight,
    benchmark: benchmark_module.Benchmark,
    output_value: value_module.OutputValue,
    constant_operation: operation_base.Operation,
    settings: settings_module.Settings,
    multipliers: Optional[Dict[Text, float]] = None
) -> None:
    """Adds constant/input Values to values_by_weight, and prints to stdout."""
    # Conceptually this is a set, but it's actually a list so that constants can
    # be printed in the same order they are chosen by the heuristics. The reduced
    # efficiency of membership-checking is not a big deal because we have few
    # constants.
    constants_so_far = set()
    constants_to_print = []

    # User-provided constants.
    for c in benchmark.constants:
        if not _constant_exists(c, constants_so_far):
            constant_value = value_module.ConstantValue(c)
            weight = torch_functions.PROVIDED_CONSTANT_WEIGHT
            if multipliers:
                weight = max(1, int(round(weight * multipliers.get(str(constant_value.value), 1))))
            _add_value_by_weight(values_by_weight, constant_value, weight)
            constants_so_far.add(c)
            constants_to_print.append(c)

    # Add inputs, while computing some info for extra constants later.
    max_input_tensor_rank = 0
    dimension_lengths = set()
    input_names_to_objects = _input_names_to_objects(benchmark.examples[0].inputs)
    for name, input_object in input_names_to_objects.items():
        input_value = value_module.InputValue(input_object, name)
        if input_value.is_tensor:
            max_input_tensor_rank = max(max_input_tensor_rank, len(input_value.shape))
            dimension_lengths.update(input_value.shape)
        if input_value.is_primitive and constant_operation is not None:
            scalar_tensor_value = constant_operation.apply([input_value], settings)
            weight = torch_functions.PRIMITIVE_INPUT_AS_TENSOR_WEIGHT
            if multipliers:
                weight = max(1, int(round(weight * multipliers.get(name, 1))))
            _add_value_by_weight(
                values_by_weight,
                scalar_tensor_value,
                weight,
            )

        weight = torch_functions.INPUT_VARIABLE_WEIGHT
        if multipliers:
            weight = max(1, int(round(weight * multipliers.get(name, 1))))
        _add_value_by_weight(
            values_by_weight, input_value, weight
        )
        if input_value.is_primitive:
            constants_so_far.add(input_value.value)
            constants_to_print.append(input_value.value)
        if settings.printing.print_examples:
            print(
                "Input '{}'-{}:\n{!s}\n".format(name, input_value.type, input_value.value)
            )

    if output_value.shape is not None:
        dimension_lengths.update(output_value.shape)

    # Always include these as constants.
    common_constants = [0, 1, -1]
    # common_constants = [0, 1, -1, True, False]
    # Also include 2, 3, ..., max_example_input_tensor_rank - 1 when applicable.
    axis_constants = list(range(2, max_input_tensor_rank))
    # Also include dimension lengths of input and output tensors.
    shape_constants = sorted(dimension_lengths)

    constant_weight_pairs = (
        [(c, torch_functions.COMMON_CONSTANT_WEIGHT) for c in common_constants]
        + [(c, torch_functions.AXIS_CONSTANT_WEIGHT) for c in axis_constants]
        + [(c, torch_functions.SHAPE_CONSTANT_WEIGHT) for c in shape_constants]
    )

    for constant, weight in constant_weight_pairs:
        if not _constant_exists(constant, constants_so_far):
            constant_value = value_module.ConstantValue(constant)
            if multipliers:
                weight = max(1, int(round(weight * multipliers.get(str(constant_value.value), 1))))
            _add_value_by_weight(values_by_weight, constant_value, weight)
            constants_so_far.add(constant)
            constants_to_print.append(constant)


    if output_value.shape:
        # Add the output shape as a constant.
        shape_tuple = tuple(output_value.shape)
        shape_tuple_value = value_module.ConstantValue(shape_tuple)
        weight = torch_functions.OUTPUT_SHAPE_TUPLE_WEIGHT
        if multipliers:
            weight = max(1, int(round(weight * multipliers.get(str(shape_tuple_value.value), 1))))
        _add_value_by_weight(values_by_weight, shape_tuple_value, weight)
        # Don't add shape_tuple to constants_to_print, because printing it out could
        # be confusing to users.

    # Only for experiments in the PLDI paper.
    if settings.paper_experiments.uniform_weights:
        # Count the number of values.
        num_values = sum(
            len(values_with_weight) for values_with_weight in values_by_weight
        )
        # Take all values and put them in the collection for weight 1.
        for weight in range(2, len(values_by_weight)):
            for heavy_value in values_by_weight[weight]:
                values_by_weight[1][heavy_value] = heavy_value
            values_by_weight[weight].clear()
        # Make sure we did it right.
        for weight, values_with_weight in enumerate(values_by_weight):
            assert len(values_with_weight) == (num_values if weight == 1 else 0)
    if settings.printing.print_examples:
        print("Output-{}:\n{!s}\n".format(output_value.type, output_value.value))
        print("Constants: {!r}\n".format(constants_to_print))
        if benchmark.snippet:
            print("Original snippet: {!r}\n".format(benchmark.snippet))
        if benchmark.target_program:
            print("Target snippet: {!r}\n".format(benchmark.target_program))
        if benchmark.description:
            print("Description: {}\n".format(benchmark.description))
        print("Searching...\n")
    sys.stdout.flush()  # Flush so the inputs/output appear in Colab immediately.


def _check_solution(
    expression: Text,
    used_input_names: Set[Text],
    benchmark: benchmark_module.Benchmark,
    settings: settings_module.Settings,
) -> bool:
    """Checks that the solution is good."""
    del expression  # Unused for now.

    if settings.require_all_inputs_used:
        if len(used_input_names) < len(benchmark.examples[0].inputs):
            return False
    elif settings.require_one_input_used:
        if not used_input_names:
            return False

    # TODO(kshi): Check that the solution works (floating-point errors may
    # accumulate beyond an acceptable threshold).
    return True


def _record_solutions(
    value: value_module.Value,
    weight: int,
    start_time: float,
    solutions: List[Solution],
    solution_expression_set: Set[Text],
    benchmark: benchmark_module.Benchmark,
    settings: settings_module.Settings,
) -> None:
    """Records new solutions in the `solutions` list."""
    reconstructions = value.reconstruct_all_expressions_with_input_names()
    this_solution_time = timeit.default_timer() - start_time
    for expression, used_input_names in reconstructions:
        if expression in solution_expression_set:
            continue
        if not _check_solution(expression, used_input_names, benchmark, settings):
            if settings.printing.bad_solutions:
                print("Bad solution: {}".format(expression))
            continue
        solution_expression_set.add(expression)
        solutions.append(
            Solution(
                value=value,
                expression=expression,
                weight=weight,
                time=this_solution_time,
            )
        )
        if settings.printing.print_solutions:
            print("Found solution: {}".format(expression))
        # Flush so the solutions appear in Colab immediately.
        sys.stdout.flush()
        if len(solutions) >= settings.max_solutions:
            break

def _check_solution_found(value, output_value, benchmark,
                        weight, start_time, end_time,
                        solutions, solution_expression_set, settings, is_prediction=False):
    possible_first_solution = not solutions
    if settings.printing.print_solutions:
        if is_prediction:
            print("Found Solution using prediction")
        else:
            print("Found Solution from enumerative search")
    # Found solution(s), but some may be bad.
    _record_solutions(
        value,
        weight,
        start_time,
        solutions,
        solution_expression_set,
        benchmark,
        settings,
    )
    if possible_first_solution and solutions:
        end_time = min(
            end_time,
            timeit.default_timer()
            + settings.max_extra_solutions_time,
        )
    return end_time


def _find_solutions_multi_model(
    benchmark: benchmark_module.Benchmark,
    operations: List[operation_base.Operation],
    start_time: float,
    settings: settings_module.Settings,
    prediction_model: Optional[PredictionModel] = None,
    snippet_constant_multipliers: Optional[Dict[Text, float]] = None
) -> Tuple[
    List[Solution],
    Set[value_module.Value],
    ValuesByWeight,
    Optional[operation_statistics.OperationStatistics],
]:
    """Helper, returning (solutions, value_set, values_by_weight, statistics)."""
    timeout_reached = False
    end_time = start_time + settings.timeout

    only_minimal_solutions = settings.only_minimal_solutions
    if settings.max_solutions == 1:
        # If we only want one solution, it will be minimal.
        only_minimal_solutions = True

    # An object to track statistics, if requested.
    statistics = (
        operation_statistics.OperationStatistics()
        if settings.printing.statistics
        else None
    )

    # A list of Solution namedtuples.
    solutions = []

    # A set of string solution expressions (don't return duplicate solutions).
    solution_expression_set = set()

    # The output value to search for.
    output_value = value_module.OutputValue(benchmark.examples[0].output)

    # A list of OrderedDicts mapping Value objects to themselves. The i-th
    # OrderedDict contains all Value objects of weight i.
    values_by_weight = [
        collections.OrderedDict() for _ in range(settings.max_weight + 1)
    ]

    # Find and cache the constant and casting operations for use later.
    constant_operation = None
    int_operation = None
    float_operation = None
    bool_operation = None
    for operation in operations:
        if operation.name == torch_functions.CONSTANT_OPERATION_NAME:
            constant_operation = operation
        elif operation.name == torch_functions.INT_OPERATION_NAME:
            int_operation = operation
        elif operation.name == torch_functions.FLOAT_OPERATION_NAME:
            float_operation = operation
        elif operation.name == torch_functions.BOOL_OPERATION_NAME:
            bool_operation = operation
    # Create the output dtype value for use later.
    dtype_value = value_module.ConstantValue(output_value.dtype)

    # Populate values_by_weight with inputs and constants. This also prints
    # inputs/output/constants to stdout.
    _add_constants_and_inputs_and_print(
        values_by_weight, benchmark, output_value, constant_operation, settings, snippet_constant_multipliers
    )

    # A set storing all values found so far.
    value_set = set().union(*values_by_weight)
    constants_values = [value for value in value_set if not value.is_tensor]

    input_values = [value for value in value_set if isinstance(value, value_module.InputValue)]

    value_trial_list = []
    value_trial_list.extend([[value] for value in input_values])
    double_products = product(list(input_values), list(input_values))
    double_products = [list(p) for p in double_products]
    value_trial_list.extend(double_products)

    # TODO(daye): update this to cover every combination, with smarter prioritization
    # Current version covers all the benchmark cases.
    # It might be better to ignore some combinations, give up some examples that will
    # take long time either way (i.e., complicated ones)
    example_trial_list = []
    # single input tensor, 1 api call - [in1]
    single_1 = [[{"inputs": [value], "output": output_value}] for value in input_values]
    example_trial_list.extend(single_1)
    # double input tensor, 1 api call - [in1, in2]
    double_products = product(list(input_values), list(input_values))
    double_products = [list(p) for p in double_products]
    double_1 = [[{"inputs": values, "output": output_value}] for values in double_products]
    example_trial_list.extend(double_1)
    # double input tensor, 2 api calls - [in1], [in2, 0]
    double_2 = [[{"inputs": [values[0]], "output": output_value},{"inputs": [values[1], 0], "output": output_value}] for values in double_products]
    example_trial_list.extend(double_2)
    # double input tensor, 2 api calls, output1 being the only input to api2. - [in1, in2], [0]
    double_2_1 = [[{"inputs": [values[0], values[1]], "output": output_value},{"inputs": [0], "output": output_value}] for values in double_products]
    example_trial_list.extend(double_2_1)
    triple_products = product(list(input_values), list(input_values), list(input_values))
    triple_products = [list(p) for p in triple_products]
    # [in1, in2], [in3, 0]
    triple_2 = [[{"inputs": [values[0], values[1]], "output": output_value},{"inputs": [values[2], 0], "output": output_value}] for values in triple_products]
    example_trial_list.extend(triple_2)
    # single input tensor, 2 api calls - [in1], [0]
    single_2 = [[{"inputs": [value], "output": output_value},{"inputs": [0], "output": output_value}] for value in input_values]
    example_trial_list.extend(single_2)
    # # double input tensor, 2 api calls, output1 being the first input to api2. - [in1], [0, in1]
    # double_2_1 = [[{"inputs": [values[0]], "output": output_value},{"inputs": [0, values[1]], "output": output_value}] for values in double_products]
    # example_trial_list.extend(double_2_1)
    # # double input tensor, 2 api calls - [in1], [in2], [0, 0, in2]
    double_2 = [[{"inputs": [values[0]], "output": output_value},{"inputs": [values[1]], "output": output_value},{"inputs": [0, 0, values[2]], "output": output_value}] for values in triple_products]
    example_trial_list.extend(double_2)

    # # single input tensor, 2 api calls - [], [in1]
    # single_2 = [[{"inputs": [], "output": output_value},{"inputs": [value], "output": output_value}] for value in input_values]
    # example_trial_list.extend(single_2)
    # # double input tensor, 3 api calls, first api input to be none. - [], [in1], [0, in2]
    # double_3 = [[{"inputs": [], "output": output_value},{"inputs": [values[0]], "output": output_value},{"inputs": [0, values[1]], "output": output_value}] for values in double_products]
    # example_trial_list.extend(double_3)


    # for values in value_trial_list:
    for example_sequence in example_trial_list:
        result_values = set()
        predicted_sequences = prediction_model.get_predicted_sequence(example_sequence=example_sequence, settings=settings)
        # predicted_sequences: [sequence, sequence, ...]
#                            : [[operation, operation, ...], [operation, operation ...]]
        for sequence in predicted_sequences:
            if settings.printing.predicted_operations:
                print("sequence: {}".format([op.name for op in sequence]))
            # intermediate_values = set(value_set)
            intermediate_values = []
            prev_intermediate_values = []
            # sequence: [operation, operation, ...]
            for i_op, operation in enumerate(sequence):
                intermediate_values = []
                new_intermediate_values = set()
                if i_op < len(example_sequence):
                    cur_api_inputs = example_sequence[i_op]["inputs"]
                    # print("Cur API Input")
                    # print("With example, inputs: [{}],".format(", ".join([i.reconstruct_expression() if isinstance(i, value_module.Value) else str(i) for i in example_sequence[i_op]['inputs']])))
                    # 0 is a placeholder for the previous api's output.
                    if 0 not in cur_api_inputs:
                        intermediate_values.append(cur_api_inputs)
                        # cur_api_inputs = [cur_api_inputs+[i_value] for i_value in intermediate_values]
                        # intermediate_values.extend(cur_api_inputs)
                    elif cur_api_inputs.count(0) == 1:
                        # for this version, there will be at most one 0 in each api input
                        cur_intermediate_values = []
                        for in_value in prev_intermediate_values:
                            intermediate_value = []
                            for iv in cur_api_inputs:
                                if iv == 0:
                                    intermediate_value.append(in_value)
                                else:
                                    intermediate_value.append(iv)
                            cur_intermediate_values.append(intermediate_value)
                            # intermediate_values.append([[in_value] if cur_api_inputs[iv] == 0 else cur_api_inputs[iv] for iv in range(len(cur_api_inputs))])
                            # print(intermediate_value)
                            # intermediate_values.append(intermediate_value)
                        # cur_api_inputs = [[i_value]+cur_api_inputs[1:] for i_value in intermediate_values]
                        intermediate_values.extend(cur_intermediate_values)
                    elif cur_api_inputs.count(0) == 2:
                        # for this version, there will be at most one 0 in each api input
                        cur_intermediate_values = []
                        for in_values in product(prev_intermediate_values, prev_intermediate_values):
                            intermediate_value = []
                            in_value_idx = 0
                            for iv in cur_api_inputs:
                                if iv == 0:
                                    intermediate_value.append(in_values[in_value_idx])
                                    in_value_idx += 1
                                else:
                                    intermediate_value.append(iv)
                            cur_intermediate_values.append(intermediate_value)
                            # intermediate_values.append([[in_value] if cur_api_inputs[iv] == 0 else cur_api_inputs[iv] for iv in range(len(cur_api_inputs))])
                            # print(intermediate_value)
                            # intermediate_values.append(intermediate_value)
                        # cur_api_inputs = [[i_value]+cur_api_inputs[1:] for i_value in intermediate_values]
                        intermediate_values.extend(cur_intermediate_values)
                if settings.printing.verbose:
                    print("availalbe input for API-{}".format(i_op))
                    print([i.reconstruct_expression() if isinstance(i, value_module.Value) else i for i in intermediate_values])
                for intermediate_value in intermediate_values:
                    if len(intermediate_value) == 2 and operation.name in ['torch.mul(input, other)']:
                        new_values = []
                        for value in intermediate_value:
                            if value.is_tensor and value.value.dtype == torch.bool:
                                new_values.append(all_operations.find_operation_with_name('IntOperation').apply([value], settings))
                            else:
                                new_values.append(value)
                        intermediate_value = new_values
                    elif len(intermediate_value) == 3 and operation.name in ['torch.where(condition, input, other)', 'torch.where(condition, self, other)']:
                        if intermediate_value[0].is_tensor and intermediate_value[0].value.dtype != torch.bool:
                            intermediate_value[0] = all_operations.find_operation_with_name('BoolOperation').apply([intermediate_value[0]], settings)
                    if not isinstance(intermediate_value, list):
                        intermediate_value = [[intermediate_value]]
                    else:
                        intermediate_value = [[v] for v in intermediate_value]

                    predicted_values = operation.enumerate_values_with_values(
                        given_values=intermediate_value,
                        potential_value_list=constants_values,
                        end_time=end_time,
                        settings=settings,
                        statistics=statistics
                    )
                    for predicted_value in predicted_values:
                        if predicted_value not in value_set:
                            if settings.printing.verbose:
                                expression = predicted_value.reconstruct_expression()
                                print("{} produces:\n{}".format(expression, predicted_value))

                            if predicted_value == output_value:
                                end_time = _check_solution_found(predicted_value, output_value, benchmark,
                                                            0, start_time, end_time,
                                                            solutions, solution_expression_set, settings, True)
                                if len(solutions) >= settings.max_solutions:
                                    return (
                                        solutions,
                                        value_set,
                                        values_by_weight,
                                        statistics,
                                    )
                            elif all_operations.find_operation_with_name('IntOperation').apply([predicted_value], settings) == output_value:
                                end_time = _check_solution_found(predicted_value, output_value, benchmark,
                                                                0, start_time, end_time,
                                                                solutions, solution_expression_set, settings, True)
                                if len(solutions) >= settings.max_solutions:
                                    return (
                                        solutions,
                                        value_set,
                                        values_by_weight,
                                        statistics
                                    )
                            else:
                                new_intermediate_values.add(predicted_value)
                                # do casting to new values
                        if i_op == len(sequence)-1:
                            result_values.add(predicted_value)
                prev_intermediate_values += list(new_intermediate_values)
            if timeit.default_timer() > end_time:
                timeout_reached = True
                # Don't return immediately; still try to cast new values because this is
                # relatively quick.
                break

        # Try casting new values to the output dtype if this has a chance of being
        # a correct solution.
        for new_value in result_values:
            if (new_value.shape == output_value.shape
                and new_value.dtype != output_value.dtype
                and operation_filtering.is_castable(new_value, dtype_value)
            ):
                casted_value = None
                if output_value.dtype == torch.int:
                    casted_value = int_operation.apply([new_value], settings)
                elif output_value.dtype == torch.bool:
                    casted_value = bool_operation.apply([new_value], settings)
                elif output_value.dtype == torch.float:
                    casted_value = float_operation.apply([new_value], settings)
                if casted_value == output_value:
                    possible_first_solution = not solutions
                    # Found solution(s), but some may be bad.
                    _record_solutions(
                        casted_value,
                        0,
                        start_time,
                        solutions,
                        solution_expression_set,
                        benchmark,
                        settings,
                    )
                    if possible_first_solution and solutions:
                        end_time = min(
                            end_time,
                            timeit.default_timer() + settings.max_extra_solutions_time,
                        )
                    if len(solutions) >= settings.max_solutions:
                        return solutions, value_set, values_by_weight, statistics

        if settings.printing.progress:
            print(
                "Found {} distinct values of weight {}, or {} total.".format(
                    len(result_values), 0, len(value_set)
                )
            )
        if only_minimal_solutions and solutions:
            return solutions, value_set, values_by_weight, statistics
        if timeout_reached:
            break

    return solutions, value_set, values_by_weight, statistics

def _get_predicted_values(values, predicted_operation, constants_values, end_time, settings, statistics):
    if len(values) > 1 and predicted_operation.name in ['torch.cat(tensors, dim)', 'torch.stack(tensors)', 'torch.stack(tensors, dim)']:
        stacked_value = all_operations.find_operation_with_name('PairCreationOperation').apply(values, settings)
        if stacked_value is None:
            predicted_values = []
        else:
            predicted_values = predicted_operation.enumerate_values_with_values(
                given_values=[[stacked_value]],
                potential_value_list=constants_values,
                end_time=end_time,
                settings=settings,
                statistics=statistics
            )
    if len(values) == 2 and predicted_operation.name in ['torch.mul(input, other)']:
        new_values = []
        for value in values:
            if value.is_tensor and value.value.dtype == torch.bool:
                new_values.append(all_operations.find_operation_with_name('IntOperation').apply([value], settings))
            else:
                new_values.append(value)
        # values = [all_operations.find_operation_with_name('IntOperation').apply(value, settings) if value.is value.value.dtype == torch.bool and value is not None else value for value in values]
        predicted_values = predicted_operation.enumerate_values_with_values(
            given_values=[[value] for value in new_values],
            potential_value_list=constants_values,
            end_time=end_time,
            settings=settings,
            statistics=statistics
        )
    elif len(values) == 3 and predicted_operation.name in ['torch.where(condition, input, other)', 'torch.where(condition, self, other)']:
        if values[0].value.dtype != torch.bool:
            values[0] = all_operations.find_operation_with_name('BoolOperation').apply([values[0]], settings)
        predicted_values = predicted_operation.enumerate_values_with_values(
                given_values= [[value] for valule in values],
                potential_value_list=constants_values,
                end_time=end_time,
                settings=settings,
                statistics=statistics
            )
    elif len(values) == 1 and predicted_operation.name in ['torch.argmax(input)', 'torch.argmax(input, dim)']:
        if values[0].value.dtype != torch.int:
            values[0] = all_operations.find_operation_with_name('IntOperation').apply([values[0]], settings)
        predicted_values = predicted_operation.enumerate_values_with_values(
                given_values=[[values[0]]],
                potential_value_list=constants_values,
                end_time=end_time,
                settings=settings,
                statistics=statistics
            )
    else:
        predicted_values = predicted_operation.enumerate_values_with_values(
            given_values=[[value] for value in values],
            potential_value_list=constants_values,
            end_time=end_time,
            settings=settings,
            statistics=statistics
        )
    return predicted_values

# TODO: DFS will speed up the search further
def _find_solutions_first_sequence(
    benchmark: benchmark_module.Benchmark,
    operations: List[operation_base.Operation],
    start_time: float,
    settings: settings_module.Settings,
    prediction_model: Optional[PredictionModel] = None,
    snippet_constant_multipliers: Optional[Dict[Text, float]] = None
) -> Tuple[
    List[Solution],
    Set[value_module.Value],
    ValuesByWeight,
    Optional[operation_statistics.OperationStatistics],
]:
    """Helper, returning (solutions, value_set, values_by_weight, statistics)."""
    timeout_reached = False
    end_time = start_time + settings.timeout

    only_minimal_solutions = settings.only_minimal_solutions
    if settings.max_solutions == 1:
        # If we only want one solution, it will be minimal.
        only_minimal_solutions = True

    # An object to track statistics, if requested.
    statistics = (
        operation_statistics.OperationStatistics()
        if settings.printing.statistics
        else None
    )

    # A list of Solution namedtuples.
    solutions = []

    # A set of string solution expressions (don't return duplicate solutions).
    solution_expression_set = set()

    # The output value to search for.
    output_value = value_module.OutputValue(benchmark.examples[0].output)

    # A list of OrderedDicts mapping Value objects to themselves. The i-th
    # OrderedDict contains all Value objects of weight i.
    values_by_weight = [
        collections.OrderedDict() for _ in range(settings.max_weight + 1)
    ]

    # Find and cache the constant and casting operations for use later.
    constant_operation = None
    int_operation = None
    float_operation = None
    bool_operation = None
    for operation in operations:
        if operation.name == torch_functions.CONSTANT_OPERATION_NAME:
            constant_operation = operation
        elif operation.name == torch_functions.INT_OPERATION_NAME:
            int_operation = operation
        elif operation.name == torch_functions.FLOAT_OPERATION_NAME:
            float_operation = operation
        elif operation.name == torch_functions.BOOL_OPERATION_NAME:
            bool_operation = operation
    # Create the output dtype value for use later.
    dtype_value = value_module.ConstantValue(output_value.dtype)

    # Populate values_by_weight with inputs and constants. This also prints
    # inputs/output/constants to stdout.
    _add_constants_and_inputs_and_print(
        values_by_weight, benchmark, output_value, constant_operation, settings, snippet_constant_multipliers
    )

    # A set storing all values found so far.
    value_set = set().union(*values_by_weight)
    constants_values = [value for value in value_set if not value.is_tensor]
    # non_primitive_values = [value for value in value_set if value.is_tensor or value.is_sequence]
    non_primitive_values = [value for value in value_set if isinstance(value, value_module.InputValue)]

    filter_cache = filtered_values_cache.FilteredValuesCache()

    if settings.model.do_first_in_seq:
        value_set = []
        for _ in range(3):
            value_set = list(set(value_set).union(set(non_primitive_values)))
            value_trial_list = [[]]
            value_trial_list.extend([value] for value in value_set)
            value_trial_list.extend(product(value_set, value_set))
            for values in value_trial_list:
                example = {"inputs": values, "output": output_value}
                predicted_operations = prediction_model.get_first_in_sequence(example=example, settings=settings)
                for predicted_operation in predicted_operations:
                    predicted_values = _get_predicted_values(values, predicted_operation, constants_values, end_time, settings, statistics)
                    for predicted_value in predicted_values:
                        if predicted_value not in value_set:
                            if settings.printing.verbose:
                                expression = predicted_value.reconstruct_expression()
                                print("[prediction] {} produces:\n{}".format(expression, predicted_value))

                            if predicted_value == output_value:
                                end_time = _check_solution_found(predicted_value, output_value, benchmark,
                                                                0, start_time, end_time,
                                                                solutions, solution_expression_set, settings, True)
                                if len(solutions) >= settings.max_solutions:
                                    return (
                                        solutions,
                                        value_set,
                                        values_by_weight,
                                        statistics
                                    )
                            elif all_operations.find_operation_with_name('IntOperation').apply([predicted_value], settings) == output_value:
                                end_time = _check_solution_found(predicted_value, output_value, benchmark,
                                                                0, start_time, end_time,
                                                                solutions, solution_expression_set, settings, True)
                                if len(solutions) >= settings.max_solutions:
                                    return (
                                        solutions,
                                        value_set,
                                        values_by_weight,
                                        statistics
                                    )
                            else:
                                value_set.append(predicted_value)

                if timeit.default_timer() > end_time:
                    timeout_reached = True
                    # Don't return immediately; still try to cast new values because this is
                    # relatively quick.
                    break

            # Try casting new values to the output dtype if this has a chance of being
            # a correct solution.
            for new_value in value_set:
                if (new_value.shape == output_value.shape
                    and new_value.dtype != output_value.dtype
                    and operation_filtering.is_castable(new_value, dtype_value)
                ):
                    casted_value = None
                    if output_value.dtype == torch.int:
                        casted_value = int_operation.apply([new_value], settings)
                    elif output_value.dtype == torch.bool:
                        casted_value = bool_operation.apply([new_value], settings)
                    elif output_value.dtype == torch.float:
                        casted_value = float_operation.apply([new_value], settings)
                    if casted_value == output_value:
                        possible_first_solution = not solutions
                        # Found solution(s), but some may be bad.
                        _record_solutions(
                            casted_value,
                            0,
                            start_time,
                            solutions,
                            solution_expression_set,
                            benchmark,
                            settings,
                        )
                        if possible_first_solution and solutions:
                            end_time = min(
                                end_time,
                                timeit.default_timer() + settings.max_extra_solutions_time,
                            )
                        if len(solutions) >= settings.max_solutions:
                            return solutions, value_set, values_by_weight, statistics

            if only_minimal_solutions and solutions:
                return solutions, value_set, values_by_weight, statistics
            if timeout_reached:
                break

    return solutions, value_set, values_by_weight, statistics

def _find_solutions(
    benchmark: benchmark_module.Benchmark,
    operations: List[operation_base.Operation],
    start_time: float,
    settings: settings_module.Settings,
    prediction_model: Optional[PredictionModel] = None,
    snippet_constant_multipliers: Optional[Dict[Text, float]] = None
) -> Tuple[
    List[Solution],
    Set[value_module.Value],
    ValuesByWeight,
    Optional[operation_statistics.OperationStatistics],
]:
    """Helper, returning (solutions, value_set, values_by_weight, statistics)."""
    timeout_reached = False
    end_time = start_time + settings.timeout

    only_minimal_solutions = settings.only_minimal_solutions
    if settings.max_solutions == 1:
        # If we only want one solution, it will be minimal.
        only_minimal_solutions = True

    # An object to track statistics, if requested.
    statistics = (
        operation_statistics.OperationStatistics()
        if settings.printing.statistics
        else None
    )

    # A list of Solution namedtuples.
    solutions = []

    # A set of string solution expressions (don't return duplicate solutions).
    solution_expression_set = set()

    # The output value to search for.
    output_value = value_module.OutputValue(benchmark.examples[0].output)

    # A list of OrderedDicts mapping Value objects to themselves. The i-th
    # OrderedDict contains all Value objects of weight i.
    values_by_weight = [
        collections.OrderedDict() for _ in range(settings.max_weight + 1)
    ]

    # Find and cache the constant and casting operations for use later.
    constant_operation = None
    int_operation = None
    float_operation = None
    bool_operation = None
    for operation in operations:
        if operation.name == torch_functions.CONSTANT_OPERATION_NAME:
            constant_operation = operation
        elif operation.name == torch_functions.INT_OPERATION_NAME:
            int_operation = operation
        elif operation.name == torch_functions.FLOAT_OPERATION_NAME:
            float_operation = operation
        elif operation.name == torch_functions.BOOL_OPERATION_NAME:
            bool_operation = operation
    # Create the output dtype value for use later.
    dtype_value = value_module.ConstantValue(output_value.dtype)

    # Populate values_by_weight with inputs and constants. This also prints
    # inputs/output/constants to stdout.
    _add_constants_and_inputs_and_print(
        values_by_weight, benchmark, output_value, constant_operation, settings, snippet_constant_multipliers
    )

    # A set storing all values found so far.
    value_set = set().union(*values_by_weight)
    constants_values = [value for value in value_set if not value.is_tensor]
    non_primitive_values = [value for value in value_set if value.is_tensor or value.is_sequence]

    filter_cache = filtered_values_cache.FilteredValuesCache()

    if settings.model.do_iterative_prediction:
    # try with values in value_set and run prediction
        value_trial_list = []
        value_trial_list.extend([[value] for value in non_primitive_values])
        value_trial_list.extend(product(non_primitive_values, non_primitive_values))
        for values in value_trial_list:
            example = {"inputs": values, "output": output_value}
            predicted_operations = prediction_model.get_predicted_operations(example=example, settings=settings)
            for predicted_operation in predicted_operations:
                if len(values) > 1 and predicted_operation.name in ['torch.cat(tensors, dim)', 'torch.stack(tensors)', 'torch.stack(tensors, dim)']:
                    stacked_value = all_operations.find_operation_with_name('PairCreationOperation').apply(values, settings)
                    if stacked_value is None:
                        predicted_values = []
                    else:
                        predicted_values = predicted_operation.enumerate_values_with_values(
                            given_values=[[stacked_value]],
                            potential_value_list=constants_values,
                            end_time=end_time,
                            settings=settings,
                            statistics=statistics
                        )
                else:
                    predicted_values = predicted_operation.enumerate_values_with_values(
                        given_values=[[value] for value in values],
                        potential_value_list=constants_values,
                        end_time=end_time,
                        settings=settings,
                        statistics=statistics
                    )
                for predicted_value in predicted_values:
                    if predicted_value not in value_set:
                        if settings.printing.verbose:
                            expression = predicted_value.reconstruct_expression()
                            print("[prediction] {} produces:\n{}".format(expression, predicted_value))

                        if predicted_value == output_value:
                            end_time = _check_solution_found(predicted_value, output_value, benchmark,
                                                        0, start_time, end_time,
                                                        solutions, solution_expression_set, settings, True)
                            if len(solutions) >= settings.max_solutions:
                                return (
                                    solutions,
                                    value_set,
                                    values_by_weight,
                                    statistics,
                                )

                        else:
                            if settings.model.do_first_in_seq:
                                value_set.add(predicted_value)

    # Value search by weight.
    for weight in range(1, settings.max_weight + 1):
        if settings.printing.progress:
            print("Searching weight {}...".format(weight))

        # Values with the current weight. This might already include leaf values.
        new_values = values_by_weight[weight]

        # # Random iteration of operations
        for operation in random.sample(operations, len(operations)):
            for value in operation.enumerate_values_with_weight(
                target_weight=weight,
                values_by_weight=values_by_weight,
                filter_cache=filter_cache,
                end_time=end_time,
                settings=settings,
                statistics=statistics,
            ):
                if value not in value_set:
                    # This value has never been seen before, or it's the desired output.
                    if settings.printing.verbose:
                        expression = value.reconstruct_expression()
                        print("{} produces:\n{}".format(expression, value))

                    if value == output_value:
                        end_time = _check_solution_found(value, output_value, benchmark,
                                                        weight, start_time, end_time,
                                                        solutions, solution_expression_set, settings)
                        if len(solutions) >= settings.max_solutions:
                            return (
                                solutions,
                                value_set,
                                values_by_weight,
                                statistics,
                            )
                    else:
                        # Only store the value if it isn't a solution. Otherwise, we'll get
                        # lots of "almost duplicate" solutions, e.g., by adding 0.
                        new_values[value] = value
                        # We should never add output_value (or anything equal) to value_set
                        # so that we can continue finding other solutions.
                        value_set.add(value)
                        if settings.model.do_iterative_prediction:
                            if not value.is_tensor:
                                continue
                            value_trial_list = [[value]]
                            value_trial_list.extend(product([value], non_primitive_values))
                            value_trial_list.extend(product(non_primitive_values, [value]))
                            for values in value_trial_list:
                                example = {"inputs": values, "output": output_value}
                                predicted_operations = prediction_model.get_predicted_operations(example=example, settings=settings)
                                for predicted_operation in predicted_operations:
                                    if len(values) > 1 and predicted_operation.name in ['torch.cat(tensors, dim)', 'torch.stack(tensors)', 'torch.stack(tensors, dim)']:
                                        stacked_value = all_operations.find_operation_with_name('PairCreationOperation').apply(values, settings)
                                        if stacked_value is None:
                                            predicted_values = []
                                        else:
                                            predicted_values = predicted_operation.enumerate_values_with_values(
                                                given_values=[[stacked_value]],
                                                potential_value_list=constants_values,
                                                end_time=end_time,
                                                settings=settings,
                                                statistics=statistics
                                            )
                                    else:
                                        predicted_values = predicted_operation.enumerate_values_with_values(
                                            given_values=[[value] for value in values],
                                            potential_value_list=constants_values,
                                            end_time=end_time,
                                            settings=settings,
                                            statistics=statistics
                                        )
                                    for predicted_value in predicted_values:
                                        if predicted_value not in value_set:
                                            if settings.printing.verbose:
                                                expression = predicted_value.reconstruct_expression()
                                                print("[prediction] {} produces:\n{}".format(expression, predicted_value))

                                            if predicted_value == output_value:
                                                end_time = _check_solution_found(predicted_value, output_value, benchmark,
                                                                                0, start_time, end_time,
                                                                                solutions, solution_expression_set, settings, True)

                                                if len(solutions) >= settings.max_solutions:
                                                    return (
                                                        solutions,
                                                        value_set,
                                                        values_by_weight,
                                                        statistics,
                                                    )
                                            else:
                                                if settings.model.do_first_in_seq:
                                                    value_set.add(predicted_value)

                else:  # This value has been seen before.
                    if value in new_values:
                        # The value was already computed differently with this weight.
                        original_value = new_values[value]
                        if isinstance(original_value, value_module.OperationValue):
                            # Only merge reconstructions if this was originally an
                            # OperationValue. (It could be a ConstantValue instead.)
                            operation_value = (
                                original_value
                            )  # type: value_module.OperationValue
                            operation_value.merge_reconstructions(value)
                    elif not only_minimal_solutions:
                        # If we want non-minimal solutions, we need to store the value even
                        # if we have already seen that value with a smaller weight.
                        new_values[value] = value

            if timeit.default_timer() > end_time:
                timeout_reached = True
                # Don't return immediately; still try to cast new values because this is
                # relatively quick.
                break

        # Try casting new values to the output dtype if this has a chance of being
        # a correct solution.
        for new_value in new_values:
            if (new_value.shape == output_value.shape
                and new_value.dtype != output_value.dtype
                and operation_filtering.is_castable(new_value, dtype_value)
            ):
                casted_value = None
                if output_value.dtype == torch.int:
                    casted_value = int_operation.apply([new_value], settings)
                elif output_value.dtype == torch.bool:
                    casted_value = bool_operation.apply([new_value], settings)
                elif output_value.dtype == torch.float:
                    casted_value = float_operation.apply([new_value], settings)
                if casted_value == output_value:
                    possible_first_solution = not solutions
                    # Found solution(s), but some may be bad.
                    _record_solutions(
                        casted_value,
                        weight,
                        start_time,
                        solutions,
                        solution_expression_set,
                        benchmark,
                        settings,
                    )
                    if possible_first_solution and solutions:
                        end_time = min(
                            end_time,
                            timeit.default_timer() + settings.max_extra_solutions_time,
                        )
                    if len(solutions) >= settings.max_solutions:
                        return solutions, value_set, values_by_weight, statistics

        if settings.printing.progress:
            print(
                "Found {} distinct values of weight {}, or {} total.".format(
                    len(new_values), weight, len(value_set)
                )
            )
        if only_minimal_solutions and solutions:
            return solutions, value_set, values_by_weight, statistics
        if timeout_reached:
            break

    return solutions, value_set, values_by_weight, statistics

def _combine_multipliers(
    first: Dict[Text, float], second: Dict[Text, float]
) -> Dict[Text, float]:
    """Combines operation weight multiplier dicts. Modifies the first dict."""
    for name in second:
        first[name] = first.get(name, 1.0) * second[name]
    return first


def get_reweighted_operations(
    benchmark: benchmark_module.Benchmark,
    settings: settings_module.Settings,
    description_handler: Optional[DescriptionHandler] = None,
    prediction_model: Optional[PredictionModel] = None,
    snippet_operation_multipliers: Optional[Dict[Text, float]] = None
) -> List[operation_base.Operation]:
    """Returns a list of operations with correct weights for the problem."""
    include_sparse_operations = (
        not settings.operations.limit_sparse_operations or _contains_sparse(benchmark)
    )
    operations = all_operations.get_operations(
        include_sparse_operations=include_sparse_operations
    )

    operation_names = [op.name for op in operations]
    if len(operation_names) != len(set(operation_names)):
        raise ValueError("Operation names were not unique.")

    if settings.paper_experiments.uniform_weights:
        # Only for experiments in the PLDI paper.
        for operation in operations:
            operation.weight = 1
        return operations

    multipliers = {}
    if description_handler and benchmark.description:
        multipliers = _combine_multipliers(
            multipliers,
            description_handler.get_operation_multipliers(benchmark, settings),
        )

    if prediction_model and settings.model.use_multiplier:
        multipliers = _combine_multipliers(
            multipliers,
            prediction_model.get_operation_multipliers(benchmark, settings),
        )

    if snippet_operation_multipliers:
        multipliers = _combine_multipliers(
            multipliers,
            snippet_operation_multipliers
        )

    for operation in operations:
        operation.weight = max(
            1, int(round(operation.weight * multipliers.get(operation.name, 1)))
        )

    return operations


def run_value_search(
    benchmark: benchmark_module.Benchmark,
    settings: settings_module.Settings,
    description_handler: Optional[DescriptionHandler] = None,
    prediction_model: Optional[PredictionModel] = None,
    snippet_handler: Optional[SnippetHandler] = None,
) -> ValueSearchResults:
    """Performs value search, iterating by the expression weight.

    Starts with the constants and user-provided inputs, and applies the given
    operations, for a given number of iterations. An expression's "weight" is the
    number of nodes in the expression tree.

    Args:
      benchmark: The Benchmark containing input-output examples and constants.
      settings: A Settings object containing settings for this search.
      description_handler: A DescriptionHandler that scores operations based on
        the benchmark's description.
      prediction_model: A PredictionModel that scores operations based on
        the pre-trained prediction model.
      snippet_handler: A SnippetHandler that scores operations based on
        the the original snippet.

    Returns:
      A ValueSearchResults namedtuple.

    Raises:
      ValueError: If max_weight is too large to be reasonable.
    """
    _suppress_warnings()
    if len(benchmark.examples) > 1:
        print("Warning: for now, value search only uses a single example.")

    start_time = timeit.default_timer()

    snippet_operation_multipliers = None
    snippet_constant_multipliers = None
    if benchmark.snippet:
        snippet_operation_multipliers, snippet_constant_multipliers = snippet_handler.get_multipliers(benchmark, settings)

    operations = get_reweighted_operations(
        benchmark,
        settings,
        description_handler=description_handler,
        prediction_model=prediction_model,
        snippet_operation_multipliers=snippet_operation_multipliers
    )
    if settings.model.use_multi_model:
        solutions, value_set, values_by_weight, statistics = _find_solutions_multi_model(
            benchmark=benchmark,
            operations=operations,
            start_time=start_time,
            settings=settings,
            prediction_model=prediction_model,
            snippet_constant_multipliers=snippet_constant_multipliers
        )
    elif settings.model.do_first_in_seq:
        solutions, value_set, values_by_weight, statistics = _find_solutions_first_sequence(
            benchmark=benchmark,
            operations=operations,
            start_time=start_time,
            settings=settings,
            prediction_model=prediction_model,
            snippet_constant_multipliers=snippet_constant_multipliers
        )
    else:
        solutions, value_set, values_by_weight, statistics = _find_solutions(
            benchmark=benchmark,
            operations=operations,
            start_time=start_time,
            settings=settings,
            prediction_model=prediction_model,
            snippet_constant_multipliers=snippet_constant_multipliers
        )

    total_time = timeit.default_timer() - start_time

    if solutions:
        if settings.printing.print_solutions:
            print()
            print(
                "Solution was found in {:.1f} seconds:\n{}".format(
                    solutions[0].time, solutions[0].expression
                )
            )
            if settings.max_solutions != 1:
                print(
                    "Found {} solution(s) in {:.1f} seconds total.".format(
                        len(solutions), total_time
                    )
                )
    else:
        if settings.printing.print_solutions:
            print(
                "Could not find solution within {} seconds.".format(
                    min(settings.timeout, total_time)
                )
            )
    sys.stdout.flush()

    return ValueSearchResults(
        solutions=solutions,
        total_time=total_time,
        value_set=value_set,
        values_by_weight=values_by_weight,
        benchmark=benchmark,
        settings=settings,
        statistics=statistics,
    )


def run_value_search_from_example(
    inputs: Union[List[Any], Dict[Text, Any]],
    output: Any,
    settings: Optional[settings_module.Settings] = None,
    **kwargs
) -> ValueSearchResults:
    """Performs value search for a single user-provided input-output example.

    Args:
      inputs: A list of inputs, or a dict mapping input names to inputs.
      output: The corresponding desired output.
      settings: An optional Settings object to use, or None to use defaults.
      **kwargs: The kwarg 'constants' can be used to specify a list of constants,
        and 'description' can be used to provide a natural language description of
        the task. Other arguments are passed directly to run_value_search().

    Returns:
      A ValueSearchResults namedtuple.
    """
    if settings is None:
        settings = settings_module.default_settings()
    constants = kwargs.pop("constants", None)
    description = kwargs.pop("description", None)
    snippet = kwargs.pop("snippet", None)
    source = kwargs.pop("source", "From user-provided example.")
    benchmark = benchmark_module.Benchmark(
        examples=[benchmark_module.Example(inputs, output)],
        constants=constants,  # Will turn into empty list if constants=None.
        description=description,  # Will turn into '' if description=None.
        snippet=snippet,
        source=source,
    )

    description_handler = description_handler_factory.create_handler(
        settings.description_handler_name
    )
    if settings.printing.print_init:
        print("Description handler: {!r}\n".format(description_handler))
    prediction_model = prediction_model_factory.load_model(
        "classification"
    )
    if settings.printing.print_init:
        print("Prediction model: {!r}\n".format(prediction_model))
    snippet_handler = snippet_handler_factory.create_handler(
        "function_constant"
    )
    if settings.printing.print_init:
        print("Snippet handler: {!r}\n".format(snippet_handler))


    return run_value_search(benchmark, settings, **kwargs)
