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
"""Benchmarks collected/inspired from StackOverflow."""

# Avoid wrapping URLs and target programs to ease clicking and copying.
# pylint: disable=line-too-long

# Every function in this module takes no arguments and creates a benchmark.
# pylint: disable=missing-docstring

import math

import torch
from tf_coder.benchmarks import benchmark


def stackoverflow_01():
    examples = [
        benchmark.Example(
            inputs=[
                # [[5.0, 2.0], [1.0, 3.0], [0.0, -1.0]],
                [[5, 2], [1, 3], [0, 2]]
            ],
            output=[
                # [[5.0, 5.0], [1.0, 1.0], [0.0, 0.0]],
                # [[2.0, 2.0], [3.0, 3.0], [-1.0, -1.0]],
                [[5, 5], [1, 1], [0, 0]],
                [[2, 2], [3, 3], [2, 2]]
            ],
        ),
    ]
    constants = []
    description = "reshape by separating and duplicating columns"
    target_program = "torch.transpose(torch.stack((in1, in1)), 0, 2)"
    source = "https://stackoverflow.com/questions/40441503/tensorflow-tensor-reshape"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_01",
    )


def stackoverflow_02():
    examples = [
        benchmark.Example(
            inputs=[
                # [5, 1, 0, 3, 0, -1, 2, -10, 2],
                [5, 1, 0, 3, 0, 0, 2, 0, 2],
                1,
            ],
            output=[1, 1, 0, 1, 0, 0, 1, 0, 1]
            # [1, 1, 0, 1, 0, -1, 1, -10, 1],
        ),
    ]
    constants = [1]
    description = "clip values that are greater than 1"
    target_program = "torch.minimum(in1, torch.tensor(1))"
    source = (
        "https://stackoverflow.com/questions/46408839/tensorflow-trim-values-in-tensor"
    )
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_02",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_03():
    examples = [
        benchmark.Example(
            inputs=[
                [[11, 22, 33, 44, 55, 66, 77], [70, 60, 50, 40, 30, 20, 10]],
                [[-9, -8, -7, -6, -5, -4, -3], [11, 12, 13, 14, 15, 16, 17]],
            ],
            output=[[11, 22, 33, -6, -5, 66, 77], [70, 60, 50, 14, 15, 20, 10]],
        ),
    ]
    constants = [3, 4, 5]
    description = "replace certain columns with columns from the other tensor"
    target_program = """
  mask = torch.sum(torch.nn.functional.one_hot(torch.tensor(range(3,5)), in1.size(1)), 0)
  solution = torch.add(torch.mul(mask, in2), torch.mul(torch.sub(torch.ones(mask.size(), dtype=torch.int), mask), in1))
  """
    source = "https://stackoverflow.com/questions/44657388/how-to-replace-certain-values-in-tensorflow-tensor-with-the-values-of-the-other"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_03",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_04():
    examples = [
        benchmark.Example(
            inputs=[
                [[12, 23, 34, 45], [66, 77, 88, 99]],
                [[0, 1], [0, 1], [1, 0], [0, 0]],
                [[2, 1], [1, 2], [0, 2], [0, 0]],
            ],
            output=[[34, 77], [23, 88], [66, 34], [12, 12]],
        ),
    ]
    constants = []
    description = "index into the tensor"
    target_program = """
  idxs = torch.stack((in2, in3), dim=1)
  solution = in1[idxs[:, 0], idxs[:, 1]]
  """
    source = "https://stackoverflow.com/questions/33736795/tensorflow-numpy-like-tensor-indexing"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_04",
    )


def stackoverflow_05():
    examples = [
        benchmark.Example(
            inputs=[
                [[4, 3, 1], [6, 5, 2]],
                [[[5, 5]], [[1, 5]], [[6, 0]]],
            ],
            output=[[[29, 35]], [[47, 55]]],
        ),
    ]
    constants = []
    description = "tensor multiplication like np.tensordot"
    target_program = "torch.tensordot(in1, in2, dims=1)"
    source = "https://stackoverflow.com/questions/43067338/tensor-multiplication-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_05",
    )


def stackoverflow_06():
    examples = [
        benchmark.Example(
            inputs=[
                [3, 5, 0, 2, 3, 3, 0],
            ],
            output=[
                [1, 0, 0, 0, 1, 1, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [1, 0, 0, 0, 1, 1, 0],
                [1, 0, 0, 0, 1, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
            ],
            # [
            #     [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            #     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            #     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            #     [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            #     [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            #     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            # ],
        ),
    ]
    constants = []
    description = "binary tensor from vector indicating if elements are equal"
    target_program = "torch.eq(in1, torch.unsqueeze(in1, dim=1)).float()"
    source = "https://stackoverflow.com/questions/47816231/create-binary-tensor-from-vector-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_06",
    )


def stackoverflow_07():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    [[8, 4, 6], [2, 12, 3]],
                    [[11, 12, 5], [9, 12, 12]],
                    [[9, 2, 13], [7, 0, 7]],
                    [[2, 10, 5], [7, 1, 2]],
                ],
            ],
            output=[
                [[8, 4, 6], [11, 12, 5], [9, 2, 13], [2, 10, 5]],
                [[2, 12, 3], [9, 12, 12], [7, 0, 7], [7, 1, 2]],
            ],
        ),
    ]
    constants = []
    description = "swap the first two dimensions of the tensor"
    target_program = "torch.transpose(in1, 0, 1)"
    source = (
        "https://stackoverflow.com/questions/38212205/swap-tensor-axes-in-tensorflow"
    )
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_07",
    )


def stackoverflow_08():
    examples = [
        benchmark.Example(
            inputs=[
                # [-1, 0, -3, 2, 1, 3, 5, -1, -9, 2, 10],
                [1, 0, 0, 2, 1, 3, 5, 0, 1, 2, 10],
                [12, 3, 45, 6, 7, 8, 9, 87, 65, 4, 32],
                1
            ],
            output=[6, 8, 9, 4, 32],
        ),
    ]
    constants = [1]
    description = (
        "select the values in the second tensor where the first "
        "tensor is greater than 1"
    )
    target_program = "torch.masked_select(in2, torch.gt(in1, 1))"
    source = "https://stackoverflow.com/questions/33769041/tensorflow-indexing-with-boolean-tensor"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_08",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_09():
    examples = [
        benchmark.Example(
            inputs=[
                [37, 42, 42, 37, 28, 15, 42, 15],
            ],
            output=[0, 1, 1, 0, 2, 3, 1, 3],
        ),
    ]
    constants = []
    description = "group items by value and get the group indices"
    target_program = """
  original_unique = torch.masked_select(values, torch.tensor([values[i] not in values[:i] for i in range(values.size(0))]))
  solution = torch.argsort(original_unique)[torch.unique(values, return_inverse=True)[1]]
  """
    source = "https://stackoverflow.com/questions/53054668/assign-values-between-0-and-n-1-for-a-vector-of-length-l-with-n-different-eleme"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_09",
    )

@benchmark.ignore('Out of scope - api3(api1, api2)')
def stackoverflow_10():
    examples = [
        benchmark.Example(
            inputs=[
                [[15, 10], [20, -5]],
                [[2, 3, 1], [-2, 5, 0]],
            ],
            output=[[[30, 45, 15], [20, 30, 10]], [[-40, 100, 0], [10, -25, 0]]],
        ),
    ]
    constants = []
    description = "perform matrix multiplication"
    target_program = "torch.matmul(torch.unsqueeze(in1, -1), torch.unsqueeze(in2, 1))"
    source = "https://stackoverflow.com/questions/53094212/tensorflow-sxn-matrix-multiply-with-sxd-matrix-to-output-sxnxd-array"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_10",
    )


def stackoverflow_11():
    examples = [
        benchmark.Example(
            inputs=[
                [4, 0, 1, 1, 0, 4, 0, 0, 3, 4, 1],
            ],
            output=[4, 3, 0, 1, 3],
        ),
    ]
    constants = []
    description = "count the number of occurences of each distinct number"
    target_program = "torch.bincount(in1)"
    source = "https://stackoverflow.com/questions/45194672/how-to-count-elements-in-tensorflow-tensor"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_11",
    )

@benchmark.ignore('Out of scope - api3(api1, api2)')
def stackoverflow_12():
    examples = [
        benchmark.Example(
            inputs=[[[12, 34, 56], [33, 22, 11]]], output=[[12, 56], [33, 11]]
        ),
    ]
    constants = [0, 1, 2]
    description = "remove a column from the tensor"
    target_program = "torch.stack((in1[:,0], in1[:,2]), dim=1)"
    source = "https://stackoverflow.com/questions/47447183/remove-a-set-of-tensors-from-a-tensor-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_12",
    )


def stackoverflow_13():
    examples = [
        benchmark.Example(
            inputs=[
                [[3, 5], [10, 2]],
                # [[[1, 0], [5, 4]], [[3, 10], [2, -2]]],
                [[[1, 0], [5, 4]], [[3, 10], [2, 0]]]
            ],
            output=[[[28, 20], [19, 30]], [[20, 8], [34, 100]]]
            # [[[28, 20], [19, 20]], [[20, 8], [34, 96]]],
        ),
    ]
    constants = []
    description = "multiply vectors by tensor"
    target_program = "torch.transpose(torch.matmul(in1, in2), 0, 1)"
    source = "https://stackoverflow.com/questions/50777704/n-d-tensor-matrix-multiplication-with-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_13",
    )


def stackoverflow_14():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    [
                        [0, 0, 1],
                        [0, 0, 0],
                        [1, 0, 1],
                        [0, 1, 0],
                        [0, 0, 0],
                        [1, 1, 1],
                        [1, 1, 0],
                        # [False, False, True],
                        # [False, False, False],
                        # [True, False, True],
                        # [False, True, False],
                        # [False, False, False],
                        # [True, True, True],
                        # [True, True, False],
                    ]
                ],
            ],
            output=[[1, 0, 1, 1, 0, 1, 1]]
            # [[True, False, True, True, False, True, True]],
        ),
    ]
    constants = []
    target_program = "torch.sum(in1, -1).bool()"
    description = "choose True if any value in a row is True, False otherwise"
    source = "https://stackoverflow.com/questions/35657003/aggregate-each-element-of-tensor-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_14",
    )


def stackoverflow_15():
    examples = [
        benchmark.Example(
            inputs=[
                # [3, 1, 2, 0, 1, -1, 10, 1, -10],
                [3, 1, 2, 0, 1, 0, 10, 1, 0],
                1,
            ],
            output=[3, 0, 2, 0, 0, 0, 10, 0, 0]
            # [3, 0, 2, 0, 0, -1, 10, 0, -10],
        ),
    ]
    constants = [0, 1]
    description = "set all instances of 1 to 0"
    target_program = "torch.where(torch.ne(in1,1), in1, 0)"
    source = "https://stackoverflow.com/questions/39045797/conditional-assignment-of-tensor-values-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_15",
    )


def stackoverflow_16():
    examples = [
        benchmark.Example(
            inputs=[
                # [[2, 5], [3, 0], [8, -7]],
                [[2, 5], [3, 0], [8, 7]],
                # [4, 10, -6],
                [4, 10, 6]
            ],
            output=[[8, 20], [30, 0], [48, 42]]
            # [[8, 20], [30, 0], [-48, 42]],
        ),
    ]
    constants = []
    description = "multiply tensors across the first axis"
    target_program = "torch.mul(in1, torch.unsqueeze(in2, 1))"
    source = "https://stackoverflow.com/questions/46240646/tensor-multiply-along-axis-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_16",
    )


def stackoverflow_17():
    examples = [
        benchmark.Example(
            inputs=[
                # [17, -32, 99],
                [17, 32, 99]
            ],
            output=[[17, 17], [32, 32], [99, 99]]
            # [[17, 17], [-32, -32], [99, 99]],
        ),
    ]
    constants = []
    description = "duplicate each element of a tensor"
    # StackOverflow answer doesn't work.
    target_program = "torch.stack((in1, in1),1)"
    source = "https://stackoverflow.com/questions/51761353/about-tensor-of-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_17",
    )


def stackoverflow_18():
    examples = [
        benchmark.Example(
            inputs=[
                # shape=[2, 2, 3].
                [[[1, 1, 1], [1, 0, 1]], [[1, 2, 3], [4, 5, 6]]],
                # shape=[3, 4].
                [[1, 1, 1, 1], [1, 2, 3, 4], [5, 6, 7, 8]],
                # shape=[4].
                [100, 200, 300, 400],
            ],
            # Shape=[sequence_length, batch_size, 4]=[2, 2, 4].
            output=[
                [[107, 209, 311, 413], [106, 207, 308, 409]],
                [[118, 223, 328, 433], [139, 250, 361, 472]],
            ],
        ),
    ]
    constants = []
    description = "multiply 3D tensor and 2D tensor and add another tensor"
    target_program = "torch.add(in3, torch.matmul(in1, in2))"
    source = "https://stackoverflow.com/questions/38222126/tensorflow-efficient-way-for-tensor-multiplication"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_18",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_19():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    [3, 1, 2],
                    [1, 0, 4],
                    [1, 2, 3],
                    [0, 5, 1],
                    [1, 1, 2],
                    [2, 3, 1],
                    [2, 1, 0],
                ],
            ],
            output=[
                [0, 5, 1],
                [1, 0, 4],
                [1, 1, 2],
                [1, 2, 3],
                [2, 1, 0],
                [2, 3, 1],
                [3, 1, 2],
            ],
        ),
    ]
    constants = []
    description = (
        "sort a tensor considering the first column, breaking ties "
        "using the second column"
    )
    target_program = """
  second_sorted = in1[torch.sort(in1[:, 1])[1]]
  solution = second_sorted[torch.sort(second_sorted[:, 0])[1]]
  """
    source = "https://stackoverflow.com/questions/49399198/sort-a-tensor-based-on-two-columns-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_19",
    )


def stackoverflow_20():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    [7, 2, 1],
                    [4, 5, 1],
                    [4, 4, 2],
                    [3, 4, 3],
                    [0, 0, 1],
                    # [0.7, 0.2, 0.1],
                    # [0.4, 0.5, 0.1],
                    # [0.4, 0.4, 0.2],
                    # [0.3, 0.4, 0.3],
                    # [0.0, 0.0, 1.0],
                ],
            ],
            output=[[1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
        ),
    ]
    constants = []
    description = "compute argmax in each tensor and set it to 1"
    target_program = "torch.nn.functional.one_hot(torch.argmax(in1, 1), in1.size(1))"
    source = "https://stackoverflow.com/questions/44834739/argmax-on-a-tensor-and-ceiling-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_20",
    )


def stackoverflow_21():
    examples = [
        benchmark.Example(
            inputs=[
                [[2], [0], [1], [0]],
                # [[2], [0], [1], [0]],
                [[2, 5, 3], [1, 3, 6], [1, 6, 3], [7, 0, 3]]
                # [[0.2, 0.5, 0.3], [0.1, 0.3, 0.6], [0.1, 0.6, 0.3], [0.7, 0.0, 0.3]],
            ],
            output=[[3], [1], [6], [7]],
            # [[0.3], [0.1], [0.6], [0.7]],
        ),
    ]
    constants = []
    description = "gather elements in a tensor along axis 1"
    target_program = "torch.gather(in2, 1, in1)"
    source = "https://stackoverflow.com/questions/51690095/how-to-gather-element-with-index-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_21",
    )


def stackoverflow_22():
    examples = [
        benchmark.Example(
            inputs=[
                [3, 1, 10],
                [[6, 4], [5, 1], [3, 4]]
                # [[0.6, 0.4], [0.5, 1.0], [3.0, 4.0]],
            ],
            output=[53, 53]
            # [32.3, 42.2],
        ),
    ]
    constants = []
    description = "multiply a vector with a matrix without reshaping the vector"
    target_program = "torch.squeeze(torch.matmul(torch.unsqueeze(in1, 0).float(), in2))"
    source = "https://stackoverflow.com/questions/43284897/how-can-i-multiply-a-vector-and-a-matrix-in-tensorflow-without-reshaping"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_22",
    )


def stackoverflow_23():
    # Simplified slightly because the user already knows how to do the mod part.
    examples = [
        benchmark.Example(
            inputs=[
                [[0, 5, 2], [3, 1, 4], [5, 1, 5]],
            ],
            output=[
                [1, 0, 1, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 1, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0, 0],
            ],
        ),
    ]
    constants = []
    description = "place 1 at the indices in the input tensor"
    target_program = (
        "torch.max(torch.nn.functional.one_hot(in1, 9), 1, keepdim=False, out=None)[0]"
    )
    source = (
        "https://stackoverflow.com/questions/53414433/tensorflow-tensor-binarization"
    )
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_23",
    )


def stackoverflow_24():
    examples = [
        benchmark.Example(
            inputs=[
                # [3.0, 1.0, 4.0, 5.0, 2.0, 8.0, -6.0, -7.0],
                [3, 1, 4, 5, 2, 8, 6, 7],
                # [0.5, 0.0, -2.0, 0.0, 1.0, -1.0, 0.0, 2.0],
                [1, 0, 2, 0, 1, 1, 0, 2],
                0
            ],
            output=[3, 1, 2, 5, 2, 8, 6, 3.5]
            # [6.0, 1.0, -2.0, 5.0, 2.0, -8.0, -6.0, -3.5],
        ),
    ]
    constants = [0]
    description = "like tf.divide(), but when dividing by 0, return the " "numerator"
    target_program = "torch.where(torch.ne(in2, 0), torch.divide(in1, in2), in1)"
    source = "https://stackoverflow.com/questions/53643339/tensorflow-overriding-tf-divide-to-return-the-numerator-when-dividing-by-0"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_24",
    )


def stackoverflow_25():
    examples = [
        benchmark.Example(
            inputs=[
                3,
                4,
            ],
            output=[
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                # [1.0, 0.0, 0.0],
                # [0.0, 1.0, 0.0],
                # [0.0, 0.0, 1.0],
                # [1.0, 0.0, 0.0],
                # [0.0, 1.0, 0.0],
                # [0.0, 0.0, 1.0],
                # [1.0, 0.0, 0.0],
                # [0.0, 1.0, 0.0],
                # [0.0, 0.0, 1.0],
                # [1.0, 0.0, 0.0],
                # [0.0, 1.0, 0.0],
                # [0.0, 0.0, 1.0],
            ],
        ),
    ]
    constants = []
    description = "copy the tensor torch.eye(3), 4 times"
    target_program = "torch.tile(torch.eye(in1), (in2, 1))"
    source = "https://stackoverflow.com/questions/53602691/duplicate-a-tensor-n-times"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_25",
    )


def stackoverflow_26():
    examples = [
        benchmark.Example(
            inputs=[[[[3, 4], [1, 2]], [[5, 2], [10, 3]], [[10, 20], [4, 7]]]],
                # [[[3, 4], [1, 2]], [[5, -2], [-10, 3]], [[10, 20], [-4, 7]]]],
            output=[10, 20, 41]
            # [10, -4, 33],
        ),
    ]
    constants = []
    description = "reduction operation for multiple dimensions simultaneously"
    target_program = "torch.sum(torch.sum(in1, 1), 1)"
    source = "https://stackoverflow.com/questions/54294780/how-to-perform-reduce-op-on-multiple-dimensions-at-once"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_26",
    )


def stackoverflow_27():
    examples = [
        benchmark.Example(
            inputs=[
                [0, 3, 5, 6],
                8,
            ],
            output=[1, 0, 0, 1, 0, 1, 1, 0],
        ),
    ]
    constants = []
    description = "boolean tensor with 1 at the indices in the input tensor"
    target_program = "torch.sum(torch.nn.functional.one_hot(in1, in2), 0)"
    source = "https://stackoverflow.com/questions/54225704/how-do-i-get-a-tensor-representing-the-on-positions-in-the-original-tensor"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_27",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_28():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    [[5, 3], [0, 2]],
                    [[7, 4], [5, 1]],
                    [[10, 20], [15, 30]],
                    [[11, 16], [14, 12]],
                    [[-2, -7], [-4, 6]],
                ],
                [1, 0, 1, 1, 0],
            ],
            output=[[3, 2], [7, 5], [20, 30], [16, 12], [-2, -4]],
        ),
    ]
    constants = []
    description = "extract columns from a 3D tensor given column indices"
    target_program = "torch.transpose(in1, 1, 2)[torch.arange(in1.size(0)), in2, :]"
    source = "https://stackoverflow.com/questions/54274074/selecting-columns-from-3d-tensor-according-to-a-1d-tensor-of-indices-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_28",
    )


def stackoverflow_29():
    examples = [
        benchmark.Example(
            inputs=[
                [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
                # [-1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
                [12, 0, 10, 23, 16],
                # [0.1, -10, -0.1, 1.1, 0.41],
            ],
            output=[6, 0, 5, 11, 8],
        ),
    ]
    constants = []
    description = "place continuous values into buckets given bucket boundaries"
    target_program = "torch.searchsorted(in1, in2)"
    source = "https://stackoverflow.com/questions/54155085/bucketing-continous-value-tensors-in-tensorflow"  # lint: NOTYPO
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_29",
    )


def stackoverflow_30():
    examples = [
        benchmark.Example(
            inputs=[
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                [[9.0, 4.0], [8.0, 5.0], [7.0, 6.0]],
            ],
            output=[
                [math.sqrt(68), math.sqrt(58), math.sqrt(52)],
                [math.sqrt(36), math.sqrt(26), math.sqrt(20)],
                [math.sqrt(20), math.sqrt(10), math.sqrt(4)],
            ],
        ),
    ]
    constants = []
    description = "compute Euclidean distance between two tensors"
    # StackOverflow answer is incorrect.
    target_program = "torch.cdist(in1, in2)"
    source = "https://stackoverflow.com/questions/54147780/tensorflow-how-to-calculate-the-euclidean-distance-between-two-tensor"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_30",
    )

@benchmark.ignore('Input contains sparse tensor')
def stackoverflow_31():
    examples = [
        benchmark.Example(
            inputs=[
                torch.sparse_coo_tensor(
                    indices=torch.tensor([[0, 0, 1], [0, 1, 1]]),
                    values=[1.0, 1.5, -2.0],
                    size=[2, 2],
                ),
                [[3.0, 1.0], [0.2, -1.0]],
            ],
            output=5.29,
        ),
    ]
    constants = []
    description = "squared error between two tensors, one being a sparse tensor"
    target_program = "torch.sum(torch.square(torch.sub(in1.to_dense(), in2)))"
    source = "https://stackoverflow.com/questions/45032668/tensorflow-how-to-compute-the-square-error-between-a-tensor-and-a-sparse-tensor"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_31",
    )


def stackoverflow_32():
    examples = [
        benchmark.Example(
            inputs=[
                [[1, 6, 2, 1], [3, 1, 4, 2], [2, 1, 2, 5]]
                # [[0.1, 0.6, 0.2, 0.1], [0.3, 0.1, 0.4, 0.2], [0.2, 0.1, 0.2, 0.5]],
            ],
            output=[13, 15, 20]
            # [1.3, 1.5, 2.0],
        ),
    ]
    constants = []
    description = "weighted sum across rows, where the column index is the weight"
    target_program = "torch.sum(torch.mul(in1, torch.unsqueeze(torch.arange(in1.size(1)),0).expand(in1.size(0),-1)), 1)"
    source = "https://stackoverflow.com/questions/48659449/how-to-compute-the-weighted-sum-of-a-tensor-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_32",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_33():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    [0.3, 0.1, 0.4],
                    [0.1, 0.5, 0.9],
                    [0.2, 0.6, 0.5],
                    [0.3, 0.5, 0.8],
                    [0.9, 0.7, 0.9],
                ],
                [[0.3, 0.2, 0.3], [0.8, 0.4, 0.6], [0.2, 0.6, 0.4], [0.3, 0.3, 0.8]],
            ],
            output=[0.02, 0.19, 0.01, 0.04],
        ),
    ]
    constants = []
    description = "find the minimum distance between two sets of points"
    target_program = "torch.min(torch.sum(torch.square(torch.sub(torch.unsqueeze(in1, 0), torch.unsqueeze(in2, 1))), 2), 1)[0]"
    source = "https://stackoverflow.com/questions/40558251/computing-minimum-distance-for-each-element-in-a-tensor-relative-to-another-tens"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_33",
    )


def stackoverflow_34():
    examples = [
        benchmark.Example(
            inputs=[
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[10, 20], [30, 40]]],
                [3, 5, 10],
            ],
            output=[[128, 236], [344, 452]],
        ),
    ]
    constants = []
    description = "compute a weighted sum of tensors"
    target_program = "torch.tensordot(in2, in1, 1)"
    source = "https://stackoverflow.com/questions/49532371/compute-a-linear-combination-of-tensors-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_34",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_35():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
                    [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]],
                ],
                [
                    [[9.0, 8.0], [7.0, 6.0], [5.0, 4.0]],
                    [[90.0, 80.0], [70.0, 60.0], [50.0, 40.0]],
                ],
                [0.1, 0.4, 0.8],
            ],
            output=[
                [[8.2, 7.4], [5.4, 5.2], [5.0, 5.6]],
                [[82.0, 74.0], [54.0, 52.0], [50.0, 56.0]],
            ],
        ),
    ]
    constants = []
    description = "linear interpolation between two tensors"
    target_program = (
        "torch.add(in2, torch.mul(torch.unsqueeze(in3, 1), torch.sub(in1, in2)))"
    )
    source = "https://stackoverflow.com/questions/49643371/keras-compute-convex-combination-of-two-tensors"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_35",
    )


def stackoverflow_36():
    examples = [
        benchmark.Example(
            inputs=[
                [1, 0, 1, 1, 0, 1, 0, 1],
            ],
            output=[1.0, 0.0, 0.333333, 0.25, 0.0, 0.166667, 0.0, 0.125],
        ),
    ]
    constants = []
    description = "divide each element by the column index"
    target_program = "torch.div(in1, torch.arange(1,in1.size(0)+1))"
    source = "https://stackoverflow.com/questions/43306788/divide-elements-of-1-d-tensor-by-the-corrispondent-index"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_36",
    )


def stackoverflow_37():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    [
                        [[10, 20, 30], [40, 50, 60]],
                        [[12, 34, 56], [78, 98, 76]],
                        # [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                        # [[1.2, 3.4, 5.6], [7.8, 9.8, 7.6]],
                    ]
                ],
                # [0.5, 1.0, 2.0],
                [5, 10, 20]
            ],
            output=[[[850, 1900], [1520, 2890]]]
            # [[[8.5, 19.0], [15.2, 28.9]]],
        ),
    ]
    constants = []
    description = "dot product a vector with last dimension of a tensor"
    target_program = "torch.tensordot(in1, in2, 1)"
    source = "https://stackoverflow.com/questions/49206051/multiply-4-d-tensor-with-1-d-tensor"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_37",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_38():
    # To simplify the problem, and to get more than one number as output, this
    # doesn't include the final reduce_sum step.
    examples = [
        benchmark.Example(
            inputs=[
                [9, 2, 5, 3, 7, 4],
                [[0, 0, 1, 0, 1, 0], [1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 1]],
            ],
            output=[35, 9, 120],
        ),
    ]
    constants = []
    description = "compute the product of marked elements"
    target_program = "torch.prod(torch.maximum(torch.max(in2), torch.mul(in1, in2)), 1)"
    source = "https://stackoverflow.com/questions/49511529/tensorflow-compute-multiplication-by-binary-matrix"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_38",
    )


def stackoverflow_39():
    examples = [
        benchmark.Example(
            inputs=[
                # [[-1.5, 1.0, 0.9, 2.0], [1.1, 0.0, -0.1, -0.9], [-1.0, 0.1, -1.1, 2.5]],
                [[15, 10, 9, 20], [11, 0, 1, 9], [10, 1, 11, 25]]
            ],
            output=[
                [225, 100, 81, 400],
                [121, 0, 1, 81],
                [100, 1, 121, 625],
                # [2.25, 1.0, 0.0, 4.0],
                # [1.21, 0.0, 0.0, 0.0],
                # [1.0, 0.0, 1.21, 6.25],
            ],
        ),
    ]
    constants = []
    description = (
        "set to 0 the elements with absolute value less than 1, and "
        "square the other elements"
    )
    target_program = (
        "torch.square(torch.where(torch.lt(torch.abs(in1), 1), torch.tensor(0.), in1))"
    )
    source = "https://stackoverflow.com/questions/37912161/how-can-i-compute-element-wise-conditionals-on-batches-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_39",
    )

@benchmark.ignore('Target program contains sparse tensor.')
def stackoverflow_40():
    examples = [
        benchmark.Example(
            inputs=[
                [4, 5, 2, 7, 8, 6],
                [[0, 2], [0, 4], [1, 1], [1, 3], [2, 0], [2, 3]],
            ],
            output=[[0, 0, 4, 0, 5], [0, 2, 0, 7, 0], [8, 0, 0, 6, 0]],
        ),
    ]
    constants = []
    description = "use the output of tf.nn.top_k to make a sparse tensor"
    target_program = (
        "torch.sparse_coo_tensor(torch.transpose(in2, 0, 1), in1, (3,5)).to_dense()"
    )
    source = "https://stackoverflow.com/questions/43996831/make-a-sparse-tensor-based-on-the-output-of-tf-nn-top-k"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_40",
    )


def stackoverflow_41():
    examples = [
        benchmark.Example(
            inputs=[
                [5, 2, 8, 2, 4, 1, 1, 0, 2, 1],
                3,
            ],
            output=[5, 2, 8, 4, 1, 1, 0, 2, 1],
        ),
    ]
    constants = []
    description = "copy all elements except at the given index"
    target_program = "torch.masked_select(in1, torch.ne(torch.arange(in1.size(0)), 3))"
    source = "https://stackoverflow.com/questions/54499051/elegant-way-to-access-python-list-and-tensor-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_41",
    )


def stackoverflow_42():
    examples = [
        benchmark.Example(
            inputs=[
                # [4, 6, 2, 6, 7, 3, -3],
                [4, 6, 2, 6, 7, 3, 3],
                7
            ],
            output=[0, 0, 0, 0, 1, 0, 0],
        ),
    ]
    constants = []
    description = "create a binary vector where the max element is 1"
    target_program = "torch.where(torch.eq(torch.max(in1),in1), 1, 0)"
    source = "https://stackoverflow.com/questions/54493814/binary-vector-of-max"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_42",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_43():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    [12, 34, 56, 78, 90, 10],
                    [99, 88, 77, 55, 44, 33],
                    [-1, -2, -3, -4, -5, -6],
                ],
                [0, 1, 1, 0, 2, 0],
            ],
            output=[12, 88, 77, 78, -5, 10],
        ),
    ]
    constants = []
    description = "extract elements of a tensor given row indices"
    target_program = "torch.squeeze(torch.gather(torch.transpose(in1, 0, 1), 1, torch.unsqueeze(in2, 1)))"
    source = "https://stackoverflow.com/questions/54455169/better-way-to-access-individual-elements-in-a-tensor"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_43",
    )


def stackoverflow_44():
    examples = [
        benchmark.Example(
            inputs=[
                [
                    # [3, 5, 2],
                    # [6, 2, 3],
                    # [8, 7, 1],
                    # [0, -3, 5],
                    # [-4, 7, 3],
                    # [2, 1, 6],
                    # [10, 20, 30],
                    # [4, 5, 6],
                    [3, 5, 2],
                    [6, 2, 3],
                    [8, 7, 1],
                    [0, 3, 5],
                    [4, 7, 3],
                    [2, 1, 6],
                    [10, 20, 30],
                    [4, 5, 6],
                ],
            ],
            output=[[9, 7, 5], [8, 19, 6], [6, 8, 9], [14, 25, 36]]
            # [[9, 7, 5], [8, 4, 6], [-2, 8, 9], [14, 25, 36]],
        ),
    ]
    constants = [2]
    description = "sum across columns for pairs of consecutive rows"
    target_program = "torch.sum(torch.reshape(in1, (-1, 2, in1.size(1))), 1)"
    source = "https://stackoverflow.com/questions/54402389/sum-the-columns-for-each-two-consecutive-rows-of-a-tensor-of-3-dimensions"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_44",
    )

def stackoverflow_45():
    examples = [
        benchmark.Example(
            inputs=[
                [1, 0, 1, 0, 1],
                [[[12, 34], [56, 78], [23, 54], [76, 78], [42, 24]]],
            ],
            output=[[[34, 12], [56, 78], [54, 23], [76, 78], [24, 42]]],
        ),
    ]
    constants = []
    description = "reverse the order in the marked rows"
    target_program = (
        "torch.where(torch.unsqueeze(in2,1).bool(), torch.roll(in1, 1, -1), in1)"
    )
    source = "https://stackoverflow.com/questions/54337925/reverse-order-of-some-elements-in-tensorflow"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_45",
    )


def stackoverflow_46():
    examples = [
        benchmark.Example(
            inputs=[
                [3, 4, 1],
            ],
            output=[0, 0, 0, 1, 1, 1, 1, 2],
        ),
    ]
    constants = []
    description = "convert segment lengths to segment ids"
    target_program = """
  mask = torch.arange(torch.max(in1)).expand(in1.size(0), torch.max(in1)) < torch.unsqueeze(in1, dim=1)
  solution = torch.mul(torch.unsqueeze(torch.arange(mask.size(0)), 1), mask)
  solution = torch.masked_select(solution, mask)
  """
    source = "https://stackoverflow.com/questions/58652161/how-to-convert-2-3-4-to-0-0-1-1-1-2-2-2-2-to-utilize-tf-math-segment-sum"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_46",
    )

@benchmark.ignore('Out of scope')
def stackoverflow_47():
    examples = [
        benchmark.Example(
            inputs=[
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
                [
                    [True, True, True, False, False],
                    [True, True, False, False, False],
                    [True, True, True, True, True],
                    [True, True, True, True, False],
                    [True, False, False, False, False],
                    [True, True, False, False, False],
                ],
            ],
            output=[
                [0, 1, 2, 0, 0],
                [3, 4, 0, 0, 0],
                [5, 6, 7, 8, 9],
                [10, 11, 12, 13, 0],
                [14, 0, 0, 0, 0],
                [15, 16, 0, 0, 0],
            ],
        ),
    ]
    constants = []
    description = "put given values into a sequence mask"
    target_program = """
  mask = torch.reshape(in2, [-1])
  solution = torch.reshape(torch.where(mask, torch.sub(torch.cumsum(mask, 0), 1), torch.tensor(0)), in2.size())
  """
    source = "https://stackoverflow.com/questions/58641546/how-can-i-put-the-sequential-values-to-the-sequence-mask"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_47",
    )


def stackoverflow_48():
    examples = [
        benchmark.Example(
            inputs=[
                [32, 53, 45, 38, 29, 89, 64, 23],
                [38, 53, 89, 38, 32, 64],
            ],
            output=[3, 1, 5, 3, 0, 6],
        ),
    ]
    constants = []
    description = "find the indices of all elements"
    target_program = "torch.argmax(torch.eq(in1, torch.unsqueeze(in2, 1)).int(), 1)"
    source = "https://stackoverflow.com/questions/58481332/getting-the-indices-of-several-elements-in-a-tensorflow-at-once"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_48",
    )


def stackoverflow_49():
    examples = [
        benchmark.Example(
            inputs=[
                # Shape = [3, 1, 2, 3].
                # [
                #     [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]],
                #     [[[0.8, 1.0, 0.0], [0.6, 0.4, 0.2]]],
                #     [[[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]]],
                # ],
                [
                    [[[1, 2, 3], [4, 5, 6]]],
                    [[[8, 10, 0], [6, 4, 2]]],
                    [[[9, 8, 7], [1, 2, 3]]],
                ],
                # [2.0, 0.5, 1.0],
                [20, 5, 10]
            ],
            output=[
                [[[20, 40, 60], [80, 100, 120]]],
                [[[40, 50, 0], [30, 20, 10]]],
                [[[90, 80, 70], [10, 20, 30]]],
                # [[[0.2, 0.4, 0.6], [0.8, 1.0, 1.2]]],
                # [[[0.4, 0.5, 0.0], [0.3, 0.2, 0.1]]],
                # [[[0.9, 0.8, 0.7], [0.1, 0.2, 0.3]]],
            ],
        ),
    ]
    constants = []
    description = "multiply tensors by scalars in a batched way"
    target_program = "torch.transpose(torch.mul(in2, torch.transpose(in1, 0, 3)), 0, 3)"
    source = "https://stackoverflow.com/questions/58466562/given-a-batch-of-n-images-how-to-scalar-multiply-each-image-by-a-different-scal"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_49",
    )


def stackoverflow_50():
    examples = [
        benchmark.Example(
            inputs=[
                # 5,  # Rows.
                # 6,  # Columns.
                3,  # Index of nonzero column.
            ],
            output=[
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 1, 0, 0],
            ],
        ),
    ]
    constants = []
    description = "create a binary matrix where a specified column is set to one"
    target_program = "torch.zeros((in1, in2), dtype=torch.int)"
    source = "https://stackoverflow.com/questions/58537495/tensorflow-initialize-a-sparse-tensor-with-only-one-line-column-not-zero"
    return benchmark.Benchmark(
        examples=examples,
        constants=constants,
        description=description,
        target_program=target_program,
        source=source,
        name="stackoverflow_50",
    )


# # A template for easy copy/pasting. Copying an existing benchmark and replacing
# # parts of it will lead to a state where the benchmark is half-correct, but not
# # obviously so. Copy this template instead when creating new benchmarks.
# """

# def stackoverflow_NUMBER():
#   examples = [
#       benchmark.Example(
#           inputs=[
#               INPUT_1,
#               INPUT_2,
#           ],
#           output=OUTPUT
#       ),
#   ]
#   constants = [CONSTANTS]
#   description = 'DESCRIPTION'
#   target_program = 'SOLUTION_PROGRAM'
#   source = 'PROBLEM_SOURCE'
#   return benchmark.Benchmark(examples=examples,
#                              constants=constants,
#                              description=description,
#                              target_program=target_program,
#                              source=source,
#                              name='stackoverflow_NUMBER')

# """  # pylint: disable=pointless-string-statement
