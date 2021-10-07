# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Lint as: python3
"""An interface for predicting operations given input and output."""

import abc
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Text
from itertools import product

import six
from tf_coder.benchmarks import benchmark as benchmark_module
from tf_coder.value_search import all_operations
from tf_coder.value_search import operation_base
from tf_coder.value_search import value_search_settings as settings_module
from tf_coder.value_search import value as value_module
from tf_coder.models.models import Net

from iopath.common.file_io import PathManager
from iopath.fb.manifold import ManifoldPathHandler

def load_checkpoint(checkpoint_path, map_location=None):
    pm = PathManager()
    pm.register_handler(ManifoldPathHandler())
    with pm.open(checkpoint_path, "rb") as f:
        if map_location is not None:
            checkpoint = torch.load(f, map_location=map_location)
        else:
            checkpoint = torch.load(f, map_location=lambda storage, loc: storage)
    return checkpoint


@six.add_metaclass(abc.ABCMeta)
class PredictionModel(object):
    """Apply prediction model's results in PyCoder.

    Attributes:
      operations: A list of operations that the handler knows about.
      all_names: A list of operation names, in the same order as the `operations`
        list.
    """

    def __init__(self, operations: Optional[List[operation_base.Operation]] = None):
        """Initializes the handler.

        Args:
          operations: A list of operations that the scorer should handle. Exposed
            for testing.
        Raises:
          ValueError: If there are duplicate operation names.
        """
        self.operations = (
            operations
            if operations
            else all_operations.get_operations(include_sparse_operations=True)
        )
        self.all_names = [operation.name for operation in self.operations]
        if len(set(self.all_names)) != len(self.operations):
            raise ValueError("Duplicate operation name.")

    @abc.abstractmethod
    def get_operation_multipliers(
        self, benchmark: benchmark_module.Benchmark, settings: settings_module.Settings
    ) -> Dict[Text, float]:
        """Returns a map from operation names to their weight multiplier.

        The weight multiplier should be between 0 and 1 if the operation should be
        prioritized, or greater than 1 if it should be deprioritized.

        Args:
          benchmark: Benchmark object corresponding to the TF-Coder task.
          settings: A Settings object storing settings for this search.

        Returns:
          A map from operation name to weight multiplier, such that the operation
          with that name should have its weight modified by that multiplier. If the
          dict does not contain a key, it means the weight should not be modified
          (equivalent to a multiplier of 1).
        """

    def __repr__(self) -> Text:
        """Returns a string containing details about this handler and parameters."""
        return self.__class__.__name__


class ClassificationModel(PredictionModel):
    def __init__(
        self,
        settings: settings_module.Settings,
        operations: Optional[List[operation_base.Operation]] = None,
    ):
        super(ClassificationModel, self).__init__(operations)
        self.checkpoint_path = settings.model.checkpoint_path
        self.api_map_path = settings.model.api_map_path
        self.multi_ffn_path = settings.model.multi_ffn_path
        self.multi_rnn_path = settings.model.multi_rnn_path
        self.multi_api_map_path = settings.model.multi_api_map_path

        self.api2indx = load_checkpoint(self.api_map_path)
        self.multi_api2indx = load_checkpoint(self.multi_api_map_path)

        self.embedding_size = settings.model.embedding_size
        self.shape_embedding_size = settings.model.shape_embedding_size

        self.use_shape_encoding = settings.model.use_shape_encoding
        self.use_type_encoding = settings.model.use_type_encoding
        self.use_value_encoding = settings.model.use_value_encoding
        self.rnn_hidden_dims = settings.model.rnn_hidden_dims
        self.rnn_num_layers = settings.model.rnn_num_layers

        self.settings = settings

        # self.load_model(settings.model.use_multi_model)
        self.load_model(settings)

    def load_model(self, settings):
        device = torch.device("cpu")
        if settings.model.use_multi_model or settings.model.do_first_in_seq:
            self.multi_ffn_model = load_checkpoint(self.multi_ffn_path).to(device)
            self.multi_model = load_checkpoint(self.multi_rnn_path).to(device)
            self.indx2api = {v: k for k, v in self.multi_api2indx.items()}
            if self.multi_api2indx.get('<eol>', -1) == -1:
                max_key = max(self.indx2api.keys())
                self.indx2api[max_key+1] = '<eol>'
                self.multi_api2indx['<eol>'] = max_key+1

        else:
            self.model = Net(self.settings, len(self.api2indx)).to(device)
            checkpoint = load_checkpoint(self.checkpoint_path)
            self.model.load_state_dict(checkpoint)
            self.model.eval()

        # check input tensor type and adjust model
    def embed_benchmark_example(self, example):
        it_pad = []

        input_list = example.inputs

        for input_tensor in input_list:
            input_tensor = torch.tensor(input_tensor)
            it_pad.append(self.tensor_flatten_pad(input_tensor))

        for _ in range(len(it_pad),3):
            t = torch.zeros(self.embedding_size + self.shape_embedding_size + 2 + 1)
            t[-1] = -1
            it_pad.append(t)

        ot_pad = self.tensor_flatten_pad(example.output, isNoise=False)
        domain_embedding = torch.flatten(torch.stack((it_pad[0], it_pad[1], it_pad[2], ot_pad)))
        return domain_embedding


    def embed_benchmark_value(self, example):
        it_pad = []

        input_list = example['inputs']

        for input_tensor in input_list:
            if input_tensor == 0:
                embedding_size = self.embedding_size
                if self.use_shape_encoding:
                    embedding_size += self.shape_embedding_size
                if self.use_type_encoding:
                    embedding_size += 2
                it_pad.append(torch.zeros(embedding_size + 1))
            else:
                if input_tensor.is_tensor:
                    input_tensor = input_tensor.value
                elif input_tensor.is_sequence and not input_tensor.elem_type_is_tensor:
                    input_tensor = torch.tensor(input_tensor.value)
                else:
                    input_tensor = torch.tensor(input_tensor.value)

                it_pad.append(self.tensor_flatten_pad(input_tensor))

        for _ in range(len(it_pad),3):
            embedding_size = self.embedding_size
            if self.use_shape_encoding:
                embedding_size += self.shape_embedding_size
            if self.use_type_encoding:
                embedding_size += 2
            t = torch.zeros(embedding_size + 1)
            t[-1] = -1
            it_pad.append(t)

        output_tensor = example['output'].value
        if not isinstance(output_tensor, torch.Tensor):
            output_tensor = torch.tensor(output_tensor.value)
        ot_pad = self.tensor_flatten_pad(output_tensor)
        domain_embedding = torch.flatten(torch.stack((it_pad[0], it_pad[1], it_pad[2], ot_pad)))
        return domain_embedding.float()

    def encode_values_to_code(self, tensor):
        tensor = tensor.clone()
        tensor[(tensor>=100) & (tensor<1000)] = 100
        tensor[(tensor>=1000)] = 101
        tensor[(tensor<=-20) & (tensor>-100)] = -20
        tensor[(tensor<=-100) & (tensor>-1000)] = -21
        tensor[(tensor<=-1000)] = -22
        return tensor

    def tensor_flatten_pad(
            self, tensor, embed_size = None, shape_embed_size = None, isNoise = False
        ):
        if embed_size is None:
            embed_size = self.embedding_size
        if shape_embed_size is None:
            shape_embed_size = self.shape_embedding_size

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.tensor(tensor)

        t_flatten = torch.flatten(tensor)

        if self.use_value_encoding:
            t_flatten = self.encode_values_to_code(t_flatten)

        padding_length = embed_size - list(t_flatten.shape)[-1]
        p1d = (0,padding_length) #just padding the last dimension
        t_pad = F.pad(input=t_flatten, pad=p1d, mode='constant', value=0)

        if self.use_type_encoding:
            type_padding = 0
            if tensor.dtype == torch.bool:
                type_padding = 1
            if tensor.dtype == torch.float:
                type_padding = 2

        '''size embedding'''
        if self.use_shape_encoding:
            if not isinstance(tensor, torch.Tensor):
                t_shape = []
            else:
                t_shape = list(tensor.shape)
            padding_length = shape_embed_size -1 - len(t_shape)
            p1d = (0,padding_length) #just padding the last dimension
            s_pad = F.pad(input=torch.tensor(t_shape), pad=p1d, mode='constant', value=0)

            t_pad_list = t_pad.tolist()
            s_pad_list = s_pad.tolist()

            if self.use_type_encoding:
                tensor_embedding = torch.tensor([type_padding] + [-1] + t_pad_list + [-1] + s_pad_list + [-1])
            else:
                tensor_embedding = torch.tensor(t_pad_list + [-1] + s_pad_list + [-1])
        else:
            t_pad_list = t_pad.tolist()
            if self.use_type_encoding:
                tensor_embedding = torch.tensor([type_padding] + [-1] + t_pad_list + [-1])
            else:
                tensor_embedding = torch.tensor(t_pad_list + [-1])

        return tensor_embedding.float()


    def predict_operation(self, example, top_n, threshold, is_example, settings):
        if is_example:
            domain_embedding = self.embed_benchmark_example(example)
        else:
            domain_embedding = self.embed_benchmark_value(example)

        with torch.no_grad():
            predicts, _, _, _ = self.model(domain_embedding)

        confidence = predicts
        num_gt_threshold = sum(c >= threshold for c in confidence)

        predicted_api_list = (torch.argsort(predicts, descending=True)).numpy()
        topn_list = predicted_api_list[:min(top_n, num_gt_threshold)]

        topn_operations = [list(self.api2indx.keys())[list(self.api2indx.values()).index(api)] for api in topn_list]
        topn_confidences = [confidence[api].item() for api in topn_list]
        return topn_operations, topn_confidences


    def predict_sequence(self, example_sequence, top_n, beam_n, threshold, is_example, settings):
        if is_example:
            domain_embedding = self.embed_benchmark_example(example_sequence)
        else:
            embeddings = []
            for example in example_sequence:
                embeddings.append(self.embed_benchmark_value(example))
            for _ in range(len(example_sequence), 3):
                embeddings.append(torch.zeros(embeddings[0].shape))
            domain_embedding = torch.stack((embeddings[0], embeddings[1], embeddings[2]))

        with torch.no_grad():
            predicts, z3, z2, z1 = self.multi_ffn_model(domain_embedding)
            temp_z3 = torch.unsqueeze(z3,0)
            model_output, hidden, int_output = self.multi_model(temp_z3)

        topn_list = []
        topn_prob_list = []

        for i, m in enumerate(model_output):
            topn = []
            topn_prob = []
            prob = torch.nn.functional.softmax(m, dim=0).data
            # Taking the class with the highest probability score from the output
            topn_ops = torch.topk(prob,beam_n,dim=0)[1]
            if settings.printing.predicted_operations:
                print(i, topn_ops)
            for op in topn_ops.cpu().numpy():
                if settings.printing.predicted_operations:
                    print(self.indx2api[op])
                topn.append(self.indx2api[op])
                topn_prob.append(prob[op].item())
            topn_list.append(topn)
            topn_prob_list.append(topn_prob)
            if settings.printing.predicted_operations:
                print('====')

        topn_operations = list(product(topn_list[0], topn_list[1], topn_list[2]))
        topn_confidences = list(product(topn_prob_list[0], topn_prob_list[1], topn_prob_list[2]))
        topn_confidences = [c[0]*c[1]*c[2] for c in topn_confidences]

        num_gt_threshold = min(sum(c > threshold for c in topn_confidences), top_n)

        topn_operations = [operation for _, operation in sorted(zip(topn_confidences, topn_operations), reverse=True, key=lambda pair: pair[0])]
        topn_confidences = sorted(topn_confidences, reverse=True)

        return topn_operations[:num_gt_threshold], topn_confidences[:num_gt_threshold]


    def get_operation_multipliers(
        self, benchmark: benchmark_module.Benchmark, settings: settings_module.Settings
    ) -> Dict[Text, float]:
        """See base class."""
        if settings.model.use_multi_model:
            predicted_operations, confidence = self.predict_sequence(benchmark.examples[0], settings.model.multiplier_top_n, settings.model.threshold, True, settings)
        else:
            predicted_operations, confidence = self.predict_operation(benchmark.examples[0], settings.model.multiplier_top_n, settings.model.threshold, True, settings)
        if settings.printing.prioritized_operations:
            if settings.model.use_multi_model:
                print("Predicted operations: {}".format(", ".join(["{} ({:.2f})".format(op, c) for op, c in zip(predicted_operations, confidence)])))
                predicted_operations = [{item for seq in predicted_operations for item in seq}]
            else:
                print("Predicted operations: {}".format(", ".join(["{} ({:.2f})".format(op, c) for op, c in zip(predicted_operations, confidence)])))

        multipliers = {}
        for name in self.all_names:
            if name.startswith("torch.") and "(" in name:
                function_name = name[len("torch.") : name.index("(")].lower()
                if function_name in predicted_operations:
                    if settings.printing.prioritized_operations:
                        print(
                            "Classification Model prioritized {}".format(name)
                        )
                    multipliers[name] = settings.model.multiplier
        return multipliers

    def get_predicted_sequence(
        self, example_sequence, settings: settings_module.Settings
    ) -> List[operation_base.Operation]:
        """See base class."""

        predicted_operations, confidence = self.predict_sequence(example_sequence, settings.model.iterative_top_n, settings.model.beam_n, settings.model.threshold, False, settings)

        if settings.printing.predicted_operations:
            print()
            for example in example_sequence:
                print("With example, inputs: [{}],".format(", ".join([i.reconstruct_expression() if isinstance(i, value_module.Value) else str(i) for i in example['inputs']])))
            # print("Predicted operations: {}".format(", ".join(["{} ({:.2f})".format(op, c) for op, c in zip(predicted_operations, confidence)])))
            print(predicted_operations)
            print(confidence)

        operation_list = []
        for sequence in predicted_operations:
            sequence_list = []
            for op in sequence:
                # sequence_list: [[op1_1, op1_2, ...], [op2_1, op2_2, ...], ]
                if op == '<eol>':
                    break
                sequence_list.append(all_operations.find_operation_with_partial_name(op))
            operation_list.extend(product(*sequence_list))

        return operation_list

    def get_predicted_operations(
        self, example, settings: settings_module.Settings
    ) -> List[operation_base.Operation]:
        """See base class."""
        predicted_operations, confidence = self.predict_operation(example, settings.model.iterative_top_n, settings.model.threshold, False, settings)
        if settings.printing.predicted_operations:
            print()
            print("With example, inputs: ({}),".format(", ".join([i.reconstruct_expression() for i in example['inputs']])))
            print("Predicted operations: {}".format(", ".join(["{} ({:.2f})".format(op, c) for op, c in zip(predicted_operations, confidence)])))

        operation_list = []
        for op in predicted_operations:
            operation_list.extend(all_operations.find_operation_with_partial_name(op))
        return operation_list

    def __repr__(self) -> Text:
        """See base class."""
        return "{}".format(self.__class__.__name__)

    def predict_first_in_sequence(self, example_sequence, top_n, threshold, is_example, settings):
        if is_example:
            domain_embedding = self.embed_benchmark_example(example_sequence)
        else:
            embeddings = []
            embeddings.append(self.embed_benchmark_value(example_sequence))
            # for _ in range(1, 3):
            for _ in range(len(embeddings), 3):
                embeddings.append(torch.zeros(embeddings[0].shape))
            domain_embedding = torch.stack((embeddings[0], embeddings[1], embeddings[2]))

        with torch.no_grad():
            predicts, z3, z2, z1 = self.multi_ffn_model(domain_embedding)
            temp_z3 = torch.unsqueeze(z3, 0)
            model_output, hidden, int_output = self.multi_model(temp_z3)

        topn_operations = []
        topn_confidences = []

        topn = []
        topn_prob = []
        prob = torch.nn.functional.softmax(model_output[0], dim=0).data
        topn_ops = torch.topk(prob, top_n, dim=0)[1]
        for op in topn_ops.cpu().numpy():
            if settings.printing.predicted_operations:
                print(self.indx2api[op])
            topn.append(self.indx2api[op])
            topn_prob.append(prob[op].item())
        # topn_operations.append(topn)
        topn_operations = topn
        # topn_confidences.append(topn_prob)
        topn_confidences = topn_prob
        num_gt_threshold = sum(c > threshold for c in topn_confidences)

        topn_operations = [operation for _, operation in sorted(zip(topn_confidences, topn_operations), reverse=True, key=lambda pair: pair[0])]
        topn_confidences = sorted(topn_confidences, reverse=True)

        return topn_operations[:num_gt_threshold], topn_confidences[:num_gt_threshold]

    def get_first_in_sequence(
        self, example, settings: settings_module.Settings
    ) -> List[operation_base.Operation]:
        if settings.printing.predicted_operations:
            print()
            print("With example, inputs: ({}),".format(", ".join([i.reconstruct_expression() for i in example['inputs']])))
        predicted_opeations,confidence = self.predict_first_in_sequence(example, settings.model.iterative_top_n, settings.model.threshold, False, settings)
        if settings.printing.predicted_operations:
            print("Predicted operations: {}".format(", ".join(["{} ({:.2f})".format(op, c) for op, c in zip(predicted_opeations, confidence)])))
        operation_list = []
        for op in predicted_opeations:
            if op != '<eol>':
                operation_list.extend(all_operations.find_operation_with_partial_name(op))
        return operation_list


PREDICTION_TO_NAME_MAP = {
    'abs': "torch.abs",
    'add': "torch.add",
    'all': "torch.all",
    'any': "torch.any",
    'arange': "torch.arange",
    'argmax': "torch.argmax",
    'argsort': "torch.argsort",
    'bincount': "torch.bincount",
    'cat': "torch.cat",
    'cdist': "torch.cdist",
    'cumsum': "torch.cumsum",
    'div': "torch.div",
    'eq': "torch.eq",
    'expand': "ExpandOperation",
    'eye': "torch.eye",
    'flatten': "torch.flatten",
    'gather': "torch.gather",
    'ge': "torch.ge",
    'gt': "torch.gt",
    'index_select': "torch.index_select",
    'le': "torch.le",
    'lt': "torch.lt",
    'logical_and': "torch.logical_and",
    'masked_select': "torch.masked_select",
    'matmul': "torch.matmul",
    'max': "torch.max",
    'maximum': "torch.maximum",
    'mean': "torch.mean",
    'min': "torch.min",
    'minimum': "torch.minimum",
    'mul': "torch.mul",
    'ne': "torch.ne",
    'nonzero': "torch.nonzero",
    'normalize': "torch.nn.functional.normalize",
    'one_hot': "torch.nn.functional.one_hot",
    'pad': "torch.nn.functional.pad",
    'prod': "torch.prod",
    'repeat_interleave': "torch.repeat_interleave",
    'reshape': "torch.reshape",
    'roll': "torch.roll",
    'searchsorted': "torch.searchsorted",
    'sort': "torch.sort",
    'squeeze': "torch.squeeze",
    'sqrt': "torch.sqrt",
    'square': "torch.square",
    'stack': "torch.stack",
    'sub': "torch.sub",
    'sum': "torch.sum",
    'tensordot': "torch.tensordot",
    'tile': "torch.tile",
    'transpose': "torch.transpose",
    'where': "torch.where",
    'unique': "torch.unique",
    'unsqueeze': "torch.unsqueeze",
    'zeros': "torch.zeros",

    "masked": "torch.masked_select",
    "index": "torch.index_select",
    "logical": "torch.logical_and",
    "onehot": "nn.functional.one_hot",

    "float": "FloatOperation",
    "bool": "BoolOperation",
    "int": "IntOperation"
}
