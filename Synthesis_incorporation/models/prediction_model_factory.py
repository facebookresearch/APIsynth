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
"""Creates prediction model from strings."""

import collections
from typing import Callable, Dict, List, Text

from tf_coder.models import prediction_model
from tf_coder.value_search import value_search_settings as settings_module


# Use lambdas to avoid instantiating handlers until they're used.
PREDICTION_MODEL_FNS = collections.OrderedDict(
    [
        ("classification", prediction_model.ClassificationModel),
    ]
)  # type: Dict[Text, Callable[[], prediction_model.PredictionModel]]


def handler_string_list() -> List[Text]:
    """Returns a list of available handler strings."""
    return list(PREDICTION_MODEL_FNS.keys())


def load_model(handler_string: Text, settings: settings_module.Settings) -> prediction_model.PredictionModel:
    """Returns a PredictionModel corresponding to the given handler string."""
    if handler_string not in PREDICTION_MODEL_FNS:
        raise ValueError("Unknown snippet handler: {}".format(handler_string))
    # Evaluate the lambda to get the handler.
    return PREDICTION_MODEL_FNS[handler_string](settings)
