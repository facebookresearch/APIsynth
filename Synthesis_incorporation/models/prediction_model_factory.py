# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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
