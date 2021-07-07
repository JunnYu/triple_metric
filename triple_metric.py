# coding=utf-8
# Copyright 2021 The HuggingFace Datasets Authors.
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
""" TripleMetric. """

from dataclasses import dataclass
from enum import Enum

import datasets
import numpy as np

_CITATION = """None"""

_DESCRIPTION = """None"""

_KWARGS_DESCRIPTION = """
Produces labelling scores along with its sufficient statistics
from a source against one or more references.
Args:
    predictions: List of List of predicted labels (Estimated targets as returned by a tagger)
    references: List of List of reference labels (Ground truth (correct) target values)
Returns:
    'scores': dict. Summary of the scores for overall and per type
        micro_avg:
            'f1': F1 score, also known as balanced F-score or F-measure,
            'precision': precision,
            'recall': recall
        per_type:
            'f1': F1 score, also known as balanced F-score or F-measure,
            'precision': precision,
            'recall': recall
Examples:
    >>> metric = load_metric("triple_metric.py")
    >>> predictions = [[(0, 1, 2), (1, 1, 2)],[(1, 2, 2)],[(1, 2, 3)]]
    >>> references = [[(0, 1, 1), (1, 1, 2)],[],[(1, 2, 3)]]
    >>> metric.add_batch(predictions=predictions,references=references)
    >>> metric.compute(type_ids=[0, 1])
    {'micro_avg': Score(f1=0.5714, precision=0.5, recall=0.6667), 0: Score(f1=0.0, precision=0.0, recall=0.0), 1: Score(f1=0.8, precision=0.6667, recall=1.0)}
"""


class Triple(Enum):
    TSE = 0
    STE = 1
    SET = 2


@dataclass
class StatisticContainer:
    num_tp: int = 0
    num_pred: int = 0
    num_gold: int = 0


@dataclass
class Score:
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class TripleMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int32"), length=3),
                        id="sequence",
                    ),
                    "references": datasets.Sequence(
                        datasets.Sequence(datasets.Value("int32"), length=3),
                        id="sequence",
                    ),
                }
            ),
        )

    def _compute(
        self, predictions, references, id2label=None, format=Triple.TSE, digits=4
    ):
        type_ids = (
            ["micro_avg"] + list(id2label.keys())
            if id2label is not None
            else ["micro_avg"]
        )
        statistic_dict = {type_id: StatisticContainer() for type_id in type_ids}

        def fliter_type_by_id(data, type_id):
            if type_id == "micro_avg":
                return data
            return set(
                filter(
                    lambda data: data[
                        format if isinstance(format, int) else format.value
                    ]
                    == type_id,
                    data,
                )
            )

        for pred, gold in zip(predictions, references):
            pred_set = set(map(tuple, pred))
            gold_set = set(map(tuple, gold))

            for k, v in statistic_dict.items():
                v.num_tp += len(fliter_type_by_id(pred_set & gold_set, k))
                v.num_pred += len(fliter_type_by_id(pred_set, k))
                v.num_gold += len(fliter_type_by_id(gold_set, k))

        scores = {}
        for k, v in statistic_dict.items():
            if isinstance(k, int) and id2label is not None:
                k = id2label.get(k, k)

            scores[k] = Score(
                f1=np.round(2 * v.num_tp / (v.num_pred + v.num_gold), decimals=digits)
                if (v.num_pred + v.num_gold) != 0
                else 0.0,
                precision=np.round(v.num_tp / v.num_pred, decimals=digits)
                if v.num_pred != 0
                else 0.0,
                recall=np.round(v.num_tp / v.num_gold, decimals=digits)
                if v.num_gold != 0
                else 0.0,
            )

        return scores
