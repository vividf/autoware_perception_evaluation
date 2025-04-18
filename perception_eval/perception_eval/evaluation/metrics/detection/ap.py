# Copyright 2022 TIER IV, Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger
import os.path as osp
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from perception_eval.common.label import LabelType
from perception_eval.common.threshold import get_label_threshold
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult

logger = getLogger(__name__)


class Ap:
    """AP class.

    Attributes:
        ap (float): AP (Average Precision) score.
        matching_average (Optional[float]): Average of matching score.
            If there are no object results, this variable is None.
        matching_mode (MatchingMode): MatchingMode instance.
        matching_threshold (List[float]): Thresholds list for matching.
        matching_standard_deviation (Optional[float]): Standard deviation of matching score.
            If there are no object results, this variable is None.
        target_labels (List[LabelType]): Target labels list.
        tp_metrics (TPMetrics): Mode of TP metrics.
        ground_truth_objects_num (int): Number ground truths.
        tp_list (List[float]): List of the number of TP objects ordered by their confidences.
        fp_list (List[float]): List of the number of FP objects ordered by their confidences.

    Args:
        tp_metrics (TPMetrics): Mode of TP (True positive) metrics.
        object_results (List[List[DynamicObjectWithPerceptionResult]]): Object results list.
        num_ground_truth (int): Number of ground truths.
        target_labels (List[LabelType]): Target labels to evaluate.
        matching_mode (MatchingMode): Matching instance.
        matching_threshold_list (List[float]): Thresholds list for matching.
    """

    def __init__(
        self,
        tp_metrics: Union[TPMetricsAp, TPMetricsAph],
        object_results: List[DynamicObjectWithPerceptionResult],
        num_ground_truth: int,
        target_labels: List[LabelType],  # Only include a target
        matching_mode: MatchingMode,
        matching_threshold_list: List[List[float]],
    ) -> None:
        self.tp_metrics: Union[TPMetricsAp, TPMetricsAph] = tp_metrics
        self.num_ground_truth: int = num_ground_truth

        self.target_labels: List[LabelType] = target_labels
        self.matching_mode: MatchingMode = matching_mode
        self.matching_threshold_list: List[List[float]] = matching_threshold_list

        self.tp_list, self.fp_list, self.conf_list = self._calculate_tp_fp(tp_metrics, object_results)
        precision_list, recall_list = self.get_precision_recall_list_nus()
        self.ap = self._calculate_ap_nusc_style(precision_list, recall_list)
        self.objects_results_num = len(self.conf_list)

    def _calculate_tp_fp(
        self,
        tp_metrics: Union[TPMetricsAp, TPMetricsAph],
        object_results: List[DynamicObjectWithPerceptionResult],
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate TP/FP when object_results are stored as a dict with (matching_mode, threshold) keys.
        This assumes matching has already occurred (e.g. NuScenes-style).
        """
        tp_list: List[float] = []
        fp_list: List[float] = []
        conf_list: List[float] = []

        for obj in object_results:
            is_tp = obj.ground_truth_object is not None and obj.is_label_correct
            conf_list.append(obj.estimated_object.semantic_score)
            tp_list.append(tp_metrics.get_value(obj) if is_tp else 0.0)
            fp_list.append(0.0 if is_tp else 1.0)

        if not conf_list:
            return [], []

        # Sort by confidence
        sorted_indices = np.argsort(conf_list)[::-1]
        tp_sorted = [tp_list[i] for i in sorted_indices]
        fp_sorted = [fp_list[i] for i in sorted_indices]

        tp_list = np.cumsum(tp_sorted).tolist()
        fp_list = np.cumsum(fp_sorted).tolist()

        return tp_list, fp_list, conf_list

    def get_precision_recall_list_nus(self) -> Tuple[List[float], List[float]]:
        precision, recall = [], []
        for i in range(len(self.tp_list)):
            precision.append(self.tp_list[i] / (self.tp_list[i] + self.fp_list[i]))
            recall.append(self.tp_list[i] / self.num_ground_truth if self.num_ground_truth > 0 else 0.0)
        return precision, recall

    def _calculate_ap_nusc_style(
        self,
        precision_list: List[float],
        recall_list: List[float],
        min_recall: float = 0.1,
        min_precision: float = 0.1,
    ) -> float:
        if len(precision_list) == 0:
            return 0.0

        tp = np.array(self.tp_list, dtype=np.float32)
        fp = np.array(self.fp_list, dtype=np.float32)
        precision = tp / (tp + fp)
        recall = tp / float(self.num_ground_truth)

        precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

        recall_interp = np.linspace(0.0, 1.0, 101)
        precision_interp = np.interp(recall_interp, recall, precision_envelope, right=0)

        first_ind = int(round(100 * min_recall)) + 1
        filtered_prec = precision_interp[first_ind:] - min_precision
        filtered_prec[filtered_prec < 0] = 0.0

        return float(np.mean(filtered_prec)) / (1.0 - min_precision)
