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

from __future__ import annotations

from typing import Dict
from typing import List
from typing import Optional

from perception_eval.common import ObjectType
from perception_eval.common.dataset import FrameGroundTruth
from perception_eval.common.label import LabelType
from perception_eval.common.status import GroundTruthStatus
from perception_eval.common.status import MatchingStatus
from perception_eval.evaluation import DynamicObjectWithPerceptionResult
from perception_eval.evaluation.matching.objects_filter import divide_objects
from perception_eval.evaluation.matching.objects_filter import divide_objects_to_num
from perception_eval.evaluation.matching.objects_filter import filter_object_results
from perception_eval.evaluation.matching.objects_filter import filter_objects
from perception_eval.evaluation.metrics import MetricsScore
from perception_eval.evaluation.metrics import MetricsScoreConfig
from perception_eval.evaluation.result.perception_frame_config import CriticalObjectFilterConfig
from perception_eval.evaluation.result.perception_frame_config import PerceptionPassFailConfig
from perception_eval.evaluation.result.perception_pass_fail_result import PassFailResult

from perception_eval.common.object import DynamicObject

class PerceptionFrameFiltered:
    """The result for 1 frame (the pair of estimated objects and ground truth objects)

    Args:
        estimated_objects (List[DynamicObject]): Filtered object results to each estimated object.
        frame_ground_truth (FrameGroundTruth): Filtered ground truth of frame.
        frame_name (str): The file name of frame in the datasets.
        unix_time (int): The unix time for frame [us].
        target_labels (List[AutowareLabel]): The list of target label.
    """

    def __init__(
        self,
        estimated_objects: List[DynamicObject],
        frame_ground_truth: FrameGroundTruth,
        metrics_config: MetricsScoreConfig,
        unix_time: int,
        critical_object_filter_config: CriticalObjectFilterConfig,
        target_labels: List[LabelType],
    ) -> None:
        # TODO(ktro2828): rename `frame_name` into `frame_number`
        # frame information
        self.frame_name: str = frame_ground_truth.frame_name
        self.unix_time: int = unix_time
        self.target_labels: List[LabelType] = target_labels

        self.estimated_objects: List[DynamicObject] = estimated_objects
        self.frame_ground_truth: FrameGroundTruth = frame_ground_truth

        # init evaluation
        self.metrics_score: MetricsScore = MetricsScore(
            metrics_config,
            used_frame=[int(self.frame_name)],
        )

        self.critical_object_filter_config = critical_object_filter_config

    def evaluate_frame(
        self
    ) -> None:
        """[summary]
        Evaluate a frame from the pair of estimated objects and ground truth objects
        Args:
            previous_result (Optional[PerceptionFrameResult]): The previous frame result. If None, set it as empty list []. Defaults to None.
        """


        # Divide objects by label to dict
        object_results_dict: Dict[LabelType, List[DynamicObjectWithPerceptionResult]] = divide_objects(
            self.object_results,
            self.critical_object_filter_config.target_labels,
        )

        num_ground_truth_dict: Dict[LabelType, int] = divide_objects_to_num(
            self.frame_ground_truth.objects,
            self.critical_object_filter_config.target_labels,
        )

        self.metrics_score.evaluate_detection(object_results_dict, num_ground_truth_dict)



# def get_object_status(frame_results: List[PerceptionFrameResult]) -> List[GroundTruthStatus]:
#     """Returns the number of TP/FP/TN/FN ratios per frame as tuple.

#     Args:
#         frame_results (List[PerceptionFrameResult]): List of frame results.

#     Returns:
#         List[GroundTruthStatus]: Sequence of matching status ratios for each GT.
#     """
#     status_infos: List[GroundTruthStatus] = []
#     for frame_result in frame_results:
#         frame_num: int = int(frame_result.frame_name)
#         # TP
#         for tp_object_result in frame_result.pass_fail_result.tp_object_results:
#             if tp_object_result.ground_truth_object.uuid not in status_infos:
#                 tp_status = GroundTruthStatus(tp_object_result.ground_truth_object.uuid)
#                 tp_status.add_status(MatchingStatus.TP, frame_num)
#                 status_infos.append(tp_status)
#             else:
#                 index = status_infos.index(tp_object_result.ground_truth_object.uuid)
#                 tp_status = status_infos[index]
#                 tp_status.add_status(MatchingStatus.TP, frame_num)
#         # FP
#         for fp_object_result in frame_result.pass_fail_result.fp_object_results:
#             if fp_object_result.ground_truth_object is None:
#                 continue
#             if fp_object_result.ground_truth_object.uuid not in status_infos:
#                 fp_status = GroundTruthStatus(fp_object_result.ground_truth_object.uuid)
#                 fp_status.add_status(MatchingStatus.FP, frame_num)
#                 status_infos.append(fp_status)
#             else:
#                 index = status_infos.index(fp_object_result.ground_truth_object.uuid)
#                 fp_status = status_infos[index]
#                 fp_status.add_status(MatchingStatus.FP, frame_num)
#         # TN
#         for tn_object in frame_result.pass_fail_result.tn_objects:
#             if tn_object.uuid not in status_infos:
#                 tn_status = GroundTruthStatus(tn_object.uuid)
#                 tn_status.add_status(MatchingStatus.TN, frame_num)
#                 status_infos.append(tn_status)
#             else:
#                 index = status_infos.index(tn_object.uuid)
#                 tn_status = status_infos[index]
#                 tn_status.add_status(MatchingStatus.TN, frame_num)

#         # FN
#         for fn_object in frame_result.pass_fail_result.fn_objects:
#             if fn_object.uuid not in status_infos:
#                 fn_status = GroundTruthStatus(fn_object.uuid)
#                 fn_status.add_status(MatchingStatus.FN, frame_num)
#                 status_infos.append(fn_status)
#             else:
#                 index = status_infos.index(fn_object.uuid)
#                 fn_status = status_infos[index]
#                 fn_status.add_status(MatchingStatus.FN, frame_num)

#     return status_infos
