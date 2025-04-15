from collections import defaultdict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from perception_eval.common import DynamicObject
from perception_eval.common.label import LabelType
from perception_eval.common.threshold import get_label_threshold
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAph
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class Ap:
    def __init__(
        self,
        tp_metrics: Union[TPMetricsAp, TPMetricsAph],
        estimated_objects: List[DynamicObject],
        ground_truth_objects: List[DynamicObject],
        num_ground_truth: int,
        target_labels: List[LabelType],
        matching_mode: MatchingMode,
        matching_threshold_list: List[float],
    ) -> None:
        self.tp_metrics = tp_metrics
        self.num_ground_truth = num_ground_truth
        self.target_labels = target_labels
        self.matching_mode = matching_mode
        self.matching_threshold_list = matching_threshold_list

        self.tp_list: List[int] = []
        self.fp_list: List[int] = []
        self.conf_list: List[float] = []

        self.final = False

        all_estimated_objects: List[DynamicObject] = []
        if len(estimated_objects) == 0 or not isinstance(estimated_objects[0], list):
            all_estimated_objects = estimated_objects
        else:
            self.final = True
            for estimated_objects_sub in estimated_objects:
                all_estimated_objects += estimated_objects_sub

        all_ground_truth_objects: List[DynamicObject] = []
        if len(ground_truth_objects) == 0 or not isinstance(ground_truth_objects[0], list):
            all_ground_truth_objects = ground_truth_objects
        else:
            for ground_truth_objects_sub in ground_truth_objects:
                all_ground_truth_objects += ground_truth_objects_sub

        self._calculate_tp_fp_nusc_style(all_estimated_objects, all_ground_truth_objects)
        precision, recall = self.get_precision_recall_list()
        self.ap = self._calculate_ap_nusc_style(precision, recall, min_recall=0.1, min_precision=0.1)

        self._debug_ap(precision, recall)

    def _calculate_tp_fp_nusc_style(self, preds: List[DynamicObject], gts: List[DynamicObject]) -> None:
        preds = sorted(preds, key=lambda x: x.semantic_score, reverse=True)
        gt_by_time: Dict[str, List[DynamicObject]] = defaultdict(list)

        for gt in gts:
            token = f"{gt.unix_time:.1f}"
            gt_by_time[token].append(gt)

        matched_gt_ids = set()

        for pred_idx, pred in enumerate(preds):
            token = f"{pred.unix_time:.1f}"
            pred_label = pred.semantic_label
            candidates = gt_by_time.get(token, [])

            best_dist = float("inf")
            best_match_idx = None
            best_match_gt = None

            for gt_idx, gt in enumerate(candidates):
                if gt.semantic_label != pred_label:
                    continue
                match_key = (token, gt_idx)
                if match_key in matched_gt_ids:
                    continue

                dist = np.linalg.norm(np.array(gt.state.position[:2]) - np.array(pred.state.position[:2]))
                if dist < best_dist:
                    best_dist = dist
                    best_match_idx = gt_idx
                    best_match_gt = gt

            threshold = get_label_threshold(pred_label, self.target_labels, self.matching_threshold_list)
            is_match = best_match_idx is not None and best_dist < threshold

            # 🧠 Debug info for this prediction
            if best_match_gt is not None:
                gt_pos = best_match_gt.state.position
            else:
                gt_pos = None
            pred_pos = pred.state.position

            if self.final:
                print(
                    f"[MATCH] #{pred_idx:04d} unix_time={token}, label={pred_label}, score={pred.semantic_score:.3f}, "
                    f"min_dist={best_dist:.5f}, threshold={threshold:.3f}, match={'Yes' if is_match else 'No'}"
                )
                print(f"         PRED_POS={pred_pos[:3]}, GT_POS={gt_pos[:3] if gt_pos else 'N/A'}")

            self.tp_list.append(1 if is_match else 0)
            self.fp_list.append(0 if is_match else 1)
            self.conf_list.append(pred.semantic_score)

            if is_match:
                matched_gt_ids.add((token, best_match_idx))

        self.tp_list = np.cumsum(self.tp_list).tolist()
        self.fp_list = np.cumsum(self.fp_list).tolist()

        # 📊 Overall summary
        for i, (tp, fp, conf) in enumerate(zip(self.tp_list, self.fp_list, self.conf_list)):
            print(f"[#{i:04d}] TP={tp}, FP={fp}, conf={conf:.3f}")

        if self.final:
            if self.tp_list:
                print(f"[DEBUG SUMMARY] TP: {self.tp_list[-1]}, FP: {self.fp_list[-1]}, GT: {self.num_ground_truth}")
            else:
                print("[DEBUG SUMMARY] No predictions matched at all — tp_list is empty.")

        self.objects_results_num = sum(self.tp_list)

    def get_precision_recall_list(self) -> Tuple[List[float], List[float]]:
        precision, recall = [], []
        for i in range(len(self.tp_list)):
            precision.append(self.tp_list[i] / (self.tp_list[i] + self.fp_list[i]))
            recall.append(self.tp_list[i] / self.num_ground_truth if self.num_ground_truth > 0 else 0.0)
        return precision, recall

    def _calculate_ap_nusc_style(
        self,
        precision_list: List[float],
        recall_list: List[float],
        min_recall: float,
        min_precision: float,
    ) -> float:
        if len(precision_list) == 0:
            return 0.0

        tp = np.array(self.tp_list, dtype=np.float32)
        fp = np.array(self.fp_list, dtype=np.float32)
        precision = tp / (tp + fp)
        recall = tp / float(self.num_ground_truth)

        # Step 1: Envelope curve (right-to-left max)
        precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]

        # Step 2: 101-point interpolation
        recall_interp = np.linspace(0.0, 1.0, 101)
        precision_interp = np.interp(recall_interp, recall, precision_envelope, right=0)

        # Step 3: Filter and compute AP
        first_ind = int(round(100 * min_recall)) + 1
        filtered_prec = precision_interp[first_ind:] - min_precision
        filtered_prec[filtered_prec < 0] = 0.0

        return float(np.mean(filtered_prec)) / (1.0 - min_precision)

    def _debug_ap(self, precision_list: List[float], recall_list: List[float]) -> None:
        print("\n[DEBUG] ---- AP Debug Information ----")
        print(f"# Predictions: {len(self.conf_list)}")
        print(f"# GT: {self.num_ground_truth}")

        # ➕ 新增 TP / FP summary
        tp_sum = int(self.tp_list[-1]) if self.tp_list else 0
        fp_sum = int(self.fp_list[-1]) if self.fp_list else 0
        print(f"# True Positives (TP):      {tp_sum}")
        print(f"# False Positives (FP):     {fp_sum}")

        if not precision_list or not recall_list:
            print("[DEBUG] Skipping interpolation: precision or recall list is empty.")
            return

        recall_array, indices = np.unique(recall_list, return_index=True)
        precision_array = np.array(precision_list)[indices]
        conf_array = np.array(self.conf_list)[indices]

        recall_interp = np.linspace(0.0, 1.0, 101)
        precision_interp = np.interp(recall_interp, recall_array, precision_array, right=0)
        conf_interp = np.interp(recall_interp, recall_array, conf_array, right=0)

        print(f"Max Recall: {max(recall_list):.3f}")
        print(f"AP: {self.ap:.3f}")
        print(f"Precision: {np.round(precision_interp, 8)}")
        print(f"Recall: {np.round(recall_interp, 2)}")
        print(f"Confidences: {np.round(conf_interp, 8)}")

    def interpolate_precision_recall_list(
        self,
        precision_list: List[float],
        recall_list: List[float],
    ):
        """[summary]
        Interpolate precision and recall with maximum precision value per recall bins.
        Args:
            precision_list (List[float])
            recall_list (List[float])
        """
        max_precision_list: List[float] = [precision_list[-1]]
        max_precision_recall_list: List[float] = [recall_list[-1]]

        for i in reversed(range(len(recall_list) - 1)):
            if precision_list[i] > max_precision_list[-1]:
                max_precision_list.append(precision_list[i])
                max_precision_recall_list.append(recall_list[i])

        # append min recall
        max_precision_list.append(max_precision_list[-1])
        max_precision_recall_list.append(0.0)

        return max_precision_list, max_precision_recall_list

    @staticmethod
    def _calculate_average_sd(
        object_results: List[DynamicObjectWithPerceptionResult],
        matching_mode: MatchingMode,
    ) -> Tuple[Optional[float], Optional[float]]:
        """[summary]
        Calculate average and standard deviation.
        Args:
            object_results (List[DynamicObjectWithPerceptionResult]): The object results
            matching_mode (MatchingMode): [description]
        Returns:
            Tuple[float, float]: [description]
        """

        matching_score_list: List[float] = [
            object_result.get_matching(matching_mode).value for object_result in object_results
        ]
        matching_score_list_without_none = list(filter(lambda x: x is not None, matching_score_list))
        if len(matching_score_list_without_none) == 0:
            return None, None
        mean: float = np.mean(matching_score_list_without_none).item()
        standard_deviation: float = np.std(matching_score_list_without_none).item()
        return mean, standard_deviation

    @staticmethod
    def _get_flat_str(str_list: List[str]) -> str:
        """
        Example:
            a = _get_flat_str([aaa, bbb, ccc])
            print(a) # aaa_bbb_ccc
        """
        output = ""
        for one_str in str_list:
            output = f"{output}_{one_str}"
        return output

    def save_precision_recall_graph(
        self,
        result_directory: str,
        frame_name: str,
    ) -> None:
        """[summary]
        Save visualization image of precision and recall curve.
        The circle points represent original values and the square points represent interpolated ones.
        Args:
            result_directory (str): The directory path to save images.
            frame_name (str): The frame name.
        """

        base_name = f"{frame_name}_pr_curve_{self._get_flat_str(self.matching_threshold_list)}_"
        target_str = f"{self._get_flat_str(self.target_labels)}"
        file_name = base_name + target_str + ".png"
        file_path = osp.join(result_directory, file_name)

        precision_list: List[float] = []
        recall_list: List[float] = []
        precision_list, recall_list = self.get_precision_recall_list()
        max_precision_list, max_precision_recall_list = self.interpolate_precision_recall_list(
            precision_list, recall_list
        )
        # plot original values
        plt.plot(
            recall_list,
            precision_list,
            label="original",
            marker="o",
            color=(1, 0, 0, 0.3),
        )
        # plot interpolated values
        plt.plot(
            max_precision_recall_list,
            max_precision_list,
            label="interpolate",
            marker="s",
            color=(1, 0, 0),
        )
        plt.title("PR-curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig(file_path)
