
from typing import Callable, List, Optional, Tuple, Union, Dict
import numpy as np
import matplotlib.pyplot as plt
from perception_eval.common.label import LabelType
from perception_eval.common.threshold import get_label_threshold
from perception_eval.evaluation.matching import MatchingMode
from perception_eval.evaluation.metrics.detection.tp_metrics import TPMetricsAp, TPMetricsAph
from perception_eval.evaluation.result.object_result import DynamicObjectWithPerceptionResult


class Ap:
    def __init__(
        self,
        tp_metrics: Union[TPMetricsAp, TPMetricsAph],
        object_results: List[DynamicObjectWithPerceptionResult],
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

        all_object_results: List[DynamicObjectWithPerceptionResult] = []
        if len(object_results) == 0 or not isinstance(object_results[0], list):
            all_object_results = object_results
        else:
            for obj_results in object_results:
                all_object_results += obj_results
        self.objects_results_num: int = len(all_object_results)


        self._calculate_tp_fp_nusc_style(all_object_results)
        precision, recall = self.get_precision_recall_list()
        self.ap = self._calculate_ap_nusc_style(precision, recall, min_recall=0.1, min_precision=0.1)

        self._debug_ap(precision, recall)

    def _calculate_tp_fp_nusc_style(self, object_results: List[DynamicObjectWithPerceptionResult]) -> None:
        from collections import defaultdict

        predictions = [obj for obj in object_results if obj.estimated_object is not None]
        predictions.sort(key=lambda x: x.estimated_object.semantic_score, reverse=True)

        # 以 unix_time 分組 ground truth（模仿 sample_token 分組）
        gt_by_time: Dict[str, List[DynamicObjectWithPerceptionResult]] = defaultdict(list)
        for obj in object_results:
            gt = obj.ground_truth_object
            if gt is not None:
                token = str(gt.unix_time)  # ← 確保是字串類型的 key
                gt_by_time[token].append(obj)

        matched_gt = set()

        for pred_idx, pred_obj in enumerate(predictions):
            pred = pred_obj.estimated_object
            pred_time = str(pred.unix_time)  # ← 與 GT 的 token 對應
            pred_label = pred.semantic_label
            gt_list = gt_by_time.get(pred_time, [])

            best_dist = float("inf")
            best_match_key = None

            for gt_obj in gt_list:
                gt = gt_obj.ground_truth_object
                if gt.semantic_label != pred_label:
                    continue

                key = id(gt)
                if key in matched_gt:
                    continue

                dist = self._get_matching_distance(gt, pred)
                if dist < best_dist:
                    best_dist = dist
                    best_match_key = key

            threshold = get_label_threshold(pred_label, self.target_labels, self.matching_threshold_list)
            is_match = best_match_key is not None and best_dist < threshold

            print(f"[DEBUG MATCH] #{pred_idx}: unix_time={pred_time}, conf={pred.semantic_score:.3f}, dist={best_dist:.3f}, threshold={threshold:.3f}, match={is_match}")

            self.tp_list.append(1 if is_match else 0)
            self.fp_list.append(0 if is_match else 1)
            self.conf_list.append(pred.semantic_score)

            if is_match:
                matched_gt.add(best_match_key)

        self.tp_list = np.cumsum(self.tp_list).tolist()
        self.fp_list = np.cumsum(self.fp_list).tolist()

        if self.tp_list:
            print(f"[DEBUG SUMMARY] TP: {self.tp_list[-1]}, FP: {self.fp_list[-1]}, GT: {self.num_ground_truth}")
        else:
            print("[DEBUG SUMMARY] No predictions matched at all — tp_list is empty.")



    def _get_matching_distance(self, gt_obj, pred_obj) -> float:
        gt_pos = np.array(gt_obj.state.position[:2])
        pred_pos = np.array(pred_obj.state.position[:2])
        return np.linalg.norm(gt_pos - pred_pos)

    def get_precision_recall_list(self) -> Tuple[List[float], List[float]]:
        precision = []
        recall = []
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

        recall_array, indices = np.unique(recall_list, return_index=True)
        precision_array = np.array(precision_list)[indices]

        # Interpolate to 101 recall points
        recall_interp = np.linspace(0.0, 1.0, 101)
        precision_interp = np.interp(recall_interp, recall_array, precision_array, right=0)

        # Apply nuScenes filtering logic
        first_ind = int(round(100 * min_recall)) + 1
        filtered_prec = precision_interp[first_ind:] - min_precision
        filtered_prec[filtered_prec < 0] = 0.0

        if len(filtered_prec) == 0:
            return 0.0
        return float(np.mean(filtered_prec)) / (1.0 - min_precision)

    def _debug_ap(self, precision_list: List[float], recall_list: List[float]) -> None:
        print("\n[DEBUG] ---- AP Debug Information ----")
        print(f"# Predictions: {self.objects_results_num}")
        print(f"# GT: {self.num_ground_truth}")

        if len(precision_list) == 0 or len(recall_list) == 0:
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

        plt.figure(figsize=(6, 4))
        plt.plot(recall_interp, precision_interp, label="PR Curve", marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (debug)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


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
