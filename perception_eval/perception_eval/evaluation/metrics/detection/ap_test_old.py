
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
        object_results: List[List[DynamicObjectWithPerceptionResult]],
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

        # 按照 sample_token 分組（模仿 nuScenes 的 gt_boxes[pred_box.sample_token]）
        gt_by_token: Dict[str, List[DynamicObjectWithPerceptionResult]] = defaultdict(list)
        for obj in object_results:
            gt = obj.ground_truth_object
            if gt is not None:
                gt_by_token[gt.frame_id].append(obj)  # ← 這裡請根據你 Dataset 的實際唯一 frame token 修改！

        matched_gt = set()

        for pred_obj in predictions:
            pred = pred_obj.estimated_object
            pred_token = pred.frame_id  # ← 改成你 prediction 的唯一 frame ID（對應 sample_token）
            pred_label = pred.semantic_label
            gt_list = gt_by_token.get(pred_token, [])

            best_dist = float("inf")
            best_match = None

            for gt_idx, gt_obj in enumerate(gt_list):
                gt = gt_obj.ground_truth_object
                if gt.semantic_label != pred_label:
                    continue
                key = (pred_token, gt_idx)
                if key in matched_gt:
                    continue

                dist = self._get_matching_distance(gt, pred)
                if dist < best_dist:
                    best_dist = dist
                    best_match = key

            threshold = get_label_threshold(pred_label, self.target_labels, self.matching_threshold_list)
            is_match = best_match is not None and best_dist < threshold

            self.tp_list.append(1 if is_match else 0)
            self.fp_list.append(0 if is_match else 1)
            self.conf_list.append(pred.semantic_score)

            if is_match:
                matched_gt.add(best_match)

        self.tp_list = np.cumsum(self.tp_list).tolist()
        self.fp_list = np.cumsum(self.fp_list).tolist()

        missing_tp = self.num_ground_truth - self.tp_list[-1] if self.tp_list else self.num_ground_truth
        for _ in range(missing_tp):
            self.tp_list.append(self.tp_list[-1] + 1 if self.tp_list else 1)
            self.fp_list.append(self.fp_list[-1] if self.fp_list else 0)
            self.conf_list.append(0.0)  # 加入最低的 confidence

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

    # def _calculate_ap_nusc_style(
    #     self,
    #     precision_list: List[float],
    #     recall_list: List[float],
    #     min_recall: float,
    #     min_precision: float,
    # ) -> float:
    #     if len(precision_list) == 0:
    #         return 0.0

    #     # Step 1: 將 recall 去重並保留最早出現的 precision
    #     recall_array, indices = np.unique(recall_list, return_index=True)
    #     precision_array = np.array(precision_list)[indices]

    #     # Step 2: 插值到固定 recall bins（與 nuScenes 相同）
    #     recall_interp = np.linspace(0.0, 1.0, 101)
    #     precision_interp = np.interp(recall_interp, recall_array, precision_array, right=0)

    #     # Step 3: Clip precision 低於 min_precision 的部分
    #     precision_interp -= min_precision
    #     precision_interp = np.clip(precision_interp, 0.0, 1.0)

    #     # Step 4: 計算 AP
    #     valid = recall_interp >= min_recall
    #     if not np.any(valid):
    #         return 0.0

    #     return float(np.mean(precision_interp[valid]) / (1.0 - min_precision))

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
        first_ind = int(round(100 * min_recall)) + 1  # skip recall <= min_recall
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

        # Step 1: unique recall
        recall_array, indices = np.unique(recall_list, return_index=True)
        precision_array = np.array(precision_list)[indices]
        conf_array = np.array(self.conf_list)[indices]

        # Step 2: interpolate recall and precision to 101 bins
        recall_interp = np.linspace(0.0, 1.0, 101)
        precision_interp = np.interp(recall_interp, recall_array, precision_array, right=0)
        conf_interp = np.interp(recall_interp, recall_array, conf_array, right=0)

        print(f"Max Recall: {max(recall_list):.3f}")
        print(f"AP: {self.ap:.3f}")
        print(f"Precision: {np.round(precision_interp, 8)}")
        print(f"Recall: {np.round(recall_interp, 2)}")
        print(f"Confidences: {np.round(conf_interp, 8)}")

        # Optional plot
        plt.figure(figsize=(6, 4))
        plt.plot(recall_interp, precision_interp, label="PR Curve", marker=".")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (debug)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()