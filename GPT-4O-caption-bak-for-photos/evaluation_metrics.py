import numpy as np
from scipy.optimize import linear_sum_assignment

class EvaluationMetrics:
    def __init__(self, pred_points, gt_boxes_yolo, gt_points, img_width, img_height):
        self.pred_points = np.array(pred_points)
        self.gt_boxes_yolo = gt_boxes_yolo
        self.gt_points = np.array(gt_points)
        self.img_width = img_width
        self.img_height = img_height
        self.gt_boxes = [self.yolo_to_bbox(box) for box in gt_boxes_yolo]

    def yolo_to_bbox(self, yolo_box):
        x_center, y_center, width, height = yolo_box
        x_min = (x_center - width / 2) * self.img_width
        y_min = (y_center - height / 2) * self.img_height
        x_max = (x_center + width / 2) * self.img_width
        y_max = (y_center + height / 2) * self.img_height
        return [x_min, y_min, x_max, y_max]

    def calculate_coarse_precision_and_recall(self):
        """
        coarse_precision, recall
        """
        matched_boxes = set()
        matched_points = 0

        for point in self.pred_points:
            x, y = point
            for idx, box in enumerate(self.gt_boxes):
                if idx in matched_boxes:
                    continue
                x_min, y_min, x_max, y_max = box

                if x_min <= x <= x_max and y_min <= y <= y_max:
                    matched_points += 1
                    matched_boxes.add(idx)
                    break

        # 计算指标
        coarse_precision = matched_points / len(self.pred_points) if len(self.pred_points) > 0 else 0.0
        recall = len(matched_boxes) / len(self.gt_boxes) if len(self.gt_boxes) > 0 else 0.0

        return coarse_precision, recall

    def calculate_absolute_error(self):
        """
        :return: average_absolute_error, matches
        """
        num_pred = len(self.pred_points)
        num_gt = len(self.gt_points)
        cost_matrix = np.zeros((num_pred, num_gt))

        for i, pred in enumerate(self.pred_points):
            for j, gt in enumerate(self.gt_points):
                cost_matrix[i, j] = np.linalg.norm(pred - gt)  #

        row_indices, col_indices = linear_sum_assignment(cost_matrix)


        total_error = 0
        matches = []
        for row, col in zip(row_indices, col_indices):
            error = cost_matrix[row, col]
            total_error += error
            matches.append((self.pred_points[row], self.gt_points[col]))

        average_absolute_error = total_error / len(matches) if matches else 0.0
        return average_absolute_error, matches

    def evaluate(self):
        """
        :return: coarse_precision, recall, average_absolute_error
        """
        coarse_precision, recall = self.calculate_coarse_precision_and_recall()
        average_absolute_error, matches = self.calculate_absolute_error()
        return coarse_precision, recall, average_absolute_error, matches
