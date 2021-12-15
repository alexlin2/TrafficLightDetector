import torch 
import numpy as np
import cv2 
from model import trafficLightDetectionModel
import dataset

img_dir = "/home/alexlin/traffic_net/dataset_train_rgb/"
labels_dir = "/home/alexlin/traffic_net/dataset_train_rgb/train.yaml"

def accurracy(gt, pred):
    print(gt)
    prob_threshold = .5
    IOU_threshold = .5
    true_Positive_Counter = 0
    false_Positive_Counter = 0
    false_Negative_Counter = 0
    for i, box in enumerate(pred["boxes"]):
        if pred["scores"][i] > prob_threshold:
            pred_x_max = box[0]
            pred_x_min = box[1]
            pred_y_max = box[2]
            pred_y_min = box[3]
            for j, box_gt in enumerate(gt["boxes"]):
                gt_x_max = box_gt[0]
                gt_x_min = box_gt[1]
                gt_y_max = box_gt[2]
                gt_y_min = box_gt[3]

                xA = max(pred_x_min, gt_x_min)
                yA = max(pred_y_min, gt_y_min)
                xB = min(pred_x_max, gt_x_max)
                yB = min(pred_y_max, gt_y_max)

                interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

                predArea =  (pred_x_max - pred_x_min + 1) * (pred_y_max - pred_y_min + 1)
                gtArea = (gt_x_max - gt_x_min + 1) * (gt_y_max - gt_y_min + 1)

                iou = interArea / (predArea + gtArea - interArea)


                if iou > IOU_threshold:
                    if pred["labels"][i] == gt["labels"][j]:
                        true_Positive_Counter += 1
                    else :
                        false_Positive_Counter += 1
    
    tag = False
    if true_Positive_Counter + false_Positive_Counter == 0:
        precision_score = 0
    else :
        precision_score = true_Positive_Counter / (true_Positive_Counter + false_Positive_Counter)
    if true_Positive_Counter + false_Negative_Counter == 0:
        recall_score = 0
    else :
        recall_score = true_Positive_Counter / (true_Positive_Counter + false_Negative_Counter)
    if precision_score + recall_score == 0:
        F1_score = precision_score * recall_score / (precision_score + recall_score)

    return precision_score, recall_score, F1_score

if __name__ == '__main__':
    data = dataset.TrafficLightDataset(img_dir, labels_dir, ['background', 'GreenLeft', 'RedStraightLeft', 'RedLeft', 'off', 'GreenStraight', 'GreenStraightRight',
             'GreenStraightLeft', 'RedStraight', 'GreenRight', 'Green', 'Yellow', 'RedRight', 'Red'])
    gt = data[3][1]
    print(gt)
