import torch
import cv2
import numpy as np
import pyclipper
import torchvision.ops as ops

from shapely.geometry import Polygon

def binarize(probability_maps, threshold=0.7):
    binary_maps = (probability_maps > threshold).float()
    return binary_maps

def box_score_fast(bitmap, _box):
    h, w = bitmap.shape[:2]
    box = _box.copy()
    xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int32), 0, w - 1)
    xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int32), 0, w - 1)
    ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int32), 0, h - 1)
    ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int32), 0, h - 1)

    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    box[:, 0] = box[:, 0] - xmin
    box[:, 1] = box[:, 1] - ymin
    cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
    return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

def unclip(box, unclip_ratio=1.5):
    poly = Polygon(box)
    distance = poly.area * unclip_ratio / poly.length
    offset = pyclipper.PyclipperOffset()
    offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    try:
        expanded = np.array(offset.Execute(distance))
    except:
        expanded = np.array(sum(offset.Execute(distance),[]))
    return expanded


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

    index_1, index_2, index_3, index_4 = 0, 1, 2, 3
    if points[1][1] > points[0][1]:
        index_1 = 0
        index_4 = 1
    else:
        index_1 = 1
        index_4 = 0
    if points[3][1] > points[2][1]:
        index_2 = 2
        index_3 = 3
    else:
        index_2 = 3
        index_3 = 2

    box = [points[index_1], points[index_2],
            points[index_3], points[index_4]]
    return box, min(bounding_box[1])


def dilate_with_vatti_clipping(preds, binary_maps, dilation_factor=1.5):
    pred_boxes = []
    pred_maps = []

    for idx in range(binary_maps.size(0)):
        binary_map = binary_maps[idx, 0].cpu().numpy() 
        pred = preds[idx, 0].detach().cpu().numpy()

        # binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, np.ones((3,3)))

        contours, _ = cv2.findContours((binary_map*255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        pred_map = np.zeros_like(binary_map)
        bounding_boxes = []
        scores = []
        for contour in contours:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            score = box_score_fast(pred, points.reshape(-1, 2))
            if 0.7 > score: #box thresh hold
                continue

            if points.shape[0] > 2:
                box = unclip(points, unclip_ratio=dilation_factor)
            else:
                continue
            if len(box) != 0:
                box = box.reshape(-1, 2)
                _, sside = get_mini_boxes(box.reshape((-1, 1, 2)))
                if sside < 7: # filter min size
                    continue
                box[:, 0] = np.clip(box[:,0], 0, binary_map.shape[1])
                box[:, 1] = np.clip(box[:,1], 0, binary_map.shape[0])
                
                lx, ly = np.min(box, axis=0)
                rx, ry = np.max(box, axis=0)
    
                bounding_boxes.append([lx, ly, rx, ry])
                cv2.fillPoly(pred_map, box.reshape(1, -1, 2), 1)
                scores.append(score)

        if len(scores) > 0:
            tensor_bounding_boxes = torch.tensor(bounding_boxes,device=binary_maps.device).float()
            nms_boxes = ops.nms(tensor_bounding_boxes, torch.tensor(scores,device=binary_maps.device), 0.1)

            pred_boxes.append(tensor_bounding_boxes[nms_boxes])
            pred_maps.append(torch.tensor(pred_map))

        else:
            pred_boxes.append(torch.tensor(bounding_boxes,device=binary_maps.device))
            pred_maps.append(torch.tensor(pred_map))

    return pred_boxes, torch.stack(pred_maps)

def process_text_regions(probability_maps, threshold=0.7, dilation_factor=1.5):
    binary_maps = binarize(probability_maps, threshold)
    pred_boxes = dilate_with_vatti_clipping(probability_maps, binary_maps, dilation_factor)

    return pred_boxes

