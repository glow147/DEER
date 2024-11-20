import os
import torch
from torch.utils.data import Dataset

import pyclipper

import cv2
import numpy as np
from pycocotools.coco import COCO
from shapely.geometry import Polygon

class ICDAR_Dataset(Dataset):
    def __init__(self, config, transform, is_train=True):
        self.config = config
        self.shrink_ratio = 0.4
        self.thresh_min = 0.3 
        self.thresh_max = 0.7
        self.transform = transform

        if is_train:
            self.anno = COCO(self.config.DATA.TRAIN_LABEL)
            self.img_dir = self.config.DATA.TRAIN_IMG_DIR
        else:
            self.anno = COCO(self.config.DATA.VALID_LABEL)
            self.img_dir = self.config.DATA.VALID_IMG_DIR

        self.image_ids = list(self.anno.imgs.keys())
        self.is_train = is_train

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # image_id와 image_info 가져오기
        image_id = self.image_ids[idx]
        image_info = self.anno.loadImgs(image_id)[0]

        img_path = os.path.join(self.img_dir, image_info['file_name'])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 어노테이션 가져오기
        annotation_ids = self.anno.getAnnIds(imgIds=image_id)
        annotations = self.anno.loadAnns(annotation_ids)

        if len(annotations) == 0:
            return None

        file_name, height, width = image_info['file_name'], image_info['height'], image_info['width']
        boxes, labels, gts, masks, thresh_map, thresh_mask = self.get_annotation(annotations, height, width)

        try:
            transformed = self.transform(
                image=img,
                bboxes=np.array(boxes),
                category_ids=labels,
                masks=[gts, masks, thresh_map, thresh_mask]
            )
            
            if len(transformed['category_ids']) > self.config.DATA.SAMPLE_TEXT:
                sample_idx = np.random.choice(len(transformed['category_ids']), self.config.DATA.SAMPLE_TEXT, replace=False)
            else:
                sample_idx = range(len(transformed['category_ids'])
            )
        except Exception as e:
            print(e)
            return self.__getitem__((idx + 1) % len(self))

        gts, masks, thresh_map, thresh_mask = transformed['masks']
        
        gts = torch.clip(gts,0,1)
        masks = torch.clip(masks,0,1)
        thresh_map = torch.clip(thresh_map,0,1)
        thresh_mask = torch.clip(thresh_mask,0,1)       
       
        prefs_noise = self.calculate_reference_point(
            [transformed['bboxes'][idx] for idx in sample_idx],
            transformed['image'].shape[0], # height
            transformed['image'].shape[1] # width
        )

        if self.is_train:
            sample = {
                'filename': file_name,
                'ori_size': (width, height),
                'image': transformed['image'],
                'bboxes': torch.tensor(transformed['bboxes'], dtype=torch.float32),
                'gts': gts.unsqueeze(0),
                'masks': masks,
                'thresh_mask': thresh_map,
                'thresh_map': thresh_mask,
                'pref_noise': prefs_noise.float() / self.config.DATA.IMAGE_SIZE,
                'label': [transformed['category_ids'][idx] for idx in sample_idx]
            }

        else:
            sample = {
                'filename': file_name,
                'ori_size': (width, height),
                'image': transformed['image'],
                'bboxes': torch.tensor(transformed['bboxes'], dtype=torch.float32),
                'label': transformed['category_ids'],
            }

        return sample
        
    def get_annotation(self, annotations, height, width):
        bboxes = []
        labels = []
        gt = np.zeros((height,width), dtype=np.float32)
        mask = np.ones((height,width), dtype=np.float32)
        thresh_map = np.zeros((height,width), dtype=np.float32)
        thresh_mask = np.zeros((height,width), dtype=np.float32)
    
        for ann in annotations:
            label = ann['text']
            x,y,w,h = [v for v in ann['bbox']]
            if x+w <= x or y+h <= y or x < 0 or y < 0:
                continue
            bboxes.append([x,y,w,h])
            labels.append(label)
          
            self.ann_to_mask([np.array(ann['segmentation']).reshape(4,2)], height, width, gt, mask, thresh_map, thresh_mask)
    
        thresh_map = thresh_map * (self.thresh_max - self.thresh_min) + self.thresh_min

        return bboxes, labels, gt, mask, thresh_map, thresh_mask
    
    def generate_thresh_map_and_mask(self, canvas, mask, polygon) :   
        self.draw_border_map(polygon, canvas, mask=mask)
        thresh_map = canvas
        thresh_mask = mask
        
        return thresh_map,thresh_mask

    def draw_border_map(self, polygon, padded_polygon, distance, canvas, mask):
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs, ys = np.meshgrid(np.arange(width), np.arange(height))


        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def ann_to_mask(self, segmentation, height, width, gt, mask, thresh_map, thresh_mask):
        for i, polygon in enumerate(segmentation):
            polygon = polygon.reshape((-1, 2)).astype(np.int32)
            polygon_shape = Polygon(polygon)
            distance = polygon_shape.area * \
                (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
            subject = polygon.tolist()
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND,
                            pyclipper.ET_CLOSEDPOLYGON)
            # gt / mask
            shrinked = padding.Execute(-distance)
            if shrinked == []:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                continue
            shrinked = np.array(shrinked[0]).reshape(-1, 2)
            cv2.fillPoly(gt, [shrinked.astype(np.int32)], 1)
            # thresh_gt / mask
            padded_polygon = np.array(padding.Execute(distance)[0])
            self.draw_border_map(polygon, padded_polygon, distance, thresh_map, thresh_mask)

    def calculate_reference_point(self, boxes, height, width):# -> Any:
        prefs_noise = []
        for bbox in boxes:
            x, y, w, h = bbox
            p_tl = torch.tensor([x, y], dtype=torch.float32)
            p_tr = torch.tensor([x + w, y], dtype=torch.float32)
            p_bl = torch.tensor([x, y + h], dtype=torch.float32)
            
            # points = torch.tensor(segmentation, dtype=torch.float32).view(-1, 2)
    
            p_c = torch.tensor([x + (w / 2), y + (h / 2)])
      
            d_tl_tr = torch.norm(p_tl - p_tr)
            d_tl_bl = torch.norm(p_tl - p_bl)
    
            n = torch.distributions.Uniform(-1, 1).sample()
            p_ref = p_c + 0.5 * n * min(d_tl_tr, d_tl_bl)
        
            # clamp
            p_ref[0] = torch.clamp(p_ref[0], min=0, max=width-1)
            p_ref[1] = torch.clamp(p_ref[1], min=0, max=height-1)
            prefs_noise.append(p_ref)
                
        return torch.stack(prefs_noise)

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2)+1e-6)
        square_sin = 1 - np.square(cosin)
        
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                             square_sin / (square_distance + 1e-6)+1e-6)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        return result
