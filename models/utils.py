import os
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import _LRScheduler
import math
from collections import namedtuple
import numpy as np
from shapely.geometry import Polygon
import cv2

class DetectionIoUEvaluator(object):
    def __init__(self, iou_constraint=0.5, area_precision_constraint=0.5):
        self.iou_constraint = iou_constraint
        self.area_precision_constraint = area_precision_constraint

    def evaluate_image(self, gt, pred):

        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        perSampleMetrics = {}

        gtPols = []
        detPols = []
        gtDontCarePolsNum = []
        detDontCarePolsNum = []
        
        matched_gt_texts = 0  # 검출된 실제 텍스트 수
        correct_text_matches = 0  # 텍스트가 일치하는 경우의 수

        # 실제 다각형 정보 설정
        for n in range(len(gt)):
            points = gt[n]['points']
            dontCare = gt[n]['ignore']

            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            gtPols.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)

        # 예측 다각형 정보 설정
        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid or not Polygon(points).is_simple:
                continue

            detPols.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, points)
                    pdDimensions = Polygon(points).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if precision > self.area_precision_constraint:
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break

        # IoU 및 텍스트 일치도 계산
        if len(gtPols) > 0 and len(detPols) > 0:
            iouMat = np.empty([len(gtPols), len(detPols)])
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)

            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if (
                        gtRectMat[gtNum] == 0
                        and detRectMat[detNum] == 0
                        and gtNum not in gtDontCarePolsNum
                        and detNum not in detDontCarePolsNum
                    ):
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            matched_gt_texts += 1  # 실제 텍스트 검출 성공

                            # 텍스트가 일치하는 경우
                            if gt[gtNum].get('text') == pred[detNum].get('text'):
                                correct_text_matches += 1

        numGtCare = len(gtPols) - len(gtDontCarePolsNum)
        numDetCare = len(detPols) - len(detDontCarePolsNum)
        
        # Recall: 실제 텍스트 중에서 검출된 비율
        recall = float(matched_gt_texts) / numGtCare if numGtCare > 0 else 1.0
        # Precision: 검출한 텍스트 중에서 실제와 일치하는 비율
        precision = float(correct_text_matches) / numDetCare if numDetCare > 0 else 1.0
        hmean = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        perSampleMetrics = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'matched_texts': matched_gt_texts,
            'correct_text_matches': correct_text_matches,
        }

        return perSampleMetrics



class CustomCosineAnnealingWarmupRestarts(_LRScheduler):
    """
        src: https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup/blob/master/cosine_annealing_warmup/scheduler.py

        optimizer (Optimizer): Wrapped optimizer.
        first_cycle_steps (int): First cycle step size.
        cycle_mult(float): Cycle steps magnification. Default: -1.
        max_lr(float): First cycle's max learning rate. Default: 0.1.
        min_lr(float): Min learning rate. Default: 0.001.
        warmup_steps(int): Linear warmup step size. Default: 0.
        gamma(float): Decrease rate of max learning rate by cycle. Default: 1.
        last_epoch (int): The index of last epoch. Default: -1.
    """

    def __init__(self,
                 optimizer : torch.optim.Optimizer,
                 first_cycle_steps : int,
                 cycle_mult : float = 1.,
                 max_lr : float = 0.1,
                 min_lr : float = 0.001,
                 warmup_steps : int = 0,
                 gamma : float = 1.,
                 last_epoch : int = -1
        ):
        assert warmup_steps < first_cycle_steps

        self.first_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle_mult = cycle_mult # cycle steps magnification
        self.base_max_lr = max_lr # first max learning rate
        self.max_lr = max_lr # max learning rate in the current cycle
        self.min_lr = min_lr # min learning rate
        self.warmup_steps = warmup_steps # warmup step size
        self.gamma = gamma # decrease rate of max learning rate by cycle

        self.cur_cycle_steps = first_cycle_steps # first cycle step size
        self.cycle = 0 # cycle count
        self.step_in_cycle = last_epoch # step size of the current cycle

        super(CustomCosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)

        # set learning rate min_lr
        self.init_lr()

    def init_lr(self):
        self.base_lrs = []
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
            self.base_lrs.append(self.min_lr)

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - base_lr)*self.step_in_cycle / self.warmup_steps + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.max_lr - base_lr) \
                    * (1 + math.cos(math.pi * (self.step_in_cycle-self.warmup_steps) \
                                    / (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(math.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

def save_overlayed_dilated_maps(images, dilated_maps, directory="./visualize"):
    """
    원본 이미지 위에 dilated_maps를 겹쳐서 보여주고 저장합니다.
    
    :param images: [batch_size, channels, height, width] 형태의 원본 이미지 텐서
    :param dilated_maps: [batch_size, 1, height, width] 형태의 dilated map 텐서
    :param directory: 이미지를 저장할 디렉터리 경로
    """
    # 디렉터리가 존재하지 않으면 생성
    if not os.path.exists(directory):
        os.makedirs(directory)

    batch_size = images.size(0)
    for i in range(batch_size):
        fig, ax = plt.subplots(figsize=(8, 8))

        # 원본 이미지 처리
        image = images[i].permute(1, 2, 0).cpu().numpy()  # CHW -> HWC
        image = (image - image.min()) / (image.max() - image.min())  # 정규화
        
        # Dilated map 처리
        dilated_map = dilated_maps[i, 0].cpu().numpy()

        # 원본 이미지 표시
        ax.imshow(image, cmap='gray')
        # Dilated map 투명하게 겹치기
        ax.imshow(dilated_map, cmap='jet', alpha=0.5)  # 'jet'는 임의의 컬러맵

        ax.axis('off')  # 축 레이블 제거
        plt.title(f'Overlayed Image {i + 1}')
        
        # 이미지 파일로 저장
        plt.savefig(os.path.join(directory, f'overlayed_image_{i + 1}.png'))
        plt.close()
        return 

def draw_pred_text(filename, image, center_points, words):
    output_folder = 'test_image'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)


    mean = torch.tensor([0.485, 0.456, 0.406]) 
    std = torch.tensor([0.229, 0.224, 0.225]) 

    # 정규화 해제 함수 정의
    def denormalize(img_tensor, mean, std):
        mean = mean.view(3, 1, 1)  
        std = std.view(3, 1, 1)    
        return img_tensor * std + mean

    if isinstance(image, torch.Tensor):
        image = denormalize(image, mean, std)
        image = image.permute(1, 2, 0).numpy() 

    image = (image * 255).astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if center_points is not None and words is not None:
        for i, (center_point, word) in enumerate(zip(center_points, words)):
            x, y = center_point
            x *= image.shape[1]
            y *= image.shape[0]
            cv2.circle(image, (int(x), int(y)), radius=5, color=(255, 255, 0), thickness=1)
            
            text_position = (int(x), int(y) + 15)  
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            color = (0, 0, 0)  
            thickness = 1

            text_size = cv2.getTextSize(word, font, font_scale, thickness)[0]
            text_width, text_height = text_size

            background_top_left = (text_position[0], text_position[1] - text_height)
            background_bottom_right = (text_position[0] + text_width, text_position[1] + 5)
            background_color = (255, 255, 255)
            cv2.rectangle(image, background_top_left, background_bottom_right, background_color, cv2.FILLED)

            cv2.putText(image, word, text_position, font, font_scale, color, thickness, lineType=cv2.LINE_AA)        

    cv2.imwrite(os.path.join(output_folder, filename), image)