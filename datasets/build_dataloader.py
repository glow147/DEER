import torch
import lightning as L
from .TextOCR import TextOCR_Dataset
from .ICDAR import ICDAR_Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2


def custom_valid_collate_fn(batch):
    images = []
    bboxes = []
    labels = []
    filenames = []
    ori_sizes = []

    for item in batch:
        if item is None:
            continue
        images.append(item['image'])
        bboxes.append((item['bboxes']))
        labels.append(item['label'])
        filenames.append(item['filename'])
        ori_sizes.append(item['ori_size'])

    return {
        'image': torch.stack(images),
        'bboxes': bboxes,
        'label': labels,
        'filename': filenames,
        'ori_size': ori_sizes,
    }


def custom_collate_fn(batch):
    images = []
    gts = []
    masks = []
    bboxes = []
    thresh_maps = []
    thresh_masks = []
    prefs_noise = []
    labels, filenames, ori_sizes = [], [], []
    for item in batch:
        if item is None:
            continue
        images.append(item['image'])
        prefs_noise.append(item['pref_noise'])
        gts.append(item['gts'])
        bboxes.append((item['bboxes']))
        masks.append(item['masks'])
        labels.append(item['label'])
        filenames.append(item['filename'])
        ori_sizes.append(item['ori_size'])
        thresh_masks.append(item['thresh_mask'])
        thresh_maps.append(item['thresh_map'])

    return {
        'image': torch.stack(images),
        'pref_noise': pad_sequence(prefs_noise, batch_first=True, padding_value=float('-inf')),
        'gts': torch.stack(gts),
        'masks': torch.stack(masks),
        'thresh_mask': torch.stack(thresh_masks),
        'thresh_map': torch.stack(thresh_maps),
        'bboxes': bboxes,
        'label': labels,
        'filename': filenames,
        'ori_size': ori_sizes,
    }


def load_dataset(name, *args, **kwargs):
    if name == 'TextOCR':
        return TextOCR_Dataset(*args, **kwargs)
    elif name == 'ICDAR15':
        return ICDAR_Dataset(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {name}")


class OCRDataModule(L.LightningDataModule):
    def __init__(self, config) -> None:
        super().__init__()
        self.cfg = config

        self.transform = A.Compose([
            A.SafeRotate(),
            A.RandomScale(scale_limit=(0.5, 3.0), p=1),
            A.RandomSizedBBoxSafeCrop(
                height=self.cfg.DATA.IMAGE_SIZE, width=self.cfg.DATA.IMAGE_SIZE, p=1),
            A.ColorJitter(p=0.8),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

        self.val_transform = A.Compose([
            A.LongestMaxSize(max_size=self.cfg.DATA.VALID_IMAGE_SIZE),
            A.Resize(
                height=int(self.cfg.DATA.VALID_IMAGE_SIZE * 3 / 4),
                width=self.cfg.DATA.VALID_IMAGE_SIZE
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    def setup(self, stage: str):
        if stage == 'fit':
            self.train_dataset = load_dataset(self.cfg.DATA.NAME, self.cfg, self.transform, is_train=True)
            self.val_dataset = load_dataset(self.cfg.DATA.NAME, self.cfg, self.val_transform, is_train=False)

        if stage == 'test':
            self.test_dataset = load_dataset(self.cfg.DATA.NAME, self.cfg, self.val_transform, is_train=False)




    def train_dataloader(self):  # -> DataLoader:
        data_loader = DataLoader(self.train_dataset, batch_size=self.cfg.MODEL.PARAMS.BATCH_SIZE,
                                 shuffle=True, num_workers=16, collate_fn=custom_collate_fn)
        return data_loader

    def val_dataloader(self):
        data_loader = DataLoader(self.val_dataset, batch_size=self.cfg.MODEL.PARAMS.BATCH_SIZE // 2, 
                                 shuffle=False, num_workers=16, collate_fn=custom_valid_collate_fn)
        return data_loader
    
    def test_dataloader(self):
        data_loader = DataLoader(self.test_dataset, batch_size=self.cfg.MODEL.PARAMS.BATCH_SIZE // 2, 
                                 shuffle=False, num_workers=16, collate_fn=custom_valid_collate_fn)
        return data_loader
