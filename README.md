# DEER

## Introduction

This repository is a PyTorch implementation of the paper [DEER: Detection-agnostic End-to-End Recognizer for Scene Text Spotting](https://arxiv.org/pdf/2203.05122)

Part of the code is inherited from

Backbone : [Vovnet](https://github.com/youngwanLEE/vovnet-detectron2?tab=readme-ov-file)

LocationHead : [Real-time Scene Text Detection with Differentiable Binarization](https://github.com/MhLiao/DB/tree/master)

Transformer Encoder-Decoder : [Deformable Detr](https://github.com/fundamentalvision/Deformable-DETR)

## Result

![result_1.jpg](figures/result_1.jpg)

![result_2.jpg](figures/result_2.jpg)

![result_3.jpg](figures/result_3.jpg)

| F1 Score(E2E) | Our (EM) | Paper |
| --- | --- | --- |
| ICDAR15 | 62.92 | 71.72 |

We tested exact match without any preprocessing on the label.

*In the paper, they remove words shorter than 3 characters and special characters at the beginning or end of words.

## Installation

### Requirements:

- Following the basic requirements of Deformable DETR

We used under environments

- torch==2.2.0
- torchvision==0.17.0
- torchaudio==2.2.0
- lightning==2.3.3
- omegaconf==2.3.0
- munkres==1.1.4
- pyclipper==1.3.0
- albumentations==1.4.13
- opencv-python==4.10.0.84

```jsx
pip install -r requirements.txt
```

### Backbone
You can download backbone model(vovnet39) [HERE](https://drive.google.com/file/d/1KX-2HxAub777qxgLub-IeB41i8AWRrXR/view?usp=sharing)

## Dataset

```jsx
ICDAR15/
├── test
│     ├── images
│     └── labels
│         └── coco_annotations_test.json
└── train
         ├── images
         └── labels
             └── coco_annotations_test.json
                                    
TextOCR/
├── test
├── train
├── TextOCR_train.json
└── TextOCR_val.json
```

## Training
```
python main.py —config {config_path}
```
### finetune or continue
```
python main.py —config {config.path} —weights {weight.path}
```
## Evaluation
```
python main.py —config {config.path} —weights {weight.path} —test
```

