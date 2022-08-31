'''
Author: Derek Ka Kin Lau, Ringo Chu
Usage:

1. Set "export PYTHONPATH=$PWD" at root folder, the folder that has .git
2. At root folder

    python evaluation\img_object_detection.py \
                        --data_path <DATA> \
                        --model_path <FOLDER>\<PARAM>.pth \
                        --device cpu \
                        --debug True

3. Any questions go to Ringo, do not bother to ask
'''
from __future__ import absolute_import, division, print_function

import torch, torchvision
import cv2

from model.helper.parser import GeneralParser
from model.helper.utility import cpu_or_gpu, add_bbox
import model.helper.vision.utils as util

from dataset.vidvrddataset import ObjectDetectVidVRDDataset

import os
from tqdm import tqdm
import numpy as np

# TODO: Derek and Winson Please get it Done
# Description: Single Image Detection
# Save as json/txt format
# https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

def evaluate_COCO(dataloader, model, device):
    model.to(device)
    model.eval()

    for img, blob in dataloader:

        img, blob = img[0].to(device), blob[0]
        with torch.no_grad():
            inference = model(torch.unsqueeze(img, 0))

        # Derek/Winson Please carry on from here......

def _visualise_single(img, gt_box=None, gt_label=None, pred_box=None, pred_label=None, colour=None):

    img = img
    if gt_box is not None:
        for c, b in zip(gt_label, gt_box):
            img = add_bbox(img,
                           int(round(b[0])), int(round(b[1])),
                           int(round(b[2])), int(round(b[3])),
                           color='navy')
    if pred_box:
        pass

    return img

def visualise(datasetloader, model, device='cpu'):

    model.to(device)
    model.eval()

    for index, (img, blob) in enumerate(tqdm(datasetloader)):
        img, blob = img[0].to(device), blob[0]


        with torch.no_grad():
            inference = model(torch.unsqueeze(img, 0))

        print(inference)

        gt_boxes, gt_labels = blob['boxes'], blob['labels']
        gt_boxes, gt_labels = list(gt_boxes.cpu().detach().numpy()), list(gt_labels.cpu().detach().numpy())

        img = img.cpu().numpy().transpose(1, 2, 0)
        img = img[:, :, ::-1]
        img = cv2.UMat(img).get()
        img = _visualise_single(img, gt_boxes, gt_labels)

        cv2.imshow('bb_visualise', img)
        cv2.waitKey(20)

def get_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.40
    return model

def main(parse_options):
    testset_detection = ObjectDetectVidVRDDataset(data_path=parse_options.data_path,
                                                   set='test',
                                                   transforms=None)

    device = parse_options.device
    checkpoint = torch.load(parse_options.model_path, map_location=cpu_or_gpu(device))

    fasterrcnn_resnet50_fpn = get_detection_model(testset_detection.get_num_classes()+1)
    fasterrcnn_resnet50_fpn.load_state_dict(checkpoint['model_state_dict'])
    fasterrcnn_resnet50_fpn.to(device)

    if parse_options.debug:
        print('DEBUGGING')
        indices = torch.randperm(len(testset_detection)).tolist()
        testset_detection = torch.utils.data.Subset(testset_detection, indices[:50])

    dataloader_test = torch.utils.data.DataLoader(testset_detection,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=2,
                                              collate_fn=util.collate_fn)

    #evaluate_COCO(dataloader_test, fasterrcnn_resnet50_fpn, device)
    visualise(dataloader_test, torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True), device)

if __name__ =='__main__':

    parse = GeneralParser()
    parse_options = parse.parse()

    main(parse_options)
